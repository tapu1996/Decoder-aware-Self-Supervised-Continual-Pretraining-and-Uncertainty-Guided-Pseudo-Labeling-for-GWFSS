import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))
from decoder import MMSeg_FCN_Decoder, FPNDecoder, CustomEncoder_FPN

class DINOHead(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Conv2d(in_dim, bottleneck_dim, 1)
        else:
            layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, temp=0.07, eps=1e-6, verbose=False):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.temp = temp
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)
        self.verbose = verbose

    def forward(self, x):
        x_prev = x

        slots = self.slot_embed(torch.arange(0, self.num_slots, device=x.device)).unsqueeze(0).repeat(x.size(0), 1, 1)

        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(x, dim=1))

        attn = (dots / self.temp).softmax(dim=1) + self.eps

        slots = torch.einsum('bdhw,bkhw->bkd', x_prev, attn / attn.sum(dim=(2, 3), keepdim=True))

        return slots, dots

class SlotCon(nn.Module):
    def __init__(self, encoder, args, use_decoder=False, verbose=False):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out
        assert args.decoder_downstream_dataset in ["cityscapes", "ade20k"], (
            "Only Cityscapes and ADE20K configs are supported for the FCN decoder for now."
        )
        if args.decoder_type == "FCN":

            if args.decoder_downstream_dataset == "cityscapes":
                decoder_args = dict(
                    out_channels = 256,
                    padding = 6,
                    dilation = 6,
                )
            else:
                decoder_args = dict(
                    out_channels = 512,
                    padding = 1,
                    dilation = 1,
                )
        else:
            decoder_args = dict(
                out_channels = 256,
            )
        self.use_decoder = use_decoder
        self.decoder_type = args.decoder_type
        assert self.decoder_type in ["FPN", "FCN"], "Only FPN and FCN decoders are supported for now."
        self.teacher_momentum = args.teacher_momentum

        if args.arch in ('resnet18', 'resnet34'):
            self.num_channels = 512
        elif args.arch in ('resnet50'):
            self.num_channels = 2048
        elif args.arch in ('convnext_tiny', 'convnext_small'):
            self.num_channels = 768
        elif args.arch in ('convnext_base'):
            self.num_channels = 1024
        elif args.arch in ('convnext_large','convnextv2_large'):
            self.num_channels = 1536
        elif args.arch in ('convnext_xlarge'):
            self.num_channels = 2048
        elif args.arch in ('convnextv2_huge'):
            self.num_channels = 2816
        else:
            raise ValueError("Untested architecture. We need to figure out the number of channels.")

        if self.use_decoder and self.decoder_type == "FPN":
            head_type = 'multi_layer'
        else:
            # Normal Slotcon case and for FCN decoder 
            head_type = 'early_return'
        self.encoder_q = encoder(head_type=head_type)
        self.encoder_k = encoder(head_type=head_type)
        self.verbose = verbose

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.use_decoder:
            if self.decoder_type == "FPN":
                if not args.no_ddp:
                    norm =  "SyncBN" 
                else:
                    norm = "BN" 
                self.decoder_k = FPNDecoder( **dict(
                    bottom_up=CustomEncoder_FPN(self.encoder_k, arch=args.arch),
                    top_block=None, 
                    only_keep_last_output_conv= (not args.decoder_deep_supervision),
                    in_features=('res2', 'res3', 'res4', 'res5'),
                    out_channels=decoder_args["out_channels"],
                    norm=norm, 
                    sk_dropout_prob=args.sk_dropout_prob, 
                    sk_channel_dropout_prob=args.sk_channel_dropout_prob)
                )
                self.decoder_q = FPNDecoder(**dict(
                    bottom_up=CustomEncoder_FPN(self.encoder_q, arch=args.arch),
                    top_block=None,
                    only_keep_last_output_conv= (not args.decoder_deep_supervision),
                    in_features=('res2', 'res3', 'res4', 'res5'),
                    out_channels=decoder_args["out_channels"],
                    norm=norm, 
                    sk_dropout_prob=args.sk_dropout_prob, 
                    sk_channel_dropout_prob=args.sk_channel_dropout_prob)
                )
            else: # FCN
                self.decoder_k = MMSeg_FCN_Decoder(in_channels=self.num_channels, **decoder_args)
                self.decoder_q = MMSeg_FCN_Decoder(in_channels=self.num_channels, **decoder_args)

            for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        
        if not args.no_ddp:
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
            if self.use_decoder:
                if self.decoder_type == "FCN":
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder_q)
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.decoder_k)

        self.group_loss_weight = args.group_loss_weight
        self.group_loss_weight_dec = args.group_loss_weight_dec
        self.encoder_loss_weight = args.encoder_loss_weight
        self.deep_supervision = False
        if self.use_decoder:
            assert self.encoder_loss_weight != 1, "Encoder loss weight cannot be 1 when using decoder."
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
        
        if self.encoder_loss_weight > 0.0:
            self.projector_q = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
            self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)

            for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        if self.use_decoder:
            if args.decoder_deep_supervision:
                self.deep_supervision = True
                self.dec_projector_q = nn.ModuleList()
                self.dec_projector_k = nn.ModuleList()
                for i in range(4):
                    self.dec_projector_q.append(DINOHead2d(decoder_args["out_channels"], hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out))
                    self.dec_projector_k.append(DINOHead2d(decoder_args["out_channels"], hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out))
                    for param_q, param_k in zip(self.dec_projector_q[-1].parameters(), self.dec_projector_k[-1].parameters()):
                        param_k.data.copy_(param_q.data)
                        param_k.requires_grad = False
            else:
                self.dec_projector_k = DINOHead2d(decoder_args["out_channels"], hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
                self.dec_projector_q = DINOHead2d(decoder_args["out_channels"], hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
                for param_q, param_k in zip(self.dec_projector_q.parameters(), self.dec_projector_k.parameters()):
                    param_k.data.copy_(param_q.data)
                    param_k.requires_grad = False            
            
        if not args.no_ddp:
            if self.encoder_loss_weight > 0.0:
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
            if self.use_decoder:
                nn.SyncBatchNorm.convert_sync_batchnorm(self.dec_projector_q)
                nn.SyncBatchNorm.convert_sync_batchnorm(self.dec_projector_k)

        self.num_prototypes = args.num_prototypes
        self.num_prototypes_dec = args.num_prototypes_dec
        if self.num_prototypes_dec==0:
            self.num_prototypes_dec = args.num_prototypes
        if self.encoder_loss_weight > 0.0:
            self.center_momentum = args.center_momentum
            self.register_buffer("center", torch.zeros(1, self.num_prototypes))
            self.grouping_q = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp, verbose=self.verbose)
            self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out, self.teacher_temp, verbose=self.verbose)
            self.predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
            if not args.no_ddp:
                nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor_slot)
            for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        if self.use_decoder:
            self.center_momentum_dec = args.center_momentum
            if args.decoder_deep_supervision:
                self.dec_grouping_q = nn.ModuleList()
                self.dec_grouping_k = nn.ModuleList()
                self.dec_predictor_slot = nn.ModuleList()
                # TODO: change this hard coded 4 to be dependent on decoder
                for i in range(4):
                    self.register_buffer(f"center_dec_{i}", torch.zeros(1, self.num_prototypes_dec))
                    self.dec_grouping_q.append(SemanticGrouping(self.num_prototypes_dec, self.dim_out, self.teacher_temp, verbose=self.verbose))
                    self.dec_grouping_k.append(SemanticGrouping(self.num_prototypes_dec, self.dim_out, self.teacher_temp, verbose=self.verbose))
                    self.dec_predictor_slot.append(DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out))
                    if not args.no_ddp:
                        nn.SyncBatchNorm.convert_sync_batchnorm(self.dec_predictor_slot[-1])
                    for param_q, param_k in zip(self.dec_grouping_q[-1].parameters(), self.dec_grouping_k[-1].parameters()):
                        param_k.data.copy_(param_q.data)
                        param_k.requires_grad = False
            else:
                self.register_buffer("center_dec", torch.zeros(1, self.num_prototypes_dec))
                self.dec_grouping_q = SemanticGrouping(self.num_prototypes_dec, self.dim_out, self.teacher_temp, verbose=self.verbose)
                self.dec_grouping_k = SemanticGrouping(self.num_prototypes_dec, self.dim_out, self.teacher_temp, verbose=self.verbose)
                self.dec_predictor_slot = DINOHead(self.dim_out, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
                if not args.no_ddp:
                    nn.SyncBatchNorm.convert_sync_batchnorm(self.dec_predictor_slot)
                for param_q, param_k in zip(self.dec_grouping_q.parameters(), self.dec_grouping_k.parameters()):
                    param_k.data.copy_(param_q.data)  # initialize
                    param_k.requires_grad = False  # not update by gradient

        self.K = int(args.num_instances * 1. / args.world_size / args.batch_size * args.epochs)
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))
        self.args = args
        if not self.args.no_ddp:
            self.rank_for_labels = torch.distributed.get_rank()
            self.ws = dist.get_world_size()
            self.ddp = True
        else:
            self.ddp = False
            self.rank_for_labels = 0
            self.ws = 1

    def re_init(self, args):
        self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        if self.encoder_loss_weight > 0.0:
            for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
            for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)  
            
    @torch.no_grad()
    def _momentum_update_key_decoder(self):
        """
        Momentum update of the key decoder
        """
        momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
        self.k += 1
        for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
            param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        if self.deep_supervision:
            for dec_projq, dec_projk  in zip(self.dec_projector_q, self.dec_projector_k):
                for param_q, param_k in zip(dec_projq.parameters(), dec_projk.parameters()):
                    param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
        else:
            for param_q, param_k in zip(self.dec_projector_q.parameters(), self.dec_projector_k.parameters()):
                param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    def invaug(self, x, coords, flags):
        N, C, H, W = x.shape

        batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
        coords_rescaled = coords.clone()

        coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
        coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
        coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
        coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2

        coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)

        x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)

        x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])

        return x_flipped

    def self_distill(self, q, k, center):

        q = F.log_softmax(q / self.student_temp, dim=-1)

        k = F.softmax((k - center) / self.teacher_temp, dim=-1)

        return torch.sum(-k * q, dim=-1).mean()

    def ctr_loss_filtered(self, q, k, score_q, score_k, predictor_slot, tau=0.2):

        q = q.flatten(0, 1)
        k = F.normalize(k.flatten(0, 1), dim=1)

        mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        mask_intersection = (mask_q * mask_k).view(-1)
        idxs_q = mask_intersection.nonzero().squeeze(-1)

        mask_k = self.concat_all_gather(mask_k.view(-1))
        idxs_k = mask_k.nonzero().squeeze(-1)

        N = k.shape[0]
        logits = torch.einsum('nc,mc->nm', [F.normalize(predictor_slot(q[idxs_q]), dim=1), self.concat_all_gather(k)[idxs_k]]) / tau
        labels = mask_k.cumsum(0)[idxs_q + N * self.rank_for_labels] - 1
        return F.cross_entropy(logits, labels) * (2 * tau)

    def forward(self, input):
        crops, coords, flags = input

        # all_enc_features_x1 is a tuple returned by Slotcon resnet backbone in FPN setting
        # Otherwise it is a single tensor
        all_enc_features_x1 = self.encoder_q(crops[0])
        all_enc_features_x2 = self.encoder_q(crops[1])

        if self.use_decoder and self.decoder_type == "FPN":
            enc_x1, enc_x2 = all_enc_features_x1[-1], all_enc_features_x2[-1]
        else:
            enc_x1, enc_x2 = all_enc_features_x1, all_enc_features_x2
        
        if self.encoder_loss_weight > 0.0:
            x1, x2 = self.projector_q(enc_x1), self.projector_q(enc_x2)

        if self.use_decoder:
            # tuple in case of FPN, else a single tensor
            out_dec_x1 = self.decoder_q(all_enc_features_x1)
            out_dec_x2 = self.decoder_q(all_enc_features_x2)
            if self.decoder_type == "FPN":
                out_dec_x1, out_dec_x2 = self.select_features_for_slotcon_criteria(
                    out_dec_x1, out_dec_x2
                    )
            if self.deep_supervision:
                dec_x1 = [proj(out_dec_x) for proj, out_dec_x in zip(self.dec_projector_q, out_dec_x1)]
                dec_x2 = [proj(out_dec_x) for proj, out_dec_x in zip(self.dec_projector_q, out_dec_x2)]
            else: 
                dec_x1 = self.dec_projector_q(out_dec_x1)
                dec_x2 = self.dec_projector_q(out_dec_x2)

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder + semantic grouping
            all_enc_features_y1 = self.encoder_k(crops[0])
            all_enc_features_y2 = self.encoder_k(crops[1])
            if self.decoder_type == "FPN":
                enc_y1, enc_y2 = all_enc_features_y1[-1], all_enc_features_y2[-1]
            else:
                enc_y1, enc_y2 = all_enc_features_y1, all_enc_features_y2
            
            if self.encoder_loss_weight > 0.0:
                y1, y2 = self.projector_k(enc_y1), self.projector_k(enc_y2)

            if self.use_decoder:
                self._momentum_update_key_decoder() # update the key decoder
                out_dec_y1 = self.decoder_k(all_enc_features_y1)
                out_dec_y2 = self.decoder_k(all_enc_features_y2)
                if self.decoder_type == "FPN":
                    out_dec_y1, out_dec_y2 = self.select_features_for_slotcon_criteria(
                        out_dec_y1, out_dec_y2
                        )
                if self.deep_supervision:
                    dec_y1 = [proj(out_dec_y) for proj, out_dec_y in zip(self.dec_projector_k, out_dec_y1)]
                    dec_y2 = [proj(out_dec_y) for proj, out_dec_y in zip(self.dec_projector_k, out_dec_y2)]
                else: 
                    dec_y1 = self.dec_projector_k(out_dec_y1)
                    dec_y2 = self.dec_projector_k(out_dec_y2)
        if self.encoder_loss_weight > 0.0:
            group_loss_enc, contrastive_loss_enc, score_k1, score_k2 = self.slotcon_criteria(
                x1, x2, y1, y2, coords, flags, self.group_loss_weight, self.center,
                self.grouping_q, self.grouping_k, self.predictor_slot
                )
            self.center = self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2), self.center, self.center_momentum)
        else:
            group_loss_enc = 0.0
            contrastive_loss_enc = 0.0

        if self.use_decoder:
            if self.deep_supervision:
                group_loss_dec = 0.0
                contrastive_loss_dec = 0.0
                logs = {}
                # TODO: change this hard coded 4 to be dependent on decoder
                for i in range(4):
                    group_loss_dec_i, contrastive_loss_dec_i, score_k1_dec_i, score_k2_dec_i = self.slotcon_criteria(
                        dec_x1[i], dec_x2[i], dec_y1[i], dec_y2[i], coords, flags, self.group_loss_weight_dec, getattr(self, f"center_dec_{i}"),
                        self.dec_grouping_q[i], self.dec_grouping_k[i], self.dec_predictor_slot[i]
                        )
                    setattr(self, f"center_dec_{i}", self.update_center(torch.cat([score_k1_dec_i, score_k2_dec_i]).permute(0, 2, 3, 1).flatten(0, 2), getattr(self, f"center_dec_{i}"), self.center_momentum_dec))
                    group_loss_dec += group_loss_dec_i
                    contrastive_loss_dec += contrastive_loss_dec_i
                    logs.update({   
                        f"gld_{i+1}": group_loss_dec_i,
                        f"cld_{i+1}": contrastive_loss_dec_i
                        })

                contrastive_loss_dec /= 4
                group_loss_dec /= 4
                encoder_loss = contrastive_loss_enc + group_loss_enc
                decoder_loss = contrastive_loss_dec + group_loss_dec
                total_loss = self.encoder_loss_weight * encoder_loss + (1 - self.encoder_loss_weight) * decoder_loss
                logs.update({
                    "total_loss": total_loss,
                    "encoder_loss": encoder_loss, 
                    "decoder_loss": decoder_loss, 
                    "group_loss_enc": group_loss_enc,
                    "group_loss_dec": group_loss_dec,
                    "contrastive_loss_enc": contrastive_loss_enc,
                    "contrastive_loss_dec": contrastive_loss_dec
                    })
            else:
                group_loss_dec, contrastive_loss_dec, score_k1_dec, score_k2_dec = self.slotcon_criteria(
                    dec_x1, dec_x2, dec_y1, dec_y2, coords, flags, self.group_loss_weight_dec, self.center_dec,
                    self.dec_grouping_q, self.dec_grouping_k, self.dec_predictor_slot
                    )
                self.center_dec = self.update_center(torch.cat([score_k1_dec, score_k2_dec]).permute(0, 2, 3, 1).flatten(0, 2), self.center_dec, self.center_momentum_dec)

                encoder_loss = contrastive_loss_enc + group_loss_enc
                decoder_loss = contrastive_loss_dec + group_loss_dec
                total_loss = self.encoder_loss_weight * encoder_loss + (1 - self.encoder_loss_weight) * decoder_loss
                logs = {
                    "total_loss": total_loss,
                    "encoder_loss": encoder_loss, 
                    "decoder_loss": decoder_loss, 
                    "group_loss_enc": group_loss_enc,
                    "group_loss_dec": group_loss_dec,
                    "contrastive_loss_enc": contrastive_loss_enc,
                    "contrastive_loss_dec": contrastive_loss_dec
                    }
        else:
            total_loss = contrastive_loss_enc + group_loss_enc
            logs = {
                "total_loss": total_loss,
                "group_loss": group_loss_enc, 
                "contrastive_loss": contrastive_loss_enc,
                }

        return total_loss, logs
    
    def slotcon_criteria(self, x1, x2, y1, y2, coords, flags, group_loss_weight, center, grouping_q, grouping_k, predictor_slot):
        """
        SlotCon criteria.
        Wrapping this inside a function because it will be used at different levels.
        """

        (q1, score_q1), (q2, score_q2) = grouping_q(x1), grouping_q(x2)
        q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
        with torch.no_grad():
            (k1, score_k1), (k2, score_k2) = grouping_k(y1), grouping_k(y2)
            k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])

        group_loss = group_loss_weight * self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2), center) \
             + group_loss_weight * self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2), center)
        
        contrastive_loss = (1. - group_loss_weight) * self.ctr_loss_filtered(q1, k2, score_q1, score_k2, predictor_slot) \
              + (1. - group_loss_weight) * self.ctr_loss_filtered(q2, k1, score_q2, score_k1, predictor_slot)
        
        return group_loss, contrastive_loss, score_k1, score_k2
    
    def select_features_for_slotcon_criteria(self, x1, x2):
        """
        Selecting the features for SlotCon criteria.
        """
        assert self.decoder_type == "FPN", "Only FPN decoder is supported for now."
        if self.deep_supervision:          
            x1 = [x1["p2"], x1["p3"], x1["p4"], x1["p5"]]
            x2 = [x2["p2"], x2["p3"], x2["p4"], x2["p5"]]
            return x1, x2
        else:
            x1 = x1["p2"]
            x2 = x2["p2"]
            return x1, x2   
 
    @torch.no_grad()
    def update_center(self, teacher_output, center, center_momentum):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if self.ddp:
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * self.ws)

        # ema update
        center = center * center_momentum + batch_center * (1 - center_momentum)
        return center

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.ddp:
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(self.ws)]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
            return output
        else:
            return tensor

class SlotConEval(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()

        self.dim_hidden = args.dim_hidden
        self.dim_out = args.dim_out

        self.num_channels = 512 if args.arch in ('resnet18', 'resnet34') else 2048
        self.encoder_k = encoder(head_type='early_return')
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.projector_k = DINOHead2d(self.num_channels, hidden_dim=self.dim_hidden, bottleneck_dim=self.dim_out)
        for param_k in self.projector_k.parameters():
            param_k.requires_grad = False  # not update by gradient

        self.num_prototypes = args.num_prototypes
        self.grouping_k = SemanticGrouping(self.num_prototypes, self.dim_out)
        for param_k in self.grouping_k.parameters():
            param_k.requires_grad = False  # not update by gradient

    def forward(self, x):
        with torch.no_grad():
            slots, probs = self.grouping_k(self.projector_k(self.encoder_k(x)))
            return probs