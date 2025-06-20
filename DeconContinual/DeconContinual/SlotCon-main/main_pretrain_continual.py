import argparse
import json
import os
import random
import shutil
import time
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from data.datasets import ImageFolder
from data.transforms import CustomDataAugmentation

from models import resnet
from models.slotcon import SlotCon
from utils.lars import LARS
from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.util import AverageMeter
from collections import OrderedDict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))
from distributed import _init_distributed_mode_computecan
import convnext 
import convnextv2
model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))
model_names += sorted(name for name in convnext.__all__ if name.islower() and callable(convnext.__dict__[name]))
model_names += sorted(name for name in convnextv2.__all__ if name.islower() and callable(convnextv2.__dict__[name]))

def get_parser():
    parser = argparse.ArgumentParser('SlotCon')

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO', 'COCOplus', 'ImageNet', 'gwfss'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
   
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names, help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--num-prototypes-dec', type=int, default=0, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--use-decoder', action='store_true', help='use decoder or not')
    parser.add_argument('--group-loss-weight-dec', default=0.5, type=float, help='balancing weight of the grouping loss for decoder')
    parser.add_argument('--encoder-loss-weight', default=0.5, type=float, help='balancing weight of the encoder loss when there is a decoder')
    parser.add_argument(
        '--decoder-downstream-dataset', 
        type=str, 
        default="cityscapes", 
        help='dataset used in downstream task, it influences the config of the decoder'
    )
    parser.add_argument(
        '--decoder-type', 
        type=str, 
        default="FCN", 
        help='config for decoder for now only supports FCN|FPN.'
    )
    parser.add_argument(
        '-dds',
        '--decoder-deep-supervision', 
        action='store_true',
        help='Use a loss at each level of the decoder.'
    )
    parser.add_argument(    
        "-skdp",
        '--sk-dropout-prob', 
        type=float, 
        default=0.0, 
        help='Probability of not using the lateral features for each level of the decoder'
    )
    parser.add_argument(
        "-skcdp",
        '--sk-channel-dropout-prob', 
        type=float, 
        default=0.0, 
        help='Probability of droping channels in the lateral features for each level of the decoder'
    )

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars', 'adamW'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('-ps', '--previous-scheduler', action='store_true', 
                        help="Do you want to use the old scheduler. Old one was updated to avoid warnings.")
    
    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')

    # ddp
    parser.add_argument('--no-ddp', action='store_true', help='are you using DDP?')
    parser.add_argument('--cc', action='store_true', help='are you running on Compute Canada?')



    parser.add_argument('--freeze-encoder', action='store_true', help='freezing encoder for continual?')
    parser.add_argument('--resume-new', action='store_true', help='Continual? Start from epoch 1?')
    parser.add_argument('--copy-encoder', action='store_true', help='Copying encoder q to k?')
    parser.add_argument('--load-otherweightsThanSlotcon', action='store_true', help='Loading weights other than Slotcon model?')



    args = parser.parse_args()
    if not args.no_ddp:
        if args.cc:
            local_rank, rank, world_size = _init_distributed_mode_computecan()
            args.local_rank = local_rank
            args.rank = rank
            args.world_size = world_size
        else:
            if os.getenv("LOCAL_RANK", None)  is not None:
                args.local_rank = int(os.environ["LOCAL_RANK"])
            else:
                print("local rank is not defined, you better not be trying to use DDP!")
                args.local_rank = 0
    else:
        # Only one gpu or no GPU
        args.world_size = 1
    
            
    return args 

def build_model(args):
    if "resnet" in args.arch:
        encoder = resnet.__dict__[args.arch]
    elif "convnextv2" in args.arch:
        encoder = convnextv2.__dict__[args.arch]
    elif "convnext" in args.arch:
        encoder = convnext.__dict__[args.arch]
    
    else:
        raise NotImplementedError(f"Architecture {args.arch} not implemented")
    
    model = SlotCon(encoder, args, use_decoder=args.use_decoder).cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        optimizer = LARS(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'adamW':
        assert 'convnext' in args.arch, "AdamW optimizer should only be used for convnext."
        def get_params_groups(model):
            regularized = []
            not_regularized = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                # we do not regularize biases nor Norm parameters
                if name.endswith(".bias") or len(param.shape) == 1:
                    not_regularized.append(param)
                else:
                    regularized.append(param)
            return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
        params_groups = get_params_groups(model)
        optimizer = torch.optim.AdamW(
            params_groups,
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if not args.no_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)

    return model, optimizer


def save_checkpoint(args, epoch, model, optimizer, scheduler, scaler=None):
    logger.info('==> Saving... in {}'.format(args.output_dir))
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.fp16:
        state['scaler'] = scaler.state_dict()
    torch.save(state, os.path.join(args.output_dir, 'current.pth'))
    if epoch%50==0:
        torch.save(state, os.path.join(args.output_dir, f'epoch-{epoch}.pth'))

def load_checkpoint(args, model, optimizer, scheduler, scaler=None):
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')
        #print("Checkpoint keys:", checkpoint.keys())
        from collections import OrderedDict

        if args.load_otherweightsThanSlotcon:
            new_state_dict = OrderedDict()
            for k, v in checkpoint['model'].items():
                new_key = 'module.encoder_q.' + k
                new_state_dict[new_key] = v

            checkpoint['model'] = new_state_dict
            print("new checkpoint weight found")
            # for key in checkpoint['model'].keys():
            #     # print(key)

        if not args.resume_new:
            args.start_epoch = checkpoint['epoch'] + 1
        args.start_epoch = 1
        if not args.no_ddp:
            model.module.re_init(args)
        else:
            model.re_init(args)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'],strict=False)
        print("Missing keys:", missing_keys)
        # print("Unexpected keys:", unexpected_keys)
        # exit()
        if not args.resume_new:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

        if args.fp16 and not args.load_otherweightsThanSlotcon and 'scaler' in checkpoint:
            print("inside scaler")
            scaler.load_state_dict(checkpoint['scaler'])
        if args.copy_encoder:
            print("copying enc")
            for param_q, param_k in zip(model.module.encoder_q.parameters(), model.module.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False 
        if args.freeze_encoder:
            print("freezing enc")
            for name, param in model.module.encoder_q.named_parameters():
                param.requires_grad = False
                logger.info(f"Frozen layer: {name}")
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, args.start_epoch ))
        #exit()



    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # prepare data
    transform = CustomDataAugmentation(args.image_size, args.min_scale)
    train_dataset = ImageFolder(args.dataset, args.data_dir, transform)
    if not args.no_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler=None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model, optimizer = build_model(args)
    logger.info(model)

    # define scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    # define scaler
    if args.fp16:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')

    if args.resume:
        load_checkpoint(args, model, optimizer, scheduler, scaler)

    for epoch in range(args.start_epoch, args.epochs + 1):
        if not args.no_ddp:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, scheduler, epoch, args)

        if (epoch % args.save_freq == 0 or epoch == args.epochs):
            if not args.no_ddp:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model, optimizer, scheduler, scaler)
            else:
                save_checkpoint(args, epoch, model, optimizer, scheduler, scaler)


def train(train_loader, model, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    meter_created = False
    loss_meters = {}
    
    # switch to train mode
    model.train()

    end = time.time()
    train_len = len(train_loader)
    for i, batch in enumerate(train_loader):
        crops, coords, flags = batch
        crops = [crop.cuda(non_blocking=True) for crop in crops]
        coords = [coord.cuda(non_blocking=True) for coord in coords]
        flags = [flag.cuda(non_blocking=True) for flag in flags]

        # compute output and loss
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=(scaler is not None)):
            loss, logs = model((crops, coords, flags))
        
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        scheduler.step()

        # avg loss from batch size
        loss_meter.update(loss.item(), crops[0].size(0))
        if meter_created:
            for k,v in logs.items():
                loss_meters[k].update(v, crops[0].size(0))
        else:
            meter_created = True
            for k, v in logs.items():
                loss_meters[k] = AverageMeter()
                loss_meters[k].update(v, crops[0].size(0))
            
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - i)
            info = (f'Train: [{epoch}/{args.epochs}][{i}/{train_len}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.8f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  ')

            for k, v in loss_meters.items():
                info += f'{k} {v.val:.4f} ({v.avg:.4f})  '

            logger.info(info)


if __name__ == '__main__':
    args = get_parser()

    cudnn.benchmark = True

    if not args.no_ddp:
        if not args.cc:
            torch.cuda.set_device(args.local_rank)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if not args.cc:
            args.world_size = dist.get_world_size()
        args.batch_size = int(args.batch_size / args.world_size)
    else:
        args.rank = 0


    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.no_ddp:
        dist_rank_logger = dist.get_rank()
    else:
        dist_rank_logger = 0
    logger = setup_logger(output=args.output_dir,
                          distributed_rank=dist_rank_logger, name="slotcon")
    
    if dist_rank_logger == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )

    main(args)
