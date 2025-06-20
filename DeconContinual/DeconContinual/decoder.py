import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool
from detectron2.modeling.backbone.backbone import Backbone



## For MMseg segmentation task
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.activate = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activate(x)
        return x

class MMSeg_FCN_Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation):
        super().__init__()
        
        
        self.convs = nn.Sequential(
            ConvModule(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False),
            ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=False)
        )
    def forward(self, x):
        return self.convs(x)

## For detectron2 Object Detection and Instance Segmentation
class CustomEncoder_FPN(Backbone):
    def __init__(self, encoder:nn.Module, arch:str="resnet50"):
        super().__init__()
        self.encoder = encoder
        self._out_features = ['res2', 'res3', 'res4', 'res5']
        if arch == "resnet50":
            self._out_feature_channels = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        elif arch in ["convnext_tiny", "convnext_small"]:
            self._out_feature_channels = {'stem': 24, 'res2': 96, 'res3': 192, 'res4': 384, 'res5': 768}
        elif arch == "convnext_base":
            self._out_feature_channels = {'stem': 32, 'res2': 128, 'res3': 256, 'res4': 512, 'res5': 1024}
        elif arch == "convnext_large":
            self._out_feature_channels = {'stem': 48, 'res2': 192, 'res3': 384, 'res4': 768, 'res5': 1536}
        elif arch == "convnext_xlarge":
            self._out_feature_channels = {'stem': 64, 'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        elif arch == "convnextv2_large":
            self._out_feature_channels = {'stem': 48, 'res2': 192, 'res3': 384, 'res4': 768, 'res5': 1536}
        elif arch == "convnextv2_huge":
            self._out_feature_channels = {'stem': 64, 'res2': 352, 'res3': 704, 'res4': 1408, 'res5': 2816}
        else:
            raise ValueError(f"Unknown architecture: {arch} to build an FPN decoder.")
        self._out_feature_strides = {'stem': 4, 'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        
    def forward(self, x):
        return self.encoder(x)

class FPNDecoder(FPN):
    def __init__(self, 
                 only_keep_last_output_conv=True, top_block=LastLevelMaxPool(), 
                 fuse_type="sum", square_pad=0, sk_dropout_prob:float=0.,
                 sk_channel_dropout_prob:float=0., **kwargs):
        kwargs['top_block'] = top_block
        kwargs['fuse_type'] = fuse_type
        kwargs['square_pad'] = square_pad

        # Dropout definition
        self.sk_dropout_prob = sk_dropout_prob
        self.sk_channel_dropout_prob = sk_channel_dropout_prob
        if sk_channel_dropout_prob != 0.0 or sk_dropout_prob != 0.0:
            assert sk_channel_dropout_prob == 0.0 or sk_dropout_prob == 0.0, "Only one type of dropout can be used at a time."

        super().__init__(**kwargs)

        if self.sk_channel_dropout_prob > 0.0:
            self.channel_dropout = nn.Dropout2d(p=self.sk_channel_dropout_prob)

        if only_keep_last_output_conv:
            # We only neeed lateral convs to get the final decoder output
            # If you need all level output, do not set the boolean to True
            len_out_conv = len(self.output_convs)
            temp = [None for _ in range(len_out_conv - 1)]
            temp.extend([self.output_convs[-1]])
            self.output_convs = temp
            del self.fpn_output3, self.fpn_output4, self.fpn_output5
        # Removing the encoder from the FPN class since it will be used separately
        del self.bottom_up

    def forward(self, x):
        """
        Instead of computing the features with the bachbone encoder, this overwritten forward method
        will only compute the decoder features.
        x is assumed to be the output of the backbone encoder, whether it returns a single feature map 
        or a list of feature maps.
        """

        #### In the true FPN class we have the following line of code:
        # bottom_up_features = self.bottom_up(x)
        #### We replace it by:
        res2, res3, res4, res5 = x 
        bottom_up_features = {'res2': res2, 'res3': res3, 'res4': res4, 'res5': res5}
        ####

        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[self.in_features[-1]])
        if self.output_convs[0] is not None:
            results.append(self.output_convs[0](prev_features))
        else:
            results.append(None)

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = self.in_features[-idx - 1]
                features = bottom_up_features[features]
                top_down_features = F.interpolate(prev_features, scale_factor=2.0, mode="nearest")
                lateral_features = lateral_conv(features)
                if self.sk_channel_dropout_prob > 0.0:
                    # Droping out randomly some channels in the lateral features, separately for each sample in the batch.                    
                    prev_features = self.channel_dropout(lateral_features) + top_down_features
                else:
                    # Droping out the lateral features depending on a random number.
                    # If probability of dropout is 0, then the random number is always >= 0 and no dropout will be 
                    # applied (no_dropout=1) and we use lateral features.
                    # If probability of dropout is 0.25, then the random number is greater than 0.25, 75% of the 
                    # time and then no_dropout is applied 75% of the time => dropout applied 25% of the time.
                    no_dropout = torch.rand(1, device=prev_features.device) >= self.sk_dropout_prob
                    prev_features = lateral_features * no_dropout.float() + top_down_features
                
                if self._fuse_type == "avg" and no_dropout:
                    prev_features /= 2
                if output_conv is not None:
                    results.insert(0, output_conv(prev_features))
                else:
                    results.insert(0, None)

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            if top_block_in_feature is not None:
                results.extend(self.top_block(top_block_in_feature))
            else:
                results.extend([None])
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
