import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class LaneSegmentationModel(nn.Module):
    
    def __init__(self, encoder_name='resnet18', encoder_weights='imagenet', in_channels=3, classes=1):
        super().__init__()
        
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None  
        )
    
    def forward(self, x):
        return self.model(x)

def get_lane_model(device='mps'):
   
    model = LaneSegmentationModel(
        encoder_name='resnet18',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1
    )
    return model.to(device)
