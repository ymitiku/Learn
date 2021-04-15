import torch 
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50
class ClassificationModel(nn.Module):
    def __init__(self, num_classes, width_per_group = 16):
        super().__init__()
        self.backbone = resnet50(width_per_group=width_per_group)
        self.linear = nn.Linear(width_per_group * 32, num_classes) 
            
    def forward(self, inputs):
        return self.linear(self.backbone(inputs))
        
class MixupModel(nn.Module):
    def __init__(self, width_per_group = 16):
        super().__init__()
        self.backbone = resnet50(width_per_group=width_per_group)
        self.linear = nn.Linear(width_per_group * 32 * 2, 1) 
            
    def forward(self, inputs):
        assert isinstance(inputs, list)
        x = self.backbone(inputs[0])
        x = torch.cat([x, self.backbone(inputs[1])], dim=1)
        x = self.linear(x)
        
        return x
        

        