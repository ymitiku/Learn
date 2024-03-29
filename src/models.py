import torch 
from torch import nn
from torch.nn import functional as F
from .resnet import ResNet18
class ClassificationModel(nn.Module):
    def __init__(self, num_classes, width_per_group = 16):
        super().__init__()
        self.backbone = ResNet18()
        self.linear = nn.Linear(self.backbone.out_dim, num_classes) 
            
    def forward(self, inputs):
        return self.linear(self.backbone(inputs))
        
class MixupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18()
        self.projection_head = nn.Linear(self.backbone.out_dim * 2, 1024)
        self.linear = nn.Linear(1024, 1)  
            
    def forward(self, inputs):
        assert isinstance(inputs, list)
        x = self.backbone(inputs[0])
        x = torch.cat([x, self.backbone(inputs[1])], dim=1)
        x = self.projection_head(x)
        x = self.linear(x)
        
        return x
        

        