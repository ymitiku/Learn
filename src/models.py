import torch 
from torch import nn
from torch.nn import functional as F
from .resnet import resnet50
class MixupLearnModel(nn.Module):
    def __init__(self, num_classes = None, siamese = False, width_per_group=16):
        super().__init__()
        if not siamese:
            assert num_classes is not None
        self.siamese = siamese
        self.backbone = resnet50(width_per_group=width_per_group)
        if siamese:
            self.fc = nn.Linear(width_per_group * 32*2, num_classes)
        else:
            self.fc = nn.Linear(width_per_group * 32, num_classes)
            
    def forward(self, inputs1, inputs2 = None):
        if self.siamese:
            assert inputs2 is not None
        
        x = self.backbone(inputs1)
        if self.siamese:
            x2 = self.backbone(inputs2)
            x = torch.cat([x, x2], dim=1)
        x = self.fc(x)
        
        return x
        

        