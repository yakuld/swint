import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

class EfficientNetModel(nn.Module):
    def __init__(self, model, duration):
        super().__init__()
        self.model = model
        self.duration = duration
    
    def forward(self, x):
        x = rearrange(x, 'b (n t c) h w -> (b n t) c h w', t=self.duration, c=3)
        print(x.shape)
        x = self.model(x)
        return x

@register_model
def EfficientNet(duration = 4):

    # efficient_net = efficientnet_b0(weights = 'IMAGENET1K_V1')
    efficient_net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    efficient_net.eval().to(device)
    model = EfficientNetModel(efficient_net, duration)
    return model

model = EfficientNet(duration = 4)
inputs = torch.randn((4, 96, 224, 224))
outputs = model(inputs)
print(outputs.shape)
print(outputs[0,:])