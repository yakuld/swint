import torch
import torch.nn as nn
from einops import rearrange
from torchvision.models import efficientnet_b0
from timm.models.registry import register_model

class EfficientNetModel(nn.Module):
    def __init__(self, model, duration):
        super().__init__()
        self.model = model
        self.duration = duration
        self.head = nn.Linear(1280, 2)
    
    def forward(self, x):
        x = rearrange(x, 'b (n t c) h w -> (b n t) c h w', t=self.duration, c=3)
        # print(x.shape)
        x = self.model(x)
        x = torch.squeeze(x)
        x = self.head(x)
        return x

@register_model
def EfficientNet(pretrained = False, **kwargs):

    if(pretrained):
        efficient_net = efficientnet_b0(weights = 'IMAGENET1K_V1')
    else:
        efficient_net = efficientnet_b0()
    # model = models.resnet152(pretrained=True)
    newmodel = torch.nn.Sequential(*(list(efficient_net.children())[:-1]))
    # print(newmodel)
    model = EfficientNetModel(newmodel, **kwargs)
    # print(model.parameters)
    return model

# model = EfficientNet(duration = 4)
# inputs = torch.randn((4, 96, 224, 224))
# outputs = model(inputs)
# outputs = torch.squeeze(outputs)
# print(outputs.shape)
# head = nn.Linear(1280, 2)
# outputs = head(outputs)
# print(outputs.shape)
# outputs = outputs.reshape(4, 8 * 4,-1).mean(dim=1)
# print(outputs.shape)