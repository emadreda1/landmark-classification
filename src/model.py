
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class ResidualBlock(nn.Module):
    def __init__(self, inp, hidden):
        super(ResidualBlock, self).__init__()
        
        self.conv_block = nn.Sequential(OrderedDict([
            ("conv1", nn.Sequential(OrderDict([
                ("conv", nn.Conv2d(inp, hidden, 3, padding="same")),
                ("batch_norm", nn.BatchNorm2d(hidden)),
                ("relu", nn.ReLU()),
            ]))),
        
            ("conv2", nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(hidden, inp, 3, padding="same")),
                ("batch_norm", nn.BatchNorm2d(inp))
            ])))
        ]))
        self.relu = nn.ReLU()

    def forward(self, x):
        
        Z = self.conv_block(x)
        
        Y = Z + x 
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = x.view(-1, 512 * 7 * 7)
#         x = self.dropout(F.relu(self.fc1(x)))
#         x = self.fc2(x)
        return self.relu(Y)


def get_conv_layer(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, pool=False):
    conv_block = nn.Sequential(
        OrderedDict(
            [
                ("conv", nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
                ("batch_norm", nn.BatchNorm2d(out_channels)),
                ("relu", nn.ReLU()),
            ]
        )
    )
    conv_layer = OrderedDict([("conv_block", conv_block)])
    
    if pool:
        conv_layer["max_pool"] = nn.MaxPool2d(kernel_size = 2, stride = 2)
    return nn.Sequential(conv_layer)

def get_fc_layer(in_channels, out_channels, dropout):
    sublayers = OrderedDict(
        [
            ("dropout", nn.Dropout(dropout)),
            ("linear", nn.Linear(in_channels, out_channels)),
            ("relu", nn.ReLU()),
        ]
    )
    return nn.Sequential(sublayers)

class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:
        super().__init__()
        
        
        self.model = nn.Sequential(OrderedDict([
            ("conv_0", get_conv_layer(3, 64, kernel_size=7, stride=2, padding=3)),
            ("res_block_0", ResidualBlock(64,64)),
            ("res_block_1", ResidualBlock(64,64)),
            ("conv_1", get_conv_layer(64, 128, kernel_size=3, stride=2, padding=1, pool=True)),
            ("res_block_2", ResidualBlock(128,128)),
            ("res_block_3", ResidualBlock(128,128)),
            ("conv_2", get_conv_layer(128, 256, kernel_size=3, stride=2, padding=1, pool=True)),
            ("res_block_4", ResidualBlock(256,256)),
            ("res_block_5", ResidualBlock(256,256)),
            ("conv_3", get_conv_layer(256, 512, kernel_size=3, stride=2, padding=1, pool=True)),
            ("res_block_6", ResidualBlock(512,512)),
            ("res_block_7", ResidualBlock(512,512)),
            ("avgpool", nn.AdaptiveAvgPool2d(output_size=(3, 3))),
            ("flatten", nn.Flatten()),
            ("fc1", get_fc_layer(3 * 3 * 512, 1024, dropout=dropout)),
            ("fc2", nn.Linear(1024, num_classes)),
        ]))
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders
    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):
    model = MyModel(num_classes=23, dropout=0.3)
    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)
    out = model(images)
    assert isinstance(out, torch.Tensor), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"
    assert out.shape == torch.Size([2, 23]), f"Expected an output tensor of size (2, 23), got {out.shape}"
