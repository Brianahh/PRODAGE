import torch.nn as nn

class Age_model_Resnet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        pretrained_model.fc = nn.Identity()
        self.pretrain_layer = pretrained_model
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 1)
        )
    def forward(self,x):
        x = self.pretrain_layer(x)
        x = self.fc(x)
        return x
