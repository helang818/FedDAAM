import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self) -> None:
        super(Conv1d, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(168, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        ) #10 * 128 * 114

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class AstroModel(nn.Module):
    def __init__(self) -> None:
        super(AstroModel, self).__init__()
        self.conv = nn.Conv1d(128, 256, 1)
        self.dropout = nn.Dropout(0.2)

        self.conv1_1 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)
        self.conv1_2 = nn.Conv1d(128, 128, 3, padding=2, dilation=2)

        self.conv2_1 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=4, dilation=4)

        self.conv3_1 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)
        self.conv3_2 = nn.Conv1d(128, 128, 3, padding=8, dilation=8)

        self.conv4_1 = nn.Conv1d(128, 256, 3, padding=16, dilation=16)
        self.conv4_2 = nn.Conv1d(256, 256, 3, padding=16, dilation=16)
    
    def forward(self, x):
        raw = x
        x = F.relu(self.conv1_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv1_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv2_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv2_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv3_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv3_2(x))
        raw = F.relu(x + raw)

        x = raw
        x = F.relu(self.conv4_1(x))
        x = self.dropout(x)
        x = self.dropout(self.conv4_2(x))
        raw = self.conv(raw)
        raw = F.relu(x + raw)

        return raw
        
class TCNModel(nn.Module):
    def __init__(self) -> None:
        super(TCNModel, self).__init__()
        self.Conv1d = Conv1d()
        self.AstroModel = AstroModel()
    
    # input.shape = 10*915*171
    def forward(self, input):
        input = input.transpose(1,2)
        x = self.Conv1d(input)
        x = self.AstroModel(x)
        x = x.transpose(1,2)
        return x