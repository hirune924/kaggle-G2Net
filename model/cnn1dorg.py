import torch
import torch.nn as nn

class CNN1d(nn.Module):
    def __init__(self):
        super(CNN1d, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(3, 64,kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(64),nn.ReLU(),nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(64, 128,kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(128, 256,kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Conv1d(256, 256,kernel_size=64, stride=2, padding=32),
            nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(kernel_size=2, stride=1))

        self.fc = nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(256,1))
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x
