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

class CNN1d_2(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32),
            nn.AvgPool1d(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16),
            nn.AvgPool1d(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16),
            nn.AvgPool1d(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class CNN1d_3(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32),
            nn.AvgPool1d(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16),
            nn.MaxPool1d(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16),
            nn.MaxPool1d(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x