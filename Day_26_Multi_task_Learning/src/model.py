import torch
import torch.nn as nn

class MultiTaskModel(nn.Module):
    def __init__(self, input_size, shared_hidden_size, task1_output_size, task2_output_size):
        super(MultiTaskModel, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_size, shared_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(shared_hidden_size),
        )

        self.task1_head = nn.Sequential(
            nn.Linear(shared_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, task1_output_size)
        )

        self.task2_head = nn.Sequential(
            nn.Linear(shared_hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, task2_output_size)
        )

    def forward(self, x):
        shared_representation = self.shared(x)
        task1_output = self.task1_head(shared_representation)
        task2_output = self.task2_head(shared_representation)
        return task1_output, task2_output
