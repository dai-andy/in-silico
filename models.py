import torch

class LinearDecoder(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearDecoder, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

class MLPDecoder(torch.nn.Module):
    def __init__(self, input_size, output_size, n_hidden):
        super(MLPDecoder, self).__init__()
        hidden_size = 1024
        self.layers = [
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
        ]
        for i in range(n_hidden):
            self.layers += [
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU(),
            ]
        self.layers += [
            torch.nn.Linear(hidden_size, output_size),
        ]
        self.mlp = torch.nn.Sequential(*self.layers)


    def forward(self, x):
        return self.mlp(x)
