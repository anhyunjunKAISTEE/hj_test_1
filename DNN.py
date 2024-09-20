import torch
import torch.nn as nn
import torch.optim as optim

class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(4, 64)  # 입력 차원: 4, 출력 차원: 64
        self.fc2 = nn.Linear(64, 128) # 입력 차원: 64, 출력 차원: 128
        self.fc3 = nn.Linear(128, 201)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    
class DNNModel2(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[60, 120, 160], output_dim=201):
        super(DNNModel2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x):
        # Input layer
        x = self.leaky_relu(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = layer(x)
            if i % 2 == 0:
                x = self.elu(x)
            else:
                x = self.selu(x)
            x = self.dropout(x)
            if x.size() == residual.size():
                x += residual  # Skip connection
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    

class DNNModel2(nn.Module):
    def __init__(self, input_dim=4, hidden_dims=[40, 80, 140], output_dim=201):
        super(DNNModel2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Activation functions
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.elu = nn.ELU()
        self.selu = nn.SELU()
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])

    def forward(self, x):
        # Input layer
        x = self.leaky_relu(self.input_layer(x))
        
        # Hidden layers with skip connections
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = layer(x)
            if i % 2 == 0:
                x = self.elu(x)
            else:
                x = self.selu(x)
            x = self.dropout(x)
            if x.size() == residual.size():
                x += residual  # Skip connection
        
        # Layer Normalization
        x = self.layer_norm(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x

    