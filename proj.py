import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class MLMPNNConv(MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, num_levels=2):
        super(MLMPNNConv, self).__init__(aggr='add')  # Use 'add' aggregation for messages
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_levels = num_levels

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = nn.Linear(hidden_channels * num_levels, out_channels)

    def message(self, x_j, edge_index):
        # No message update in the original ML-MPNN
        return x_j

def update(self, x, edge_index):
    # Multi-level message aggregation (simplified from the original paper)
    row, col = edge_index
    deg = degree(row, x.size(0), dtype=torch.float)
    
    # Ensure that deg has the same size as x
    norm = deg.pow(-0.5)  # Degree normalization
    
    # Debugging: Check sizes
    print(f"x.size(0): {x.size(0)}, deg.size: {deg.size()}, norm.size: {norm.size()}")
    
    # Make sure col is valid and matches the expected size
    messages = x[col] * norm[row].view(-1, 1)  # Basic message passing with normalization

    x_new = self.lin1(x)
    x_new = torch.relu(x_new)
    x_new = self.lin2(x_new)

    # Multi-level aggregation (alternative approach using concatenation)
    multi_level_feats = torch.cat([x_new, messages], dim=1)
    for _ in range(1, self.num_levels):
        multi_level_feats = torch.cat([multi_level_feats, messages], dim=1)

    x = self.lin3(multi_level_feats)
    x = torch.relu(x)
    return x


def forward(self, x, edge_index):
    return self.propagate(edge_index, x=x)


class MLMPNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_levels=2, num_layers=2):
        super(MLMPNN, self).__init__()
        self.convs = nn.ModuleList([MLMPNNConv(in_channels, hidden_channels, hidden_channels, num_levels) for _ in range(num_layers)])
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        
        x = torch.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# Example usage with the QM9 dataset
import torch_geometric.datasets as datasets

# Define the root directory for downloading the dataset (replace with your desired path)
root = r"C:\Users\Rajpal\Desktop\AI"

# Load the QM9 dataset
dataset = datasets.QM9(root=root)
data = dataset[0]
print(data)


# Check if data.x needs preprocessing
if data.x.dtype != torch.float:
    data.x = data.x.float()  # Convert features to float tensors

if data is None:
    print("Error: Data loading failed!")
    exit()  # Exit the program if data is not loaded

# Define model parameters
in_channels = dataset.num_features  # Number of node features
hidden_channels = 128
out_channels = 1  # Example: Predicting a single molecular property

# Create ML-MPNN model
model = MLMPNN(in_channels, hidden_channels, out_channels)

# Input:
# - data.x: Node features (e.g., atom types, coordinates)
# - data.edge_index: Edge indices representing connections between atoms

# Output:
# - Predicted molecular property (e.g., atomization energy)

# Forward pass
output = model(data.x, data.edge_index)

print(output)
