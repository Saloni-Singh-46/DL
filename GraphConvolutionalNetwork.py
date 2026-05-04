import torch
import torch.nn as nn

class SimpleGCN(nn.Module):
    def __init__(self):
        super(SimpleGCN, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x

# Example usage
model = SimpleGCN()
x = torch.rand(5, 10)      # 5 nodes
adj = torch.eye(5)         # adjacency matrix

output = model(x, adj)
print(output)