import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparsity, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        
        # Define weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features)) if bias else None
        
        
        self.reset_parameters() # Initialise weights and bias
        self.mask = self.create_mask() # Create and apply the mask
    
    def reset_parameters(self):
        # Kaiming initialisation - designed for ReLU-based networks to maintain the variance of activations across layers
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5)) 
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def create_mask(self):
        """Create a binary mask with the specified sparsity level."""
        mask = torch.rand(self.weight.shape) > self.sparsity  # Keep (1 - sparsity) fraction of weights
        return mask.float().to(self.weight.device)  # Ensure mask is on the same device
    
    def forward(self, x):
        masked_weight = self.weight * self.mask  # Apply the mask
        return F.linear(x, masked_weight, self.bias)
    
    def apply_mask(self):
        """Ensure masked weights remain zero after backpropagation."""
        with torch.no_grad():
            self.weight *= self.mask  # Reapply mask after gradient updates


# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten images

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

def buildModel(sparsity):
    model = nn.Sequential(OrderedDict([
        ('layer1', SparseLinear(784, 100, sparsity)),
        ('activation1', nn.ReLU()),
        ('layer2', SparseLinear(100, 50, sparsity)),
        ('activation2', nn.ReLU()),
        ('output', SparseLinear(50, 10, sparsity)),
        ('outActivation', nn.Sigmoid()),
    ]))
    return model

# 30, 50, 70, 80, 90, 99
model = buildModel(0.01)
results = open("MLPTestingResults.txt", "a")
results.write("-------------RESULTS FOR 1% SPARSITY-------------\n")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X, y = next(iter(train_loader))

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()         # apply backpropegation
        optimizer.step()

        # Apply the mask after updating weights
        for layer in model.modules():
            if isinstance(layer, SparseLinear):  
                layer.apply_mask()  # Ensure masked weights stay zero

        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(y_pred, 1)  # Get the class with the highest score
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_accuracy = correct / total * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    results.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n")

# Evaluate on test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X, y in test_loader:
        y_pred = model(X)
        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

test_accuracy = correct / total * 100
print(f"Final Test Accuracy: {test_accuracy:.2f}%")
results.write(f"Final Test Accuracy: {test_accuracy:.2f}%\n\n")

torch.save(model.state_dict(), "my_model.pickle")
results.close()

