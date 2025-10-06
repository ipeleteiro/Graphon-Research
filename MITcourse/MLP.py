from collections import OrderedDict
import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])  # Flatten images

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

model = nn.Sequential(OrderedDict([
    ('layer1', nn.Linear(784, 100)),
    ('activation1', nn.ReLU()),
    ('layer2', nn.Linear(100, 50)),
    ('activation2', nn.ReLU()),
    ('output', nn.Linear(50, 10)),
    ('outActivation', nn.Sigmoid()),
]))

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

        total_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(y_pred, 1)  # Get the class with the highest score
        correct += (predicted == y).sum().item()
        total += y.size(0)

    train_accuracy = correct / total * 100
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

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

torch.save(model.state_dict(), "my_model.pickle")