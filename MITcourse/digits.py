import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

n_epochs = 3                # number of times to loop over the dataset
batch_size_train = 64       # training batch
batch_size_test = 1000      # testing batch
learning_rate = 0.01        # hyperparameter for optimiser
momentum = 0.5              # hyperparameter for optimiser
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False        # disables non-deterministic algorithms (algorithm that, even for the same input, can exhibit different behaviors on different runs)
torch.manual_seed(random_seed)

# loading in the MNIST dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))  # given MNIST mean and standard deviation
                             ])),
  batch_size=batch_size_train, shuffle=True)

# test data
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

# example sample from the MNIST dataset
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
example_data.shape
# >> torch.Size([1000, 1, 28, 28])
# ie: 1000 images, graysclae (1 - no rgb channels), 28x28 pixels
fig = plt.figure()
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# Pooling
    # It divides the input feature map into small regions (typically non-overlapping or slightly overlapping).
    # From each region, it selects the maximum value.
    # This reduces the size of the feature map, making computation more efficient while retaining key features.
# Dropout
    # During training, a random subset of neurons is deactivated (set to zero) in each forward pass.
    # The remaining neurons are scaled up to compensate for the dropped neurons.
    # During inference (testing), dropout is turned off, and the full network is used.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    # convolutional layer (process image, 1 input channel, 10 output channels)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   # convolutional layer (10 input channels, 20 output channels)
        self.conv2_drop = nn.Dropout2d()            # regularisation using dropout
        self.fc1 = nn.Linear(320, 50)   # fuly connected layer
        self.fc2 = nn.Linear(50, 10)    # fully connected layer (outputs 10 classes - digits 0-9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      # apply ReLU and max pooling the first convolutional layer
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))     # apply ReLU, max pooling and regularisation the second convolutional layer
        x = x.view(-1, 320)     # Flatten the tensor from 3D (feature maps) to 2D (for fully connected layer input)
        x = F.relu(self.fc1(x))     # apply ReLU first linear layer
        x = F.dropout(x, training=self.training)    # apply dropout first linear layer
        x = self.fc2(x)     # apply ReLU second linear layer
        return F.log_softmax(x)     # return loss function


network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)


train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), 'results/model.pth')
      torch.save(optimizer.state_dict(), 'results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


test()
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')
plt.scatter(test_counter, test_losses, color='red')
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()
