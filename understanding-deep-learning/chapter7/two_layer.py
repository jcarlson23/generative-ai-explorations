import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# define the input size, hidden_layer size, etc.
D_i, D_k, D_o = 10, 40, 5

# create the model
model = nn.Sequential(
        nn.Linear(D_i, D_k),
        nn.ReLU(),
        nn.Linear(D_k,D_k),
        nn.ReLU(),
        nn.Linear(D_k,D_o)
)

# He initialization of weights
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(weights_init)

# choose least square loss function
criterion = nn.MSELoss()
# construct SGD optimizer and initial learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# object that decreases the learning rate by half every 10 epochs
sheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# create 100 random data points and store in data loader class
x = torch.randn(100,D_i)
y = torch.randn(100,D_o)

data_loader = DataLoader(TensorDataset(x,y), batch_size=10, shuffle=True)

# loop over the dataset 100 times
for epoch in range(100):
    epoch_loss = 0.0
    # loop over batches
    for i, data in enumerate(data_loader):
        # retrieve inputs and labels for this batch
        x_batch, y_batch = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward pass
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        # backward pass
        loss.backward()
        # SGD Update
        optimizer.step()
        # update statistics
        epoch_loss += loss.item()
    # print error
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')
    # tell scheduler to consider updating learning rate

