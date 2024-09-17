import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

# define the input size, hidden_layer size, etc.
D_i, D_k, D_o = 10, 40, 1 # change the output size to 1 (1) (0)

# create the model
model = nn.Sequential(
        nn.Linear(D_i, D_k),
        nn.ReLU(),
        nn.Linear(D_k,D_k),
        nn.ReLU(),
        nn.Linear(D_k,D_o),
        nn.Sigmoid() # apply sigmoid activation to map to 0-1
)

# He initialization of weights
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_normal_(layer_in.weight)
        layer_in.bias.data.fill_(0.0)
model.apply(weights_init)


# construct SGD optimizer and initial learning rate and momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# object that decreases the learning rate by half every 10 epochs
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# create 100 random data points and store in data loader class
x = torch.rand(100,D_i) # uniformly generated between 0 and 1
y = (torch.rand(100) > 0.5).float().view(-1, 1)

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
        loss = nn.functional.binary_cross_entropy(pred, y_batch)
        # backward pass
        loss.backward()
        # SGD Update
        optimizer.step()
        # update statistics
        epoch_loss += loss.item()
    # print error
    print(f'Epoch {epoch:5d}, loss {epoch_loss:.3f}')

    # tell scheduler to consider updating learning rate
    scheduler.step()
