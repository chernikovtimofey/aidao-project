import os
import numpy as np
import dvc.api
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dvclive import Live
from FCDataset import FCDataset

class Encoder(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid2_dim),
            nn.ReLU(),
            nn.Linear(hid2_dim, hid1_dim),
            nn.ReLU(),
            nn.Linear(hid1_dim, inp_dim)
        )

    def forward(self, x):
        return self.decoder(x)
    
class AE(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim):
        super().__init__()
        self.encoder = Encoder(inp_dim, hid1_dim, hid2_dim, out_dim)
        self.decoder = Decoder(inp_dim, hid1_dim, hid2_dim, out_dim)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    loss = None
    for x in dataloader:
        # compute loss
        pred = model(x)
        loss = loss_fn(pred, x)

        # backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def test_loop(dataloader, model, loss_fn):
    model.eval()

    loss = 0
    for x in dataloader:
        # compute loss
        pred = model(x)
        loss += loss_fn(pred, x).item()
    
    loss /= len(dataloader)
    return loss 
    

def encode():
    # extract params
    fc_params = dvc.api.params_show()['FCDataset']
    seed = fc_params['seed']

    ae_params = dvc.api.params_show()['autoencoder'] 
    hid1_dim = ae_params['hid1_dim']
    hid2_dim = ae_params['hid2_dim']
    out_dim = ae_params['out_dim']
    lr = ae_params['lr']
    beta1 = ae_params['beta1']
    beta2 = ae_params['beta2']
    eps = ae_params['eps']
    weight_decay = ae_params['weight_decay']
    num_epochs = ae_params['num_epochs']

    # extract the data
    file_dir = os.path.dirname(__file__)
    dataset = FCDataset(os.path.join(file_dir, '../../contest-data.npy'), seed=seed)
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # initialize autoencoder
    inp_dim = dataset.data.shape[1]
    ae = AE(inp_dim, hid1_dim, hid2_dim, out_dim)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), 
                                 lr=lr, betas=(beta1, beta2),
                                 eps=eps, weight_decay=weight_decay)
    
    # train autoencoder
    with Live() as live:
        for epoch in range(num_epochs):
            train_loss = train_loop(train_loader, ae, loss_fn, optimizer)
            test_loss = test_loop(test_loader, ae, loss_fn)

            live.log_metric('train_loss', train_loss)
            live.log_metric('test_loss', test_loss)
            live.next_step()

        np.save(os.path.join(file_dir, '../encoded_data.npy'), ae.encoder(dataset.data).detach().numpy())

if __name__ == '__main__':
    encode()