import os
import csv
import numpy as np
import dvc.api
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from dvclive import Live
from FCDataset import FCDataset

class Encoder(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim, seed=None, p=0.5):
        super().__init__()
        np.random.seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, hid1_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(hid1_dim, hid2_dim),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(hid2_dim, out_dim)
        )

    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim, seed=None, p=0.5):
        super().__init__()
        np.random.seed(seed)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hid2_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(hid2_dim, hid1_dim),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(hid1_dim, inp_dim)
        )

    def forward(self, x):
        return self.decoder(x)
    
class AE(nn.Module):
    def __init__(self, inp_dim, hid1_dim, hid2_dim, out_dim, seed=None, p=0.5):
        super().__init__()
        self.encoder = Encoder(inp_dim, hid1_dim, hid2_dim, out_dim, seed, p)
        self.decoder = Decoder(inp_dim, hid1_dim, hid2_dim, out_dim, seed, p)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()

    for x in dataloader:
        # compute loss
        pred = model(x)
        loss = loss_fn(pred, x)

        # backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(dataset, model, loss_fn):
    model.eval()
    x = next(iter(DataLoader(dataset, len(dataset))))

    pred = model(x)
    loss = loss_fn(pred, x).item()

    return loss
    

def encode():
    params = dvc.api.params_show()

    # extract the data
    file_dir = os.path.dirname(__file__)
    dataset = FCDataset(os.path.join(file_dir, '../../contest-data.npy'), seed=params['seed'])
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], torch.Generator().manual_seed(params['seed']))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # initialize autoencoder
    inp_dim = dataset.data.shape[1]
    ae = AE(inp_dim, 
            params['architecture']['hid1_dim'], 
            params['architecture']['hid2_dim'], 
            params['architecture']['out_dim'], 
            seed=params['seed'], p=params['architecture']['p'])
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(ae.parameters(), 
                                 lr=params['optimizer']['lr'], 
                                 betas=(params['optimizer']['beta1'], params['optimizer']['beta2']),
                                 eps=params['optimizer']['eps'], 
                                 weight_decay=params['optimizer']['weight_decay'])
    
    # train autoencoder
    with Live() as live, \
    open(os.path.join(file_dir, '../plots/metrics.csv'), 'w+') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        metrics_writer.writerow(['step', 'train_loss', 'test_loss'])

        for epoch in range(params['optimizer']['num_epochs']):
            train_epoch(train_loader, ae, loss_fn, optimizer)
            
            # get metrics
            train_loss = evaluate(train_dataset, ae, loss_fn)
            test_loss = evaluate(test_dataset, ae, loss_fn)

            # log metrics
            live.log_metric('train/loss', train_loss, plot=False)
            live.log_metric('test/loss', test_loss, plot=False)

            # write metrics
            metrics_writer.writerow([live.step, train_loss, test_loss])

            live.next_step()            

        # log model
        torch.save(ae.state_dict(), os.path.join(file_dir, '../model.pd'))
        live.log_artifact(os.path.join(file_dir, '../model.pd'), type='model', name='autoencoder')

        np.save(os.path.join(file_dir, '../encoded_data.npy'), ae.encoder(dataset.data).detach().numpy())

if __name__ == '__main__':
    encode()