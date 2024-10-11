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

def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

class Encoder(nn.Module):
    def __init__(self, inp_dim, layers_params, out_layer_dim):
        super().__init__()

        self.encoder = nn.Sequential()

        prev_out_dim = inp_dim
        for layer_count in range(1, len(layers_params) + 1):
            layer_params = layers_params[f'layer_params{layer_count}']

            self.encoder.append(nn.Linear(prev_out_dim, layer_params['out_dim']))
            self.encoder.append(nn.ReLU())
            self.encoder.append(nn.LazyBatchNorm1d())
            self.encoder.append(nn.Dropout(layer_params['drop_prob']))

            prev_out_dim = layer_params['out_dim']
        self.encoder.append(nn.Linear(prev_out_dim, out_layer_dim))

        self.encoder.apply(weight_init)

    def forward(self, x):
        return self.encoder(x)
    
class Decoder(nn.Module):
    def __init__(self, inp_dim, layers_params, out_layer_dim):
        super().__init__()

        self.decoder = nn.Sequential()

        prev_out_dim = layers_params[f'layer_params{len(layers_params)}']['out_dim']
        prev_drop_prob = layers_params[f'layer_params{len(layers_params)}']['drop_prob']
        self.decoder.append(nn.Linear(out_layer_dim, prev_out_dim))
        for layer_count in range(len(layers_params) - 1, 0, -1):
            layer_params = layers_params[f'layer_params{layer_count}']

            self.decoder.append(nn.Dropout(prev_drop_prob))
            self.decoder.append(nn.LazyBatchNorm1d())
            self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Linear(prev_out_dim, layer_params['out_dim']))

            prev_out_dim = layer_params['out_dim']
            prev_drop_prob = layer_params['drop_prob']

        self.decoder.append(nn.Dropout(prev_drop_prob))
        self.decoder.append(nn.LazyBatchNorm1d())
        self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Linear(prev_out_dim, inp_dim))

        self.decoder.apply(weight_init)

    def forward(self, x):
        return self.decoder(x)
    
class AE(nn.Module):

    def __init__(self, inp_dim, layers_params, out_layer_dim):
        super().__init__()
        self.encoder = Encoder(inp_dim, layers_params, out_layer_dim)
        self.decoder = Decoder(inp_dim, layers_params, out_layer_dim)

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
    np.random.seed(params['autoencoder']['seed'])
    torch.manual_seed(params['autoencoder']['seed'])

    # prepare the data
    file_dir = os.path.dirname(__file__)
    dataset = FCDataset(os.path.join(file_dir, '../../contest-data.npy'))
    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # initialize autoencoder
    inp_dim = dataset.data.shape[1]

    ae = AE(inp_dim, **params['autoencoder']['architecture'])
    loss_fn = nn.MSELoss()

    optimizer_params = params['autoencoder']['optimizer']
    optimizer_params['betas'] = (optimizer_params['beta1'], optimizer_params['beta2'])
    del optimizer_params['beta1'], optimizer_params['beta2']
    optimizer = torch.optim.Adam(ae.parameters(), **optimizer_params)
    
    # train autoencoder
    if not os.path.exists(os.path.join(file_dir, '../plots')):
        os.makedirs(os.path.join(file_dir, '../plots'))

    with Live() as live, \
    open(os.path.join(file_dir, '../plots/loss.csv'), 'w+') as loss_file:
        loss_writer = csv.writer(loss_file)
        loss_writer.writerow(['step', 'train_loss', 'test_loss'])

        for epoch in range(params['autoencoder']['num_epochs']):
            train_epoch(train_loader, ae, loss_fn, optimizer)
            
            # get metrics
            train_loss = evaluate(train_dataset, ae, loss_fn)
            test_loss = evaluate(test_dataset, ae, loss_fn)

            # log metrics
            live.log_metric('train/loss', train_loss, plot=False)
            live.log_metric('test/loss', test_loss, plot=False)

            # write metrics
            loss_writer.writerow([live.step, train_loss, test_loss])

            live.next_step()            

        # log model
        torch.save(ae.state_dict(), os.path.join(file_dir, '../model.pd'))
        live.log_artifact(os.path.join(file_dir, '../model.pd'), type='model', name='autoencoder')

        ae.encoder.eval()
        np.save(os.path.join(file_dir, '../encoded_data.npy'), ae.encoder(dataset.data).detach().numpy())

if __name__ == '__main__':
    encode()