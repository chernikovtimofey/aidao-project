import os
import csv
import sklearn.metrics
import dvc.api
import torch
import torch.nn.init   
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from dvclive import Live
from Datasets import BnuDataset
from Datasets import IhbDataset

def weight_init(m):
    if isinstance(m, nn.LazyConv2d) or isinstance(m, nn.LazyLinear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zero(m.bias)
        

class ConvNet(nn.Module):
    def __init__(self, conv1_params, conv2_params, drop_prob):
        super().__init__()

        lin_dim = \
        (419 + 2 * conv1_params['padding'] - conv1_params['kernel_size'] + 1) // conv1_params['stride']
        lin_dim = \
        (lin_dim + 2 * conv2_params['padding'] - conv2_params['kernel_size'] + 1) // conv2_params['stride']
        lin_dim = conv2_params['out_channels'] * lin_dim**2

        self.net = nn.Sequential(
            nn.Conv2d(1, **conv1_params),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Dropout(drop_prob),
            nn.Conv2d(conv1_params['out_channels'], **conv2_params),
            nn.ReLU(),
            nn.LazyBatchNorm2d(),
            nn.Dropout(drop_prob),
            nn.Flatten(),
            nn.Linear(lin_dim, 1)
        )
        self.net.apply(weight_init)
    
    def forward(self, x):
        return self.net(x)
    
def train_epoch(dataloader, model, loss_fn, optimizer):
    model.train()

    for x, y in dataloader:
        # compute loss
        score = model(x)[:, 0]
        loss = loss_fn(score, y)

        # backpropogate
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def evaluate(dataset, model, loss_fn, threshold):
    model.eval()
    x, y = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    
    with torch.no_grad():
        # calculate predictions
        score = model(x)[:, 0]
        pred = (score > threshold)

        # get metrics
        metrics = {}
        metrics['loss'] = loss_fn(score, y).item()
        metrics['accuracy'] = sklearn.metrics.accuracy_score(y, pred)
        metrics['precision'] = sklearn.metrics.precision_score(y, pred, zero_division=0)
        metrics['recall'] = sklearn.metrics.recall_score(y, pred)
        metrics['f1'] = sklearn.metrics.f1_score(y, pred)

        fpr, tpr, _ = sklearn.metrics.roc_curve(y, score)

        return metrics, y, pred, fpr, tpr

def train():
    params = dvc.api.params_show()

    # prepare the data
    file_dir = os.path.dirname(__file__)
    bnu_dataset = BnuDataset(os.path.join(file_dir, '../../contest-data'), transform=lambda inp: torch.unsqueeze(inp, 0))
    ihb_dataset = IhbDataset(os.path.join(file_dir, '../../contest-data'), transform=lambda inp: torch.unsqueeze(inp, 0))
    bnu_train, bnu_test = random_split(bnu_dataset, [0.8, 0.2])
    ihb_train, ihb_test = random_split(ihb_dataset, [0.8, 0.2])
    train_dataset = ConcatDataset([bnu_train, ihb_train])
    test_dataset = ConcatDataset([bnu_test, ihb_test])
    train_loader = DataLoader(train_dataset, batch_size=64)

    # initialize ConvNet
    net = ConvNet(**params['net_architecture'])

    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=params['net_optimizer']['lr'],
                                 betas=(params['net_optimizer']['beta1'], params['net_optimizer']['beta2']),
                                 eps=params['net_optimizer']['eps'], 
                                 weight_decay=params['net_optimizer']['weight_decay'])

    # train ConvNet
    if not os.path.exists(os.path.join(file_dir, '../plots')):
        os.makedirs(os.path.join(file_dir, '../plots'))

    with Live() as live, \
    open(os.path.join(file_dir, '../plots/loss.csv'), 'w+') as loss_file, \
    open(os.path.join(file_dir, '../plots/accuracy.csv'), 'w+') as acc_file, \
    open(os.path.join(file_dir, '../plots/precision.csv'), 'w+') as precision_file, \
    open(os.path.join(file_dir, '../plots/recall.csv'), 'w+') as recall_file, \
    open(os.path.join(file_dir, '../plots/f1.csv'), 'w+') as f1_file:
        loss_writer = csv.writer(loss_file)
        acc_writer = csv.writer(acc_file)
        precision_writer = csv.writer(precision_file)
        recall_writer = csv.writer(recall_file)
        f1_writer = csv.writer(f1_file)
        
        loss_writer.writerow(['step', 'train_loss', 'test_loss'])
        acc_writer.writerow(['step', 'train_accuracy', 'test_accuracy'])
        precision_writer.writerow(['step', 'train_precision', 'test_precision'])
        recall_writer.writerow(['step', 'train_recall', 'test_recall'])
        f1_writer.writerow(['step', 'train_f1', 'test_f1'])
        
        for epoch in range(params['net_optimizer']['num_epochs']):
            train_epoch(train_loader, net, loss_fn, optimizer)

            # evaluate 
            train_metrics, actual, train_pred, train_fpr, train_tpr = \
                evaluate(train_dataset, net, loss_fn, params['net_threshold'])
            test_metrics, actual, test_pred, test_fpr, test_tpr = \
                evaluate(test_dataset, net, loss_fn, params['net_threshold'])
            
            # log metrics
            for metric, value in train_metrics.items():
                live.log_metric(f'train/{metric}', value, plot=False)
            for metric, value in test_metrics.items():
                live.log_metric(f'test/{metric}', value, plot=False)

            # write metics
            loss_writer.writerow([live.step, train_metrics['loss'], test_metrics['loss']])
            acc_writer.writerow([live.step, train_metrics['accuracy'], test_metrics['accuracy']])
            precision_writer.writerow([live.step, train_metrics['precision'], test_metrics['precision']])
            recall_writer.writerow([live.step, train_metrics['recall'], test_metrics['recall']])
            f1_writer.writerow([live.step, train_metrics['f1'], test_metrics['f1']])

            # log train confusion matrix
            live.log_plot(
                'train_confusion_matrix',
                [{'true_lab' : true_lab.item(), 'pred_lab' : pred_lab.item()} for true_lab, pred_lab in zip(actual, train_pred)],
                x='true_lab',
                y='pred_lab',
                template='confusion',
                title='Confusion matrix for train set',
                x_label='Ground truth value',
                y_label='Predicted value'
            )

            # log test confusion matrix
            live.log_plot(
                'test_confusion_matrix',
                [{'true_lab' : true_lab.item(), 'pred_lab' : pred_lab.item()} for true_lab, pred_lab in zip(actual, test_pred)],
                x='true_lab',
                y='pred_lab',
                template='confusion',
                title='Confusion matrix for test set',
                x_label='Ground truth value',
                y_label='Predicted value'
            )

            # log train roc
            live.log_plot(
                'train_roc',
                [{'fp' : fp, 'tp' : tp} for fp, tp in zip(train_fpr, train_tpr)],
                x='fp',
                y='tp',
                template='linear',
                title='ROC for train set'
            )

            # log test roc
            live.log_plot(
                'test_roc',
                [{'fp' : fp, 'tp' : tp} for fp, tp in zip(test_fpr, test_tpr)],
                x='fp',
                y='tp',
                template='linear',
                title='ROC for test set'
            )
            
            live.next_step()

        # log model
        torch.save(net.state_dict(), os.path.join(file_dir, '../model.pd'))
        live.log_artifact(os.path.join(file_dir, '../model.pd'), type='model', name='ConvNet')

if __name__ == '__main__':
    train()