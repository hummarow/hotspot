import sys
import torch
import pandas as pd
import numpy as np
import matplotlib as plt
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from dataloader import EEGDataset, EEGDataloader

from model import Model
from utils.args import get_argparser


# Configs
EPOCH = 10

def main(args):
    # Get Dataloader
    dataset = EEGDataset(args.data_path)
    train_set_size = int(len(dataset) * 0.8)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    val_set_size = len(dataset) - train_set_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size])
    train_loader, val_loader = DataLoader(train_set, batch_size=4, shuffle=True), DataLoader(val_set, batch_size=1, shuffle=True)

    # Create model
    model = Model()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    pdist = nn.PairwiseDistance(p=2)

    # DataFrame for saving results
    df = pd.DataFrame(columns=['Label', 'Predict', 'MSE', 'Euclidean Distance'])
    # Train
    for epoch in range(EPOCH):
        print('Epoch: {}'.format(epoch))
        model.train()
        total_loss = 0
        total_dist = 0
        for i, batch in enumerate(train_loader):
            opt.zero_grad()
            x, y = batch
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
            distances = pdist(y, pred)
            total_dist += distances.mean()
            loss.backward()
            if i % 20 == 0:
                print('MSE Loss: {:2f}\tEuclidean Distance: {:2f}'.format(loss.item(), distances[0].item()))
            opt.step()
        print('Avg train loss: {:2f}\nAvg train dist: {:2f}'.format(total_loss/len(train_loader), total_dist/len(train_loader)))
        print('* Evaluating')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_dist = 0
            for i, batch in enumerate(val_loader):
                x, y = batch
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item()
                distances = pdist(y, pred)
                total_dist += distances[0].item()
                df.loc[len(df.index)] = [y, pred, loss.item(), distances[0].item()]
            print('Average Validation Loss: {:2f}'.format(total_loss / len(val_loader)))
            print('Average Distance: {:2f}'.format(total_dist / len(val_loader)))

    df.to_csv('result.csv')


if __name__=='__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
