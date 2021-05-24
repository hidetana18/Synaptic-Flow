import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch import linalg



def rescale(model, model_base):
    for mod, mod_base in zip(model.modules(), model_base.modules()):
        if(isinstance(mod, nn.Conv2d)):
            #print( 'before='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
            mod.weight.data = (mod.weight.data / torch.linalg.norm(mod.weight, dim=(1,2,3), keepdim=True)) * torch.linalg.norm(mod_base.weight, dim=(1,2,3), keepdim=True)
            #print( 'after='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
    #print('end!')
    return model

def rescale_constant(model, model_base):
    for mod, mod_base in zip(model.modules(), model_base.modules()):
        if(isinstance(mod, nn.Conv2d)):
            #print( 'before='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
            mod.weight.data = 100*mod.weight.data
            #print( 'after='+ str( torch.norm(torch.norm(mod.weight, dim=(2,3), keepdim=True), dim=1, keepdim=True)[1]) )
    #print('end!')
    return model


def train(model, model_base, norm_fix, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        # Freeze Norm
        if norm_fix:
            model = rescale(model, model_base)

        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, model_base, norm_fix, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    model = rescale_constant(model, model_base)
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, accuracy5]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, model_base, norm_fix, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    return pd.DataFrame(rows, columns=columns)
