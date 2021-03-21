import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from addict import Dict
import yaml

# TORCH
import torch
from torch.nn.modules import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

current_path = os.path.abspath(os.path.dirname(__file__))

from dataloader import MyDataset, get_training_augmentation, get_validation_augmentation
from model import MyModel


# read config file
with open('./configs/train_params.yaml') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

config = Dict(config)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

model = MyModel()
model.to(device)

train_df = pd.read_csv('../data/train_dataset.csv')
test_df = pd.read_csv('../data/test_dataset.csv')

train_dataset = MyDataset(train_df, image_folder_path='../data/images',
                          augmentations=get_training_augmentation())

test_dataset = MyDataset(test_df, image_folder_path='../data/images',
                         augmentations=get_validation_augmentation())

train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size)
test_dataloader = DataLoader(test_dataset, batch_size = 1)

optimizer = Adam(model.parameters(), lr=config.learning_rate)

loss_function = CrossEntropyLoss()

for epoch in range(config.epochs):

    print(f'EPOCH {epoch}')
    print(f'-------------')

    """ TRAIN """
    print(f'TRAINING LOOP')

    losses = []
    model.train()

    for batch in tqdm(train_dataloader):

        images = batch['image'].to(device)
        targets = batch['class_id'].type(torch.int64).to(device)

        optimizer.zero_grad()

        predictions = model(images)
        loss = loss_function(predictions, targets)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f'Mean train loss: {np.mean(losses)}')

    """ TEST """
    print(f'TEST LOOP')

    losses = []
    model.eval()

    with torch.no_grad():

        for batch in tqdm(test_dataloader):

            images = batch['image'].to(device)
            targets = batch['class_id'].type(torch.int64).to(device)

            predictions = model(images)

            losses.append(loss.item())

    print(f'Mean test loss: {np.mean(losses)}')

#torch.save(model.state_dict(), os.path.join("best_model.pt"))
