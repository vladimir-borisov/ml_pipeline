import pandas as pd
from dataloader import DataLoader, Dataset, get_training_augmentation, get_test_augmentation
from model import get_model
import tensorflow as tf
import yaml
from addict import Dict


# read config file
with open('./configs/train_params.yaml') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

config = Dict(config)

train_df = pd.read_csv('../data/train_dataset.csv')
test_df = pd.read_csv('../data/test_dataset.csv')

train_dataset = Dataset(df = train_df,
                        image_folder_path='../data/images',
                        augmentations=get_training_augmentation())

test_dataset = Dataset(df = test_df,
                        image_folder_path='../data/images',
                        augmentations=get_test_augmentation())


train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

model = get_model()

optimizer = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
loss = tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer, loss = loss, metrics=['accuracy'])

model.fit(x = train_dataloader, steps_per_epoch=len(train_dataloader), epochs=config.epochs,
          validation_data = test_dataloader, validation_steps=len(test_dataloader))
