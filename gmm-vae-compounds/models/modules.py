# General
import numpy as np
import pandas as pd
import os
import time
import sys
import re
import pickle
from functools import partial

import sklearn.metrics as metrics


# PyTorch
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as td
from torch.utils.data import Dataset, TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class FeedForwardThreeLayersConfigurableDropout(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["layers"][0], config["layers"][1])
        # self.fc1_bn = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=config["dropout_rate1"])
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(config["layers"][1], config["layers"][2])
        # self.fc2_bn = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=config["dropout_rate2"])
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(config["layers"][2], config["layers"][3])
        self.fc3_act = nn.ReLU()

        self.fc4 = nn.Linear(config["layers"][3], config["layers"][4])

        self.learning_rate = config["learning_rate"]
        self.l2_term = config["l2_term"]

    def forward(self, inputs):
        x = self.fc1_act(self.drop1(self.fc1(inputs)))
        x = self.fc2_act(self.drop2(self.fc2(x)))
        x = self.fc3_act(self.fc3(x))
        x = self.fc4(x)
        return x

    def loss_function(self, guiding_true, guiding_rec):
        return F.mse_loss(guiding_true, guiding_rec)

    def training_step(self, train_batch, batch_idx):
        input_X, guiding_X, _ = train_batch
        input_X, guiding_X = input_X.float(), guiding_X.float()
        guiding_rec = self.forward(input_X)
        loss = self.loss_function(guiding_X, guiding_rec)

        self.log("mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_term)
        return optim


class EncoderTwoLayersBatchNormConfigurable(nn.Module):
    def __init__(self, layers, var_transformation=lambda x: torch.exp(x) ** 0.5):
        super(EncoderTwoLayersBatchNormConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc1_bn = nn.BatchNorm1d(layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc2_bn = nn.BatchNorm1d(layers[2])

        self.mean_layer = nn.Linear(layers[2], layers[3])
        self.var_layer = nn.Linear(layers[2], layers[3])

        self.var_transformation = var_transformation

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars


class EncoderTwoLayersConfigurable(nn.Module):
    def __init__(self, layers, var_transformation=lambda x: torch.exp(x) ** 0.5):
        super(EncoderTwoLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])

        self.mean_layer = nn.Linear(layers[2], layers[3])
        self.var_layer = nn.Linear(layers[2], layers[3])

        self.var_transformation = var_transformation

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars


class EncoderThreeLayersBatchNormConfigurable(nn.Module):
    def __init__(self, layers, var_transformation=lambda x: torch.exp(x) ** 0.5):
        super(EncoderThreeLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc1_bn = nn.BatchNorm1d(layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc2_bn = nn.BatchNorm1d(layers[2])

        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc3_bn = nn.BatchNorm1d(layers[3])

        self.mean_layer = nn.Linear(layers[3], layers[4])
        self.var_layer = nn.Linear(layers[3], layers[4])

        self.var_transformation = var_transformation

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars


class EncoderThreeLayersConfigurable(nn.Module):
    def __init__(self, layers, var_transformation=lambda x: torch.exp(x) ** 0.5, dropout_rate=0.0):
        super(EncoderThreeLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(layers[2], layers[3])

        self.mean_layer = nn.Linear(layers[3], layers[4])
        self.var_layer = nn.Linear(layers[3], layers[4])

        self.var_transformation = var_transformation

    def forward(self, input):
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        # Apply dropout after second layer
        x = self.drop2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        z_means = self.mean_layer(x)
        z_log_vars = self.var_layer(x)
        z_vars = self.var_transformation(z_log_vars)

        return z_means, z_vars


class DecoderTwoLayersBatchNormConfigurable(nn.Module):
    def __init__(self, layers):
        super(DecoderTwoLayersBatchNormConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc1_bn = nn.BatchNorm1d(layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc2_bn = nn.BatchNorm1d(layers[2])

        self.fc3 = nn.Linear(layers[2], layers[3])

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)

        return x


class DecoderTwoLayersConfigurable(nn.Module):
    def __init__(self, layers):
        super(DecoderTwoLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])

        self.fc3 = nn.Linear(layers[2], layers[3])

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x


class DecoderThreeLayersBatchNormConfigurable(nn.Module):
    def __init__(self, layers):
        super(DecoderThreeLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc1_bn = nn.BatchNorm1d(layers[1])

        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc2_bn = nn.BatchNorm1d(layers[2])

        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc3_bn = nn.BatchNorm1d(layers[3])

        self.fc4 = nn.Linear(layers[3], layers[4])

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        x = self.fc4(x)

        return x


class DecoderThreeLayersConfigurable(nn.Module):
    def __init__(self, layers, dropout_rate=0.0):
        super(DecoderThreeLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.fc4 = nn.Linear(layers[3], layers[4])

    def forward(self, z_sample):
        x = self.fc1(z_sample)
        x = F.relu(x)
        x = self.fc2(x)
        # Apply dropout after second layer
        x = self.drop2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)

        return x
    
class EncoderDecoderTwoLayersConfigurable(nn.Module):
    def __init__(self, layers):
        super(EncoderDecoderTwoLayersConfigurable, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])
        
    def forward(self, input):
        # Input layer
        x = self.fc1(input)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        # Code layer
        x = self.fc3(x)
        
        return x
    
class AutoencoderConfigurable(pl.LightningModule):
    def __init__(self, encoder_layers, decoder_layers, learning_rate=0.001, batch_norm=False, optimizer="adam",
                 encoder_dropout_rate=0, decoders_dropout_rate=0):
        
        super(AutoencoderConfigurable, self).__init__()
        # Establish encoder
        if len(encoder_layers) == 4:
            if batch_norm:
                self.encoder = EncoderTwoLayersBatchNormConfigurable(encoder_layers, var_transformation)
            else:
                self.encoder = EncoderDecoderTwoLayersConfigurable(encoder_layers)    
        # Establish decoder
        if len(decoder_layers) == 4:
            if batch_norm:
                self.decoder = EncoderDecoderTwoLayersConfigurable(decoder_layers)
            else:
                self.decoder = EncoderDecoderTwoLayersConfigurable(decoder_layers)


        self.latent_dim = encoder_layers[-1]
        self.learning_rate = learning_rate

        self.optimizer = optimizer
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        
        return reconstruction, latent
    
    def loss_function(self, input_true, input_rec):
        return F.mse_loss(input_true, input_rec)   # NOTE: maybe modified to accept inputs with nans


class FeedForwardThreeLayersConfigurable(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["layers"][0], config["layers"][1])
        # self.fc1_bn = nn.BatchNorm1d(512)
        self.fc1_act = nn.ReLU()

        self.fc2 = nn.Linear(config["layers"][1], config["layers"][2])
        # self.fc2_bn = nn.BatchNorm1d(256)
        self.fc2_act = nn.ReLU()

        self.fc3 = nn.Linear(config["layers"][2], config["layers"][3])
        self.fc3_act = nn.ReLU()

        self.fc4 = nn.Linear(config["layers"][3], config["layers"][4])

        self.learning_rate = config["learning_rate"]
        self.l2_term = config["l2_term"]

    def forward(self, inputs):
        x = self.fc1_act(self.fc1(inputs))
        x = self.fc2_act(self.fc2(x))
        x = self.fc3_act(self.fc3(x))
        x = self.fc4(x)
        return x

    def loss_function(self, guiding_true, guiding_rec):
        return F.mse_loss(guiding_true, guiding_rec)

    def training_step(self, train_batch, batch_idx):
        input_X, guiding_X, _ = train_batch
        input_X, guiding_X = input_X.float(), guiding_X.float()
        guiding_rec = self.forward(input_X)
        loss = self.loss_function(guiding_X, guiding_rec)

        self.log("mse_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_term)
        return optim