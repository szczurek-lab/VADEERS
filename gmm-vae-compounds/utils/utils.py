# General
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import os
# import time
import sys
# import re
import pickle
# import itertools
import json
# from functools import partial

#scikit-learn
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.datasets import make_blobs
# from sklearn.decomposition import PCA, TruncatedSVD
# from sklearn import preprocessing
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.cluster import KMeans
# from sklearn.ensemble import RandomForestRegressor
# import sklearn.metrics as metrics


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
from pytorch_lightning.callbacks import Callback

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import StepLR

class FreezingCallback(Callback):
    def __init__(self, freeze_epoch=50, modules_to_freeze=["drug_model", "cell_line_model"], new_learning_rate=0.001,
                step_lr_scheduler=10, step_size=10, gamma=1.):
        super().__init__()
        self.freeze_epoch = freeze_epoch
        self.modules_to_freeze = modules_to_freeze
        self.new_learning_rate = new_learning_rate
        self.step_lr_scheduler = step_lr_scheduler
        self.step_size = step_size
        self.steps = 0
        self.gamma = gamma

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch + 1 == self.freeze_epoch:
            # Freeze parts of the model
            pl_module.vae_dataloader = None
            pl_module.cell_line_model.freeze()
            pl_module.drug_model.freeze()
            self.current_learning_rate = self.new_learning_rate
            self.steps = 0
            optimizer = torch.optim.Adam(pl_module.parameters(), lr=self.new_learning_rate)
            trainer.optimizers = [optimizer]
            print("Executing freezing")
        self.update_lr(trainer, pl_module)
        self.steps = self.steps + 1
        
    def decay_lr(self):
        self.current_learning_rate = self.gamma * self.current_learning_rate
        
    def update_lr(self, trainer, pl_module):
        if (trainer.current_epoch + 1 > self.freeze_epoch) and (self.steps % self.step_size == 0):
            self.decay_lr()
            optimizer = torch.optim.Adam(pl_module.parameters(), lr=self.current_learning_rate)
            trainer.optimizers = [optimizer]
            self.steps = 0
            
    def configure_new_optimizers(self, pl_module):
        return [torch.optim.Adam(pl_module.parameters(), lr=self.new_learning_rate)]



def get_drug_ID_guiding_class_label(data, drug_ID_col_name="PubChem CID", guiding_class_col_name="guiding_cluster_label"):
    res = {}
    for row in data.iterrows():
        drug_ID = row[1][drug_ID_col_name]
        cl_label = row[1][guiding_class_col_name]
        res[drug_ID] = cl_label
        
    return res

def generate_zs_from_cluster(drug_model, cluster_number, n_samples=100, model_as_std=True,
                            seed=None):
    """Generate samples from latent."""
    drug_model.eval()
    # Get means of appropriate Gaussian from the mixture in latent
    means = drug_model.means[cluster_number]
    # Get stds
    stds = drug_model.stds[cluster_number]
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Create Multivariate Gauss to sample from
    if model_as_std:
        rv = td.MultivariateNormal(means, torch.diag(stds ** 2))
    else:
        rv = td.MultivariateNormal(means, torch.diag(stds))

    # Sample
    cluster_z_sample = rv.sample([n_samples])
    return cluster_z_sample


def generate_from_cluster(drug_model, cluster_number, n_samples=100, model_as_std=True, seed=None):
    """Generate samples from latent and put them into decoders."""
    drug_model.eval()
    # Get means of appropriate Gaussian from the mixture in latent
    means = drug_model.means[cluster_number]
    # Get stds
    stds = drug_model.stds[cluster_number]
    
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create Multivariate Gauss to sample from
    if model_as_std:
        rv = td.MultivariateNormal(means, torch.diag(stds ** 2))
    else:
        rv = td.MultivariateNormal(means, torch.diag(stds))

    # Sample
    cluster_z_sample = rv.sample([n_samples])

    # Get input reconstructions
    input_reconstructions = drug_model.input_decoder(cluster_z_sample)
    # Get guiding reconstructions
    guiding_reconstructions = drug_model.guiding_decoder(cluster_z_sample)
    
    return input_reconstructions, guiding_reconstructions


def get_cell_lines_intersection_table(sensitivity_table, num_drugs, 
                                      drug_id_col_name="PubChem CID", 
                                      cell_line_id_col_name="COSMIC_ID", 
                                      sensitivity_metric_col_name="LN_IC50"):
    
    """Find apprpriate cell lines set common for a specified number of drugs."""
    # Create drug: set of cell lines dict
    drug_screened_cls_dict = {}
    # Placeholder for the biggest set (initial cell lines intersection set)
    intersection_set = set()
    biggest_set_drug = None
    for drug in sensitivity_table[drug_id_col_name].unique():
        screened_cell_lines = set(sensitivity_table[sensitivity_table[drug_id_col_name] == drug][cell_line_id_col_name])
        drug_screened_cls_dict[drug] = screened_cell_lines
        if len(screened_cell_lines) > len(intersection_set):
            intersection_set = screened_cell_lines
            biggest_set_drug = drug

    drugs_set = set([biggest_set_drug])
    for i in range(num_drugs - 1):
        # Search for most optimal intersection
        biggest_intersection = set()
        biggest_intersection_drug = None
        for drug in drug_screened_cls_dict:
            if drug not in drugs_set:
                candidate_intersection = intersection_set.intersection(drug_screened_cls_dict[drug])
                if len(candidate_intersection) > len(biggest_intersection):
                    biggest_intersection = candidate_intersection
                    biggest_intersection_drug = drug

        # After finding the most optimal sets, update the values
        intersection_set = biggest_intersection
        drugs_set.add(biggest_intersection_drug)

    # Create drug - sensitivity profiles table for these sets
    sensitivity_table_intersection = pd.pivot_table(sensitivity_table, values=sensitivity_metric_col_name, index=drug_id_col_name, 
                                                    columns=cell_line_id_col_name)[intersection_set].loc[drugs_set]

    return sensitivity_table_intersection

# Function for getting mappers from drug or cell line ID to index in the given data table
def get_ID_to_idx_mapper(df, id_col):
    """Function for getting mappers from drug or cell line ID to index in the given data table."""
    mapper = {}
    idx = 0
    for row in df.iterrows():
        if id_col == "index":
            ide = row[0]
        else:
            ide = row[1][id_col]
        mapper[int(ide)] = idx
        idx += 1
    return mapper

def get_vae_dataset(vae_input_data, vae_guiding_data, id_column_name="PubChem CID", guiding_class_col_name="guiding_cluster_label"):
    input_cols = vae_input_data.columns
    guiding_cols = vae_guiding_data.columns
    data = vae_guiding_data.merge(vae_input_data, on=id_column_name, how="inner")
    input_data = data[[x for x in input_cols if x not in (id_column_name, guiding_class_col_name)]].values
    guiding_data = data[[x for x in guiding_cols if x not in (id_column_name, guiding_class_col_name)]].values
    guiding_classes = data[guiding_class_col_name].values

    dataset = TensorDataset(torch.Tensor(input_data), torch.Tensor(guiding_data), torch.Tensor(guiding_classes))
    return dataset


class DatasetThreeTables(Dataset):
    """
    Dataset with three tables: cell lines features, compounds featureas and sensitivity table.
    """
    
    def __init__(self, sensitivity_table, cell_lines_biological_data, drugs_smiles_representations, drugs_inhib_profiles,
                 cell_line_ID_to_index_mapper, drugs_ID_to_smiles_rep_index_mapper, drugs_ID_to_inhib_profiles_index_mapper,
                 drug_ID_guiding_no_inhib_guiding_class_mapper=None,
                 drug_ID_name="PubChem CID", cell_line_ID_name="COSMIC_ID", guiding_data_class_name="guiding_data_class",
                 sensitivity_metric="LN_IC50", drug_ID_index=0, cell_line_ID_index=1, sensitivity_metric_index=2):
        
        self.sensitivity_table = sensitivity_table
        self.sensitivity_table_array = sensitivity_table.values
        self.cell_lines_biological_data = cell_lines_biological_data
        self.drugs_smiles_representations = drugs_smiles_representations
        self.drugs_inhib_profiles = drugs_inhib_profiles
        
        self.cell_line_ID_to_index_mapper = cell_line_ID_to_index_mapper
        self.drugs_ID_to_smiles_rep_index_mapper = drugs_ID_to_smiles_rep_index_mapper
        self.drugs_ID_to_inhib_profiles_index_mapper = drugs_ID_to_inhib_profiles_index_mapper
        
        self.drug_ID_name = drug_ID_name
        self.cell_line_ID_name = cell_line_ID_name
        self.guiding_data_class_name = guiding_data_class_name
        self.sensitivity_metric = sensitivity_metric
        
        self.drug_ID_index = drug_ID_index
        self.cell_line_ID_index = cell_line_ID_index
        self.sensitivity_metric_index = sensitivity_metric_index
        
        self.drug_ID_guiding_no_inhib_guiding_class_mapper = drug_ID_guiding_no_inhib_guiding_class_mapper
        
    def __len__(self):
        return len(self.sensitivity_table)
    
    def __getitem__(self, idx):
        sensitivity_table_entry = self.sensitivity_table_array[idx]
        pubchem_id, cosmic_id, response = sensitivity_table_entry[self.drug_ID_index], sensitivity_table_entry[self.cell_line_ID_index], sensitivity_table_entry[self.sensitivity_metric_index]
        
        cell_line_data_batch = self.cell_lines_biological_data[self.cell_line_ID_to_index_mapper[cosmic_id], :]
        
        
        smiles_data_batch = self.drugs_smiles_representations[self.drugs_ID_to_smiles_rep_index_mapper[pubchem_id]]
        if pubchem_id in self.drugs_ID_to_inhib_profiles_index_mapper:
            inhib_profiles_batch = self.drugs_inhib_profiles[self.drugs_ID_to_inhib_profiles_index_mapper[pubchem_id]][:-1]   # Guiding class at the end
            guiding_data_class = self.drugs_inhib_profiles[self.drugs_ID_to_inhib_profiles_index_mapper[pubchem_id]][-1]
        else:
            inhib_profiles_batch = np.empty((self.drugs_inhib_profiles.shape[1] - 1,))
            inhib_profiles_batch.fill(np.nan)
            guiding_data_class = np.nan
            
        if self.drug_ID_guiding_no_inhib_guiding_class_mapper is None:  
            if pubchem_id in self.drugs_ID_to_inhib_profiles_index_mapper:
                guiding_data_class = self.drugs_inhib_profiles[self.drugs_ID_to_inhib_profiles_index_mapper[pubchem_id]][-1]
            else:
                guiding_data_class = np.nan
                
        else:
            if pubchem_id in self.drug_ID_guiding_no_inhib_guiding_class_mapper:
                guiding_data_class = self.drug_ID_guiding_no_inhib_guiding_class_mapper[pubchem_id]
            else:
                guiding_data_class = np.nan
        
        return smiles_data_batch, inhib_profiles_batch, guiding_data_class, cell_line_data_batch, torch.Tensor([response]), pubchem_id, cosmic_id
        
    
    def train_val_test_split(self, num_cell_lines_val, num_cell_lines_test, seed=None, return_cell_lines=False):
        # Set the seed for random generator
        np.random.seed(seed)
        
        train_cell_lines = set(self.sensitivity_table[self.cell_line_ID_name])
        val_cell_lines = set(np.random.choice(list(train_cell_lines), size=num_cell_lines_val, replace=False))
        train_cell_lines = train_cell_lines.difference(val_cell_lines)

        test_cell_lines = set(np.random.choice(list(train_cell_lines), size=num_cell_lines_test, replace=False))
        train_cell_lines = train_cell_lines.difference(test_cell_lines)

        sensitivity_table_train = self.sensitivity_table[self.sensitivity_table[self.cell_line_ID_name].isin(train_cell_lines)]
        sensitivity_table_val = self.sensitivity_table[self.sensitivity_table[self.cell_line_ID_name].isin(val_cell_lines)]
        sensitivity_table_test = self.sensitivity_table[self.sensitivity_table[self.cell_line_ID_name].isin(test_cell_lines)]
        
        # Create corresponding Dataset instances
        dataset_train = DatasetThreeTables(sensitivity_table_train, self.cell_lines_biological_data, 
                                  self.drugs_smiles_representations, 
                                  self.drugs_inhib_profiles,
                                
                                  self.cell_line_ID_to_index_mapper, self.drugs_ID_to_smiles_rep_index_mapper, self.drugs_ID_to_inhib_profiles_index_mapper,
                                  drug_ID_guiding_no_inhib_guiding_class_mapper = self.drug_ID_guiding_no_inhib_guiding_class_mapper,
                                  drug_ID_name=self.drug_ID_name, cell_line_ID_name=self.cell_line_ID_name, guiding_data_class_name=self.guiding_data_class_name,
                                  sensitivity_metric=self.sensitivity_metric,
                                  drug_ID_index=self.drug_ID_index, cell_line_ID_index=self.cell_line_ID_index, 
                                  sensitivity_metric_index=self.sensitivity_metric_index)
        dataset_val = DatasetThreeTables(sensitivity_table_val, self.cell_lines_biological_data, 
                                  self.drugs_smiles_representations, 
                                  self.drugs_inhib_profiles,
                                  self.cell_line_ID_to_index_mapper, self.drugs_ID_to_smiles_rep_index_mapper, self.drugs_ID_to_inhib_profiles_index_mapper,
                                         drug_ID_guiding_no_inhib_guiding_class_mapper = self.drug_ID_guiding_no_inhib_guiding_class_mapper,
                                  drug_ID_name=self.drug_ID_name, cell_line_ID_name=self.cell_line_ID_name, guiding_data_class_name=self.guiding_data_class_name,
                                  sensitivity_metric=self.sensitivity_metric,
                                  drug_ID_index=self.drug_ID_index, cell_line_ID_index=self.cell_line_ID_index, 
                                  sensitivity_metric_index=self.sensitivity_metric_index)
        dataset_test = DatasetThreeTables(sensitivity_table_test, self.cell_lines_biological_data, 
                                  self.drugs_smiles_representations, 
                                  self.drugs_inhib_profiles,
                                  self.cell_line_ID_to_index_mapper, self.drugs_ID_to_smiles_rep_index_mapper, self.drugs_ID_to_inhib_profiles_index_mapper,
                                          drug_ID_guiding_no_inhib_guiding_class_mapper = self.drug_ID_guiding_no_inhib_guiding_class_mapper,
                                  drug_ID_name=self.drug_ID_name, cell_line_ID_name=self.cell_line_ID_name, guiding_data_class_name=self.guiding_data_class_name,
                                  sensitivity_metric=self.sensitivity_metric,
                                  drug_ID_index=self.drug_ID_index, cell_line_ID_index=self.cell_line_ID_index, 
                                  sensitivity_metric_index=self.sensitivity_metric_index)
        
        if return_cell_lines:
            return dataset_train, dataset_val, dataset_test, train_cell_lines, val_cell_lines, test_cell_lines
        else:
            return dataset_train, dataset_val, dataset_test
        
    def train_test_split(self, num_cell_lines_test, seed=None, return_cell_lines=False):
        # Set the seed for random generator
        np.random.seed(seed)
        
        train_cell_lines = set(self.sensitivity_table[self.cell_line_ID_name])
        test_cell_lines = set(np.random.choice(list(train_cell_lines), size=num_cell_lines_test, replace=False))
        train_cell_lines = train_cell_lines.difference(test_cell_lines)

        sensitivity_table_train = self.sensitivity_table[self.sensitivity_table[self.cell_line_ID_name].isin(train_cell_lines)]
        sensitivity_table_test = self.sensitivity_table[self.sensitivity_table[self.cell_line_ID_name].isin(test_cell_lines)]
        
        # Create corresponding Dataset instances
        dataset_train = DatasetThreeTables(sensitivity_table_train, self.cell_lines_biological_data, 
                                  self.drugs_smiles_representations, 
                                  self.drugs_inhib_profiles,
                                
                                  self.cell_line_ID_to_index_mapper, self.drugs_ID_to_smiles_rep_index_mapper, self.drugs_ID_to_inhib_profiles_index_mapper,
                                  drug_ID_guiding_no_inhib_guiding_class_mapper = self.drug_ID_guiding_no_inhib_guiding_class_mapper,
                                  drug_ID_name=self.drug_ID_name, cell_line_ID_name=self.cell_line_ID_name, guiding_data_class_name=self.guiding_data_class_name,
                                  sensitivity_metric=self.sensitivity_metric,
                                  drug_ID_index=self.drug_ID_index, cell_line_ID_index=self.cell_line_ID_index, 
                                  sensitivity_metric_index=self.sensitivity_metric_index)

        dataset_test = DatasetThreeTables(sensitivity_table_test, self.cell_lines_biological_data, 
                                  self.drugs_smiles_representations, 
                                  self.drugs_inhib_profiles,
                                  self.cell_line_ID_to_index_mapper, self.drugs_ID_to_smiles_rep_index_mapper, self.drugs_ID_to_inhib_profiles_index_mapper,
                                          drug_ID_guiding_no_inhib_guiding_class_mapper = self.drug_ID_guiding_no_inhib_guiding_class_mapper,
                                  drug_ID_name=self.drug_ID_name, cell_line_ID_name=self.cell_line_ID_name, guiding_data_class_name=self.guiding_data_class_name,
                                  sensitivity_metric=self.sensitivity_metric,
                                  drug_ID_index=self.drug_ID_index, cell_line_ID_index=self.cell_line_ID_index, 
                                  sensitivity_metric_index=self.sensitivity_metric_index)
        
        if return_cell_lines:
            return dataset_train, dataset_test, train_cell_lines, test_cell_lines
        else:
            return dataset_train, dataset_test
        
        
# Encoder variance transformations
def square_plus(x, min_std=0.1):
    return min_std + F.softplus(x) ** 0.5


def var_trans_reverse_gauss(x, min_std=0.1, max_std=1):
    return min_std + max_std - torch.exp(-torch.pow(x, 2)) * max_std