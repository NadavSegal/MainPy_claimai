from torchvision import datasets, transforms
from base import BaseDataLoader

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ClaimaiLoader(BaseDataLoader):
    """
    data in csv file for claim ai prediction
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        # self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        self.dataset = ClaimaiDataset(self.data_dir)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def to_hot(ds, to_one_hot_fields):

    for i, field in enumerate(to_one_hot_fields):
        one_hot = pd.get_dummies(ds[field])
        for col in one_hot.columns:
            one_hot.rename(columns={str(col): str(col) + '_' + str(i)}, inplace=True)
        ds = ds.drop(field, axis=1)
        ds = ds.join(one_hot)

    return ds


def incident_date_to_month(ds):
    ds['incident_month'] = pd.to_datetime(ds['incident_date']).dt.month
    ds['incident_day'] = pd.to_datetime(ds['incident_date']).dt.day
    ds = ds.drop('incident_date', axis=1)
    return ds


def ds_process(ds):
    remove_fields = [
        'months_as_customer',
        'policy_number',
        'policy_bind_date',
        'policy_bind_date',
        'policy_deductable',
        'policy_annual_premium',
        'insured_sex',
        'insured_education_level',
        'insured_occupation',
        'insured_hobbies',
        'insured_relationship',
        'incident_location'
        ]
    to_one_hot_fields = [
        'incident_type',
        'age',
        'incident_month',
        'incident_day',
        'collision_type',
        'incident_severity',
        'authorities_contacted',
        'incident_state',
        'incident_city',
        'incident_hour_of_the_day',
        'number_of_vehicles_involved',
        'property_damage',
        'bodily_injuries',
        'witnesses',
        'police_report_available',
        'auto_make',
        'auto_model',
        'auto_year',
        'fraud_reported'
    ]

    ds = incident_date_to_month(ds)
    ds = ds.drop(remove_fields, axis=1)
    ds = ds.astype(str)
    ds = to_hot(ds, to_one_hot_fields)

    return ds


class ClaimaiDataset(Dataset):

    def __init__(self, data_dir):

        # read csv file and load row data into variables
        ds = pd.read_csv(data_dir + '/insurance_claims_clean.csv')
        ds = ds_process(ds)

        # y = ds.iloc[0:, 0:4].values
        y = ds.iloc[0:, 0:1].values
        x = ds.iloc[0:, 4:].values
        # x = ds.iloc[0:, 0:1].values

        # Frature Scaling
        # sc = StandardScaler(x)
        # x = sc.fit_transform(x)
        x_train = x.astype(int)
        y_train = y.astype(int)

        # converting to torch tensors
        # self.x_train = torch.tensor(x_train, device=0, dtype=torch.float32)
        # self.y_train = torch.tensor(y_train, device=0, dtype=torch.float32)
        self.x_train = torch.from_numpy(x_train).float().to(0)
        self.y_train = torch.from_numpy(y_train).float().to(0)


    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]
