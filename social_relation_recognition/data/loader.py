"""
    Custom Pytorch dataloader.
"""


import random

import numpy
import torch
import torchvision.transforms as transforms
from tools.utils_dataset import dataset_type2mean, dataset_type2std
from torch.utils.data import DataLoader

from data.dataset import SRDataset


def my_collate(batch):
    """Custom collate function."""

    list_ids = []
    list_personal = []
    list_local = []
    list_global = []
    list_labels = []

    for item in batch:
        list_ids.append(item[0])
        list_personal.append(item[1])
        list_local.append(item[2])
        list_global.append(item[3])
        list_labels.append(item[4])

    batch_personal = torch.cat(list_personal, 0)
    batch_local = torch.cat(list_local, 0)
    batch_gobal = torch.stack(list_global, 0)
    batch_labels = torch.cat(list_labels, 0)

    return list_ids, batch_personal, batch_local, batch_gobal, batch_labels


def initialize_transformer(dataset, type, normalization, size):
    """Get data transformers."""

    mean = [0.485, 0.456, 0.406] if (
        normalization == 'ImageNet') else dataset_type2mean[dataset][type]
    std = [0.229, 0.224, 0.225] if (
        normalization == 'ImageNet') else dataset_type2std[dataset][type]

    normalize = transforms.Normalize(mean=mean, std=std)

    transformer = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize])

    return transformer


def get_test_loader(dataset, path_images, ids_images, metadata, type, size_input, size_batch, normalization, number_workers):
    """Get test loader."""

    transformer = initialize_transformer(
        dataset, type, normalization, size_input)

    test_set = SRDataset(path_images, ids_images, metadata,
                         type, transformer, transformer, transformer)
    test_loader = DataLoader(dataset=test_set, num_workers=number_workers, batch_size=size_batch,
                             shuffle=False, collate_fn=my_collate, worker_init_fn=seed_worker)

    return test_loader


def get_train_loader(dataset, path_images, ids_images, metadata, type, size_input, size_batch, normalization, number_workers):
    """Get train loader."""

    transformer = initialize_transformer(
        dataset, type, normalization, size_input)

    train_set = SRDataset(path_images, ids_images, metadata,
                          type, transformer, transformer, transformer)
    train_loader = DataLoader(dataset=train_set, num_workers=number_workers, batch_size=size_batch,
                              shuffle=True, collate_fn=my_collate, worker_init_fn=seed_worker)

    return train_loader


def seed_worker(worker_id):
    """Set worker seeds for reproduction."""

    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
