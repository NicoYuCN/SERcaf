from datasets import load_dataset, Audio
import torch
import random
import numpy as np
import torch


def load_emodb():
    dataloader_audio = load_dataset('renumics/emodb', split='all')
    dataset = []

    for pos in range(len(dataloader_audio)):
        dataset.append([dataloader_audio[pos]["audio"]["array"], dataloader_audio[pos]["audio"]["sampling_rate"],
                        dataloader_audio[pos]["emotion"]])

    random.shuffle(dataset)
    pos = int(len(dataset) * 0.8)
    train_dataset = dataset[:pos]
    test_dataset = dataset[pos:]
    print(len(train_dataset))
    print(len(test_dataset))
    return train_dataset, test_dataset


def load_ravdess():
    dataset_ravdess = load_dataset("narad/ravdess", split="train")
    dataset_ravdess = dataset_ravdess.train_test_split(test_size=0.2)
    minds = dataset_ravdess.cast_column("audio", Audio(sampling_rate=16_000))
    train_dataset = minds["train"]
    test_dataset = minds["test"]
    print(len(train_dataset))
    print(len(test_dataset))
    return train_dataset, test_dataset


def load_iemocap():
    dataloader_audio = load_dataset("minoosh/IEMOCAP_Speech_dataset")["train"]
    dataset = []

    for pos in range(len(dataloader_audio)):
        dataset.append([dataloader_audio[pos]["audio"]["array"], dataloader_audio[pos]["audio"]["sampling_rate"],
                        dataloader_audio[pos]["emotion"]])

    random.shuffle(dataset)
    pos = int(len(dataset) * 0.8)
    train_dataset = dataset[:pos]
    test_dataset = dataset[pos:]
    print(len(train_dataset))
    print(len(test_dataset))
    return train_dataset, test_dataset
