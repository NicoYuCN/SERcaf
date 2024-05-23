import torch
import torch.utils.data as tud
import pandas as pd
import numpy as np
import random
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from transformers import AutoFeatureExtractor


def audio_to_image(array, sr):
    top_db = 80
    y = array
    sr = sr
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=2048,
        n_mels=224,
        hop_length=512,
        win_length=None,
        window="hann")
    log_S = librosa.power_to_db(S, top_db=top_db)
    bytedata = (((log_S + top_db) * 255 / top_db).clip(0, 255) + 0.5).astype(np.uint8)
    image = Image.fromarray(bytedata)
    return image


class AudioDataset(tud.Dataset):
    def __init__(self, dataset_):
        self.dataset = dataset_
        self.samples = []
        self.transforms = transforms.Compose([
            transforms.Resize((0, 0)),
            transforms.ToTensor(),
            transforms.Normalize((0), (0))
        ])
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-ks")
        for tmp in self.dataset:
            y, sr, label = tmp
            audio = self.feature_extractor(y, sampling_rate=sr, return_tensors="pt")
            img = audio_to_image(y, sr)
            img = self.transforms(img)
            self.samples.append((audio, img, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio, img, label = self.samples[idx]
        return audio, img, label
