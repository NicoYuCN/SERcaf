import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import Wav2Vec2ForSequenceClassification


class BiLSTM(nn.Module):
    def __init__(self, input_size=0, num_class=0):
        super().__init__()
        self.fc0 = nn.Linear(in_features=input_size, out_features=768)
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features=768, out_features=384)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=384, out_features=num_class)


    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, h, w)
        x = self.fc0(x)
        y, (h, c) = self.lstm(x)
        hidden_last_L = h[-2]
        hidden_last_R = h[-1]
        x = torch.cat([hidden_last_L, hidden_last_R], dim=-1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DPCNN(nn.Module):
    def __init__(self, embed_dim=0, class_num=0):
        super(DPCNN, self).__init__()
        ci = 1
        kernel_num = 250
        self.conv_region = nn.Conv2d(ci, kernel_num, (3, embed_dim), stride=1)
        self.conv = nn.Conv2d(kernel_num, kernel_num, (3, 1), stride=1)
        self.max_pool = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2, 1))
        self.padding = nn.ZeroPad2d((0, 0, 1, 1))  # top bottom
        self.relu = nn.ReLU()
        self.fc = nn.Linear(kernel_num, class_num)

    def forward(self, x):
        m = self.conv_region(x)
        x = self.padding(m)
        x = self.relu(x)
        x = self.conv(x)
        x = self.padding(x)
        x = self.relu(x)
        x = self.conv(x)
        x = x+m
        while x.size()[2] > 2:
            x = self._block(x)
        if x.size()[2] == 2:
            x = self.max_pool_2(x)
        x = x.reshape(x.size()[0], x.size()[1])
        x = self.fc(x)
        return x

    def _block(self, x):
        px = self.max_pool(x)
        x = self.padding(px)
        x = F.relu(x)
        x = self.conv(x)
        x = self.padding(x)
        x = F.relu(x)
        x = self.conv(x)
        x = x + px
        return x


class Wav_TextRCNN(nn.Module):
    def __init__(self, int_dim, out_dim):
        super().__init__()
        self.wav = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks").to(device)
        self.bilstm = BiLSTM().to(device)
        self.fc = nn.Linear(int_dim, out_dim)

    #         self.fc = FiLM()
    #         self.fc = CrossMutilFeatureAttention()

    def forward(self, x, y):
        logits = self.wav(**x).logits
        hidden = self.bilstm(y)
        out = torch.cat((logits, hidden), dim=-1)
        out = self.fc(out)
        #         out = self.fc(logits + hidden)
        #         out = self.fc(logits, hidden)
        return out


class Wav_DPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-ks").to(device)
        self.dpcnn = DPCNN().to(device)
        #         self.fc = nn.Linear(12*2, 7)
        #         self.fc = FiLM()
        self.fc = CrossMutilFeatureAttention()

    def forward(self, x, y):
        logits = self.wav(**x).logits
        hidden = self.dpcnn(y)
        #         out = torch.cat((logits, hidden), dim=-1)
        #         out = self.fc(out)
        #         out = self.fc(logits + hidden)
        out = self.fc(logits, hidden)
        return out


class HuBERT_TextRCNN(nn.Module):
    def __init__(self, int_dim, out_dim):
        super().__init__()
        self.HuBERT = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks").to(device)
        self.bilstm = BiLSTM().to(device)
        self.fc = nn.Linear(int_dim, out_dim)

    #         self.fc = FiLM()
    #         self.fc = CrossMutilFeatureAttention()

    def forward(self, x, y):
        logits = self.HuBERT(**x).logits
        hidden = self.bilstm(y)
        out = self.fc(logits + hidden)
        #         out = self.fc(logits, hidden)
        return out


class HuBERT_DPCNN(nn.Module):
    def __init__(self, int_dim, out_dim):
        super().__init__()
        self.HuBERT = HubertForSequenceClassification.from_pretrained("superb/hubert-base-superb-ks").to(device)
        self.dpcnn = DPCNN().to(device)
        self.fc = nn.Linear(int_dim, out_dim)

    #         self.fc = FiLM()
    #         self.fc = CrossMutilFeatureAttention()

    def forward(self, x, y):
        logits = self.HuBERT(**x).logits
        hidden = self.dpcnn(y)
        out = self.fc(logits + hidden)
        #         out = self.fc(logits, hidden)
        return out
