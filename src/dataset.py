import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter

PAD = "<PAD>"
UNK = "<UNK>"

def build_vocab(texts, min_freq=2, max_size=50000):
    counter = Counter()
    for t in texts:
        for tok in t.split():
            counter[tok] += 1

    vocab = {PAD: 0, UNK: 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[tok] = len(vocab)
    return vocab

def encode_text(text, vocab):
    return [vocab.get(tok, vocab[UNK]) for tok in text.split()]

class OffensiveDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "label": int(self.labels[idx])
        }

def load_davidson_csv(path):
    df = pd.read_csv(path)

    # Label construction from vote columns
    df["label"] = df[["hate_speech", "offensive_language", "neither"]].idxmax(axis=1)
    label_map = {"hate_speech": 0, "offensive_language": 1, "neither": 2}
    df["label"] = df["label"].map(label_map)

    return df
