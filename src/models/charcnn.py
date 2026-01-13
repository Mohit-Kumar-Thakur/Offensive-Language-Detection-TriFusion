import torch
import torch.nn as nn
import string

def build_char_vocab():
    chars = list(string.ascii_lowercase + string.digits + "@#$%&*!?")
    vocab = {c:i+1 for i,c in enumerate(chars)}
    vocab["<PAD>"] = 0
    return vocab

class CharCNN(nn.Module):
    def __init__(self, vocab_size, emb_dim=30, num_filters=64, kernels=(3,5,7)):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernels])
        self.out_dim = num_filters * len(kernels)

    def forward(self, ids):
        x = self.emb(ids).transpose(1,2)
        conv_out = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return torch.cat(conv_out, dim=1)
