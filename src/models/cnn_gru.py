import torch
import torch.nn as nn

class CNN_GRU(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        emb_dim=200,
        num_filters=128,
        kernels=(3,4,5),
        gru_hidden=128,
        dropout=0.5,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernels])

        self.gru = nn.GRU(
            input_size=num_filters * len(kernels),
            hidden_size=gru_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden * 2, num_classes)

    def forward(self, ids, labels=None):
        x = self.embedding(ids)          # (B,T,E)
        x = x.transpose(1,2)             # (B,E,T)

        conv_feats = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        x = torch.cat(conv_feats, dim=1) # (B, F*K)

        # small pseudo-seq for GRU
        x = x.unsqueeze(1).repeat(1, 3, 1)

        _, h = self.gru(x)
        h = torch.cat([h[0], h[1]], dim=1)

        h = self.dropout(h)
        logits = self.fc(h)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits
