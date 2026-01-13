import torch
import torch.nn as nn
from transformers import AutoModel
from .cnn_gru import CNN_GRU
from .charcnn import CharCNN

class TriFusionModel(nn.Module):
    def __init__(self, vocab_size, num_classes, bert_name="bert-base-uncased", fusion_dim=256, char_vocab_size=64):
        super().__init__()

        self.word_model = CNN_GRU(vocab_size, num_classes)
        self.word_model.fc = nn.Identity()

        self.char_model = CharCNN(vocab_size=char_vocab_size)
        self.bert = AutoModel.from_pretrained(bert_name)

        self.proj_word = nn.Linear(256, fusion_dim)
        self.proj_char = nn.Linear(self.char_model.out_dim, fusion_dim)
        self.proj_bert = nn.Linear(768, fusion_dim)

        self.gate = nn.Linear(fusion_dim * 3, 3)
        self.fc = nn.Linear(fusion_dim, num_classes)

    def forward(self, word_ids, char_ids, bert_input, labels=None):
        w = self.word_model(word_ids)
        c = self.char_model(char_ids)
        b = self.bert(**bert_input).last_hidden_state[:,0,:]

        w = self.proj_word(w)
        c = self.proj_char(c)
        b = self.proj_bert(b)

        concat = torch.cat([w,c,b], dim=1)
        g = torch.softmax(self.gate(concat), dim=1)

        channels = torch.stack([w,c,b], dim=1)
        fused = torch.sum(g.unsqueeze(-1) * channels, dim=1)

        logits = self.fc(fused)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits

        return logits
