import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from preprocess import basic_clean
from dataset import load_davidson_csv, OffensiveDataset, build_vocab, encode_text, PAD
from models.charcnn import build_char_vocab
from models.trifusion import TriFusionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_NAME = "bert-base-uncased"

def encode_chars(text, char_vocab, max_len=128):
    ids = [char_vocab.get(c, 0) for c in text.lower()[:max_len]]
    return torch.tensor(ids, dtype=torch.long)

def collate(batch, vocab, tokenizer, char_vocab, max_word_len=64, max_char_len=128):
    word_ids, char_ids, bert_inputs, labels = [], [], [], []

    for b in batch:
        t = b["text"]
        word_ids.append(torch.tensor(encode_text(t, vocab)[:max_word_len]))
        char_ids.append(encode_chars(t, char_vocab, max_char_len))
        labels.append(b["label"])

        enc = tokenizer(t, truncation=True, padding="max_length", max_length=max_word_len, return_tensors="pt")
        bert_inputs.append({k:v.squeeze(0) for k,v in enc.items()})

    word_ids = pad_sequence(word_ids, batch_first=True, padding_value=vocab[PAD])
    char_ids = pad_sequence(char_ids, batch_first=True, padding_value=char_vocab["<PAD>"])
    labels = torch.tensor(labels, dtype=torch.long)

    bert_batch = {k: torch.stack([x[k] for x in bert_inputs]) for k in bert_inputs[0]}
    return {"word": word_ids, "char": char_ids, "bert": bert_batch, "labels": labels}

def main():
    df = load_davidson_csv("data/davidson.csv")
    df["clean"] = df["tweet"].apply(basic_clean)

    X = df["clean"].tolist()
    y = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vocab = build_vocab(X_train)
    char_vocab = build_char_vocab()
    tokenizer = AutoTokenizer.from_pretrained(BERT_NAME)

    train_ds = OffensiveDataset(X_train, y_train)
    test_ds = OffensiveDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate(b, vocab, tokenizer, char_vocab)
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False,
        collate_fn=lambda b: collate(b, vocab, tokenizer, char_vocab)
    )

    model = TriFusionModel(
        vocab_size=len(vocab),
        num_classes=3,
        bert_name=BERT_NAME,
        char_vocab_size=len(char_vocab)
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            w = batch["word"].to(DEVICE)
            c = batch["char"].to(DEVICE)
            bert = {k:v.to(DEVICE) for k,v in batch["bert"].items()}
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            loss, _ = model(w, c, bert, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print("Epoch", epoch+1, "Loss", total_loss/len(train_loader))

    torch.save(model.state_dict(), "trifusion.pt")
    print("Saved model -> trifusion.pt")

if __name__ == "__main__":
    main()
