import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
import pickle

nltk.download("punkt")


# -----------------------------
# Dataset
# -----------------------------
class SSTDataset(Dataset):
    def __init__(self, data, vocab, max_len=50):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = word_tokenize(text.lower())[: self.max_len]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return torch.tensor(ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encode(item["text"]), item["label"]


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([len(t) for t in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True)
    return texts, lengths, torch.tensor(labels)


# -----------------------------
# Vocabulary + GloVe
# -----------------------------
def build_vocab(dataset, max_size=10000):
    counter = Counter()
    for item in dataset:
        tokens = word_tokenize(item["text"].lower())
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_size - 2), start=2):
        vocab[word] = i
    return vocab


def load_glove_embeddings(vocab, embed_dim=100):
    print("Loading GloVe via gensim...")
    glove = api.load(f"glove-wiki-gigaword-{embed_dim}")

    embeddings = torch.randn(len(vocab), embed_dim)

    for word, idx in vocab.items():
        if word in glove:
            embeddings[idx] = torch.tensor(glove[word])

    return embeddings


# -----------------------------
# Models (with Dropout)
# -----------------------------
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)

        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        hidden = self.dropout(hidden.squeeze(0))
        return self.fc(hidden)


class BiRNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)

        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths):
        x = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.rnn(packed)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        hidden = self.dropout(hidden)
        return self.fc(hidden)


# -----------------------------
# Train & Eval
# -----------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0

    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x, lengths)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x, lengths)
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()

    return correct / len(loader.dataset)


# -----------------------------
# Prediction
# -----------------------------


def predict_text(model, text, vocab, device, max_len=50):
    model.eval()
    tokens = word_tokenize(text.lower())[:max_len]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    tensor = torch.tensor(ids).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        outputs = model(tensor, lengths)
        pred = outputs.argmax(1).item()

    return pred


# -----------------------------
# Main
# -----------------------------


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)

    # AUTO LOAD MODE
    if args.predict and os.path.exists(f"checkpoints/{args.model}.pt"):
        print("Loading existing model for inference...")

        with open("checkpoints/vocab.pkl", "rb") as f:
            vocab = pickle.load(f)

        if args.model == "rnn":
            model = RNNModel(len(vocab), args.embed_dim, args.hidden_dim, 5)
        else:
            model = BiRNNModel(len(vocab), args.embed_dim, args.hidden_dim, 5)

        model.load_state_dict(torch.load(f"checkpoints/{args.model}.pt", map_location=device))
        model.to(device)

        pred = predict_text(model, args.predict, vocab, device)
        labels = ["very negative", "negative", "neutral", "positive", "very positive"]

        print(f"\nText: {args.predict}")
        print(f"Prediction: {labels[pred]}")
        return

    # TRAIN MODE
    dataset = load_dataset("SetFit/sst5")
    train_data = dataset["train"]
    val_data = dataset["validation"]

    vocab = build_vocab(train_data, args.vocab_size)

    with open("checkpoints/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

    embeddings = load_glove_embeddings(vocab, args.embed_dim)

    train_ds = SSTDataset(train_data, vocab)
    val_ds = SSTDataset(val_data, vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    if args.model == "rnn":
        model = RNNModel(len(vocab), args.embed_dim, args.hidden_dim, 5, embeddings)
        save_path = "checkpoints/rnn.pt"
    else:
        model = BiRNNModel(len(vocab), args.embed_dim, args.hidden_dim, 5, embeddings)
        save_path = "checkpoints/birnn.pt"

    model.to(device)

    # Class weights (FIXES bias issue)
    labels = [item["label"] for item in train_data]
    counts = Counter(labels)
    weights = torch.tensor([1.0 / counts[i] for i in range(5)])
    weights = weights / weights.sum()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1} | Loss {train_loss:.3f} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["rnn", "birnn"], default="rnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--predict", type=str, default=None)

    args = parser.parse_args()
    main(args)
