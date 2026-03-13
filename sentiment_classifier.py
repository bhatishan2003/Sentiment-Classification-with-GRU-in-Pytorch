import argparse
import os
import random
import string
from collections import Counter
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from nltk.corpus import movie_reviews, stopwords
from nltk.tokenize import word_tokenize


# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("movie_reviews", quiet=True)

# ──────────────────────────────────────────────
# ARGPARSE
# ──────────────────────────────────────────────


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rnn", "birnn"], default="birnn")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--max_vocab", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--text", type=str, default=None)
    return parser.parse_args()


# ──────────────────────────────────────────────
# TEXT PREPROCESSING
# ──────────────────────────────────────────────

STOP_WORDS = set(stopwords.words("english"))


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


# ──────────────────────────────────────────────
# VOCABULARY
# ──────────────────────────────────────────────

PAD, UNK = "<PAD>", "<UNK>"


class Vocabulary:
    def __init__(self, max_size=10000):
        self.w2i = {PAD: 0, UNK: 1}
        self.i2w = {0: PAD, 1: UNK}
        self.max_size = max_size

    def build(self, token_lists):
        counter = Counter(t for tokens in token_lists for t in tokens)

        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.w2i)
            self.w2i[word] = idx
            self.i2w[idx] = word

        print(f"Vocabulary built: {len(self.w2i):,} tokens")
        return self

    def encode(self, tokens):
        return [self.w2i.get(t, 1) for t in tokens]

    def __len__(self):
        return len(self.w2i)


# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.samples = []

        for text, label in zip(texts, labels):
            ids = vocab.encode(preprocess(text))[:max_len]

            if len(ids) == 0:
                ids = [1]

            self.samples.append((torch.tensor(ids), torch.tensor(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    seqs, labels = zip(*batch)

    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)

    return padded, lengths, torch.stack(labels)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────


def load_data(seed=42):
    pos_ids = movie_reviews.fileids("pos")
    neg_ids = movie_reviews.fileids("neg")

    texts = [movie_reviews.raw(fid) for fid in pos_ids + neg_ids]
    labels = [1] * len(pos_ids) + [0] * len(neg_ids)

    combined = list(zip(texts, labels))
    random.seed(seed)
    random.shuffle(combined)

    texts, labels = zip(*combined)

    split = int(0.8 * len(texts))

    return (list(texts[:split]), list(labels[:split]), list(texts[split:]), list(labels[split:]))


# ──────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))
        _, hidden = self.rnn(emb)

        out = self.fc(self.dropout(hidden.squeeze(0)))

        return out


class BiRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))

        _, hidden = self.rnn(emb)

        out = torch.cat([hidden[0], hidden[1]], dim=-1)

        out = self.fc(self.dropout(out))

        return out


def build_model(args, vocab_size):
    if args.model == "rnn":
        return RNNClassifier(vocab_size, args.embed_dim, args.hidden_dim)

    return BiRNNClassifier(vocab_size, args.embed_dim, args.hidden_dim)


# ──────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss, correct, n = 0, 0, 0

    for padded, lengths, labels in tqdm(loader, desc="Training", leave=False):
        padded, labels = padded.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(padded, lengths)

        loss = criterion(logits, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        preds = logits.argmax(1)

        correct += (preds == labels).sum().item()
        n += labels.size(0)

        total_loss += loss.item() * labels.size(0)

    return total_loss / n, correct / n


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss, correct, n = 0, 0, 0

    with torch.no_grad():
        for padded, lengths, labels in tqdm(loader, desc="Evaluating", leave=False):
            padded, labels = padded.to(device), labels.to(device)

            logits = model(padded, lengths)

            loss = criterion(logits, labels)

            preds = logits.argmax(1)

            correct += (preds == labels).sum().item()
            n += labels.size(0)

            total_loss += loss.item() * labels.size(0)

    return total_loss / n, correct / n


# ──────────────────────────────────────────────
# PREDICTION
# ──────────────────────────────────────────────


def predict(text, model, vocab, device, max_len=200):
    model.eval()

    tokens = preprocess(text)[:max_len]

    ids = vocab.encode(tokens)

    x = torch.tensor([ids]).to(device)

    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        logits = model(x, lengths)

        probs = torch.softmax(logits, dim=-1)[0]

        pred = probs.argmax().item()

    label = "POSITIVE 👍" if pred == 1 else "NEGATIVE 👎"

    print(f"\nText      : {text}")
    print(f"Tokens    : {tokens}")
    print(f"Sentiment : {label}")
    print(f"Confidence: {probs[pred] * 100:.1f}%")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading NLTK movie_reviews corpus …")

    train_texts, train_labels, test_texts, test_labels = load_data(args.seed)

    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    print("Preprocessing & building vocabulary …")

    train_tokens = [preprocess(t) for t in train_texts]

    vocab = Vocabulary(args.max_vocab).build(train_tokens)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"{args.model}_checkpoint.pt")

    if args.predict:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        vocab = ckpt["vocab"]

        model = build_model(args, len(vocab)).to(device)

        model.load_state_dict(ckpt["model_state"])

        predict(args.text, model, vocab, device)

        return

    train_ds = SentimentDataset(train_texts, train_labels, vocab, args.max_len)
    test_ds = SentimentDataset(test_texts, test_labels, vocab, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = build_model(args, len(vocab)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()

    print(f"\nModel : {args.model.upper()}  |  Device: {device}")

    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)

        te_loss, te_acc = evaluate(model, test_loader, criterion, device)

        marker = ""

        if te_acc > best_acc:
            best_acc = te_acc

            torch.save({"model_state": model.state_dict(), "vocab": vocab}, checkpoint_path)

            marker = " ✓ saved"

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
            f"Test loss {te_loss:.3f} acc {te_acc:.3f}{marker}"
        )

    print(f"\nBest test accuracy: {best_acc * 100:.1f}%")
    print(f"Checkpoint saved to: {checkpoint_path}")

    demo = "The film was surprisingly good — great performances!"

    print("\n--- Quick demo ---")

    predict(demo, model, vocab, device)


if __name__ == "__main__":
    main()
