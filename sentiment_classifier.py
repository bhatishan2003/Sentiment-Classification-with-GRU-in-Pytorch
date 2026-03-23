import argparse
import os
import pickle
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from nltk.tokenize import word_tokenize
import nltk
import gensim.downloader as api
from sklearn.metrics import f1_score, accuracy_score
import mlflow
import mlflow.pytorch

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
warnings.filterwarnings("ignore")

LABEL_NAMES = ["very negative", "negative", "neutral", "positive", "very positive"]
EMBED_STRATEGIES = ["random", "glove_frozen", "glove_finetuned"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class SSTDataset(Dataset):
    def __init__(self, data, vocab, max_len=50):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        tokens = word_tokenize(text.lower())[: self.max_len]
        ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return self.encode(item["text"]), item["label"]


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = torch.tensor([max(len(t), 1) for t in texts])
    texts = nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
    return texts, lengths, torch.tensor(labels, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────────────────────────────────────


def build_vocab(dataset, max_size=10000):
    counter = Counter()
    for item in dataset:
        counter.update(word_tokenize(item["text"].lower()))
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_size - 2), start=2):
        vocab[word] = i
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────


def make_random_embeddings(vocab_size, embed_dim):
    """Randomly initialised — learned entirely during training."""
    emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    nn.init.normal_(emb.weight, mean=0, std=0.01)
    emb.weight.data[0].zero_()
    return emb


def load_glove_weight(vocab, embed_dim=100, oov_strategy="random"):
    """
    Builds a weight matrix from GloVe vectors.

    oov_strategy:
        'random' - OOV tokens get small random vectors  (default)
        'zero'   - OOV tokens are left as zeros
    """
    print(f"  Loading GloVe-{embed_dim}d ...")
    glove = api.load(f"glove-wiki-gigaword-{embed_dim}")

    weight = torch.randn(len(vocab), embed_dim) * 0.01 if oov_strategy == "random" else torch.zeros(len(vocab), embed_dim)

    hit = 0
    for word, idx in vocab.items():
        if word in glove:
            weight[idx] = torch.tensor(glove[word])
            hit += 1

    weight[0].zero_()
    total = len(vocab)
    print(f"  GloVe coverage: {hit}/{total} ({hit / total:.1%})")
    return weight


def make_glove_embedding_layer(vocab, embed_dim, freeze, oov_strategy="random"):
    weight = load_glove_weight(vocab, embed_dim, oov_strategy)
    return nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=0)


def build_embedding_layer(embed_strategy, vocab, embed_dim, oov_strategy="random"):
    """Factory: returns the correct nn.Embedding for the given strategy."""
    if embed_strategy == "random":
        return make_random_embeddings(len(vocab), embed_dim)
    elif embed_strategy == "glove_frozen":
        return make_glove_embedding_layer(vocab, embed_dim, freeze=True, oov_strategy=oov_strategy)
    elif embed_strategy == "glove_finetuned":
        return make_glove_embedding_layer(vocab, embed_dim, freeze=False, oov_strategy=oov_strategy)
    else:
        raise ValueError(f"Unknown embed_strategy: {embed_strategy}")


# ─────────────────────────────────────────────────────────────────────────────
# Model  (single class handles both GRU and BiGRU)
# ─────────────────────────────────────────────────────────────────────────────


class GRUModel(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, num_classes, bidirectional=False, dropout=0.5, embed_dropout=0.0):
        super().__init__()
        self.embedding = embedding_layer
        embed_dim = embedding_layer.embedding_dim

        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)

        self.embed_dropout = nn.Dropout(embed_dropout)
        self.dropout = nn.Dropout(dropout)

        factor = 2 if bidirectional else 1

        # 🔥 FIXED
        self.fc = nn.Linear(hidden_dim * factor * 2, num_classes)

        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        x = self.embed_dropout(self.embedding(x))

        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_outputs, _ = self.gru(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        mean_pool = torch.mean(outputs, dim=1)
        max_pool, _ = torch.max(outputs, dim=1)
        out = torch.cat([mean_pool, max_pool], dim=1)

        return self.fc(self.dropout(out))  # ─────────────────────────────────────────────────────────────────────────────


# Train / Evaluate
# ─────────────────────────────────────────────────────────────────────────────


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []
    for x, lengths, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, lengths)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.argmax(1).cpu().tolist())
        all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x, lengths)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, f1


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────


def predict_text(model, vocab, text, device, max_len=50):
    """Classify one sentence. Returns (label_str, probs_list)."""
    model.eval()
    tokens = word_tokenize(text.lower())[:max_len]
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens]
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([max(len(ids), 1)])

    with torch.no_grad():
        out = model(x, lengths)
        probs = torch.softmax(out, dim=1).squeeze().cpu().tolist()
        pred = int(np.argmax(probs))

    return LABEL_NAMES[pred], probs


def print_prediction(text, label, probs):
    """Pretty-print prediction with inline probability bar."""
    print(f"\n{'-' * 56}")
    print(f"  Text      : {text}")
    print(f"  Predicted : {label.upper()}")
    print(f"{'-' * 56}")
    print("  Class probabilities:")
    for name, p in zip(LABEL_NAMES, probs):
        bar = "#" * int(p * 35)
        print(f"    {name:<15}  {p:.3f}  {bar}")
    print()


def checkpoint_path(model_name, embed_strategy, embed_dim, plots_dir="plots"):
    """Canonical path where a trained checkpoint is saved."""
    return os.path.join(plots_dir, f"{model_name}_{embed_strategy}_dim{embed_dim}_model.pt")


# ─────────────────────────────────────────────────────────────────────────────
# Cosine-similarity analysis
# ─────────────────────────────────────────────────────────────────────────────

WORD_GROUPS = {
    "positive": ["good", "great", "excellent", "wonderful", "amazing"],
    "negative": ["bad", "terrible", "awful", "horrible", "dreadful"],
    "neutral": ["the", "a", "is", "in", "of"],
}


def cosine_sim_matrix(vectors):
    vecs = torch.stack(vectors)
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    n = vecs / norms
    return (n @ n.T).numpy()


def embedding_vectors(emb_layer, vocab, words):
    return [emb_layer.weight.data[vocab.get(w, vocab["<unk>"])].detach().cpu() for w in words]


def plot_cosine_heatmap(sim, words, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        sim,
        xticklabels=words,
        yticklabels=words,
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        annot=True,
        fmt=".2f",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def run_cosine_analysis(emb_layer, vocab, tag, plots_dir):
    paths = []
    for group, words in WORD_GROUPS.items():
        present = [w for w in words if w in vocab]
        if len(present) < 2:
            continue
        vecs = embedding_vectors(emb_layer, vocab, present)
        sim = cosine_sim_matrix(vecs)
        path = os.path.join(plots_dir, f"cosine_{tag}_{group}.png")
        plot_cosine_heatmap(sim, present, f"Cosine similarity - {group}\n({tag})", path)
        paths.append(path)
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_training_curves(history, title, path):
    """
    Plots loss, accuracy, and F1 curves for train / val / test per epoch.
    """
    epochs = range(1, len(history["train_acc"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ── Loss ──────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # ── Accuracy ──────────────────────────────────────────────────
    axes[1].plot(epochs, history["train_acc"], label="train acc")
    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].plot(epochs, history["test_acc"], label="test acc", linestyle=":")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    # ── F1 ────────────────────────────────────────────────────────
    axes[2].plot(epochs, history["train_f1"], label="train F1")
    axes[2].plot(epochs, history["val_f1"], label="val F1")
    axes[2].plot(epochs, history["test_f1"], label="test F1", linestyle=":")
    axes[2].set_title("Macro F1")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_comparison_bar(results, metric, title, path):
    names = list(results.keys())
    vals = [results[n][metric] for n in names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 5))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def make_summary_plots(all_results, model_name, plots_dir):
    """Bar charts comparing all embedding strategies for one architecture."""
    prefix = model_name.upper()
    paths = [
        plot_comparison_bar(
            all_results,
            "test_acc",
            f"{prefix} - test accuracy by embed strategy",
            os.path.join(plots_dir, f"{model_name}_summary_test_acc.png"),
        ),
        plot_comparison_bar(
            all_results,
            "test_f1",
            f"{prefix} - test F1 by embed strategy",
            os.path.join(plots_dir, f"{model_name}_summary_test_f1.png"),
        ),
    ]
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Single experiment runner
# ─────────────────────────────────────────────────────────────────────────────


def run_experiment(
    run_name,
    model_name,
    embed_strategy,
    embed_dim,
    hidden_dim,
    vocab,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    lr,
    batch_size,
    plots_dir,
    oov_strategy="random",
    dropout=0.5,
    patience=3,
):
    print(f"\n{'=' * 60}\n  RUN: {run_name}\n{'=' * 60}")
    print(f"  dropout={dropout}  lr={lr}  weight_decay=1e-3  patience={patience}")

    # ── Embedding layer ────────────────────────────────────────────
    emb_layer = build_embedding_layer(embed_strategy, vocab, embed_dim, oov_strategy)

    # Snapshot BEFORE training for cosine analysis
    initial_emb = nn.Embedding(len(vocab), embed_dim, padding_idx=0)
    initial_emb.weight = nn.Parameter(emb_layer.weight.data.clone())

    # ── Model ──────────────────────────────────────────────────────
    bidir = model_name == "bigru"
    # Disable embedding dropout for frozen embeddings — no benefit, hurts signal
    emb_drop = dropout if embed_strategy != "glove_frozen" else 0.0
    model = GRUModel(emb_layer, hidden_dim, num_classes=5, bidirectional=bidir, dropout=dropout, embed_dropout=emb_drop).to(
        device
    )

    # ── Class-balanced loss — computed from TRAIN set ──────────────
    label_counts = Counter(item["label"] for item in train_loader.dataset.data)
    weights = torch.tensor([1.0 / max(label_counts.get(i, 1), 1) for i in range(5)])
    weights = (weights / weights.sum()).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # history now tracks test metrics per epoch as well
    history = {k: [] for k in ("train_loss", "train_acc", "train_f1", "val_acc", "val_f1", "test_acc", "test_f1")}
    best_val_f1, best_state = 0.0, None
    patience_counter = 0  # early stopping counter

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        va_acc, va_f1 = evaluate(model, val_loader, device)
        te_acc, te_f1 = evaluate(model, test_loader, device)  # test every epoch

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)
        history["test_acc"].append(te_acc)
        history["test_f1"].append(te_f1)

        print(
            f"  Ep {epoch:02d} | loss {tr_loss:.3f} | "
            f"tr-acc {tr_acc:.3f} tr-F1 {tr_f1:.3f} | "
            f"val-acc {va_acc:.3f} val-F1 {va_f1:.3f} | "
            f"test-acc {te_acc:.3f} test-F1 {te_f1:.3f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (no val-F1 improvement for {patience} epochs)")
                break

    # ── Restore best weights & final test eval ─────────────────────
    if best_state:
        model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"  BEST TEST  ->  acc {test_acc:.4f}   F1 {test_f1:.4f}")

    # ── Save checkpoint ────────────────────────────────────────────
    ckpt = checkpoint_path(model_name, embed_strategy, embed_dim, plots_dir)
    torch.save(model.state_dict(), ckpt)
    print(f"  Saved -> {ckpt}")

    # ── Plots ──────────────────────────────────────────────────────
    curve_path = os.path.join(plots_dir, f"{run_name}_curves.png")
    plot_training_curves(history, run_name, curve_path)

    cosine_init = run_cosine_analysis(initial_emb, vocab, f"{run_name}_initial", plots_dir)
    cosine_trained = run_cosine_analysis(model.embedding, vocab, f"{run_name}_trained", plots_dir)

    # ── MLflow ─────────────────────────────────────────────────────
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": model_name,
                "embed_strategy": embed_strategy,
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "oov_strategy": oov_strategy,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "vocab_size": len(vocab),
                "dropout": dropout,
                "patience": patience,
                "embed_dropout": emb_drop,
            }
        )
        for ep, (tl, ta, tf, va, vf, tea, tef) in enumerate(
            zip(
                history["train_loss"],
                history["train_acc"],
                history["train_f1"],
                history["val_acc"],
                history["val_f1"],
                history["test_acc"],
                history["test_f1"],
            ),
            start=1,
        ):
            mlflow.log_metrics(
                {
                    "train_loss": tl,
                    "train_acc": ta,
                    "train_f1": tf,
                    "val_acc": va,
                    "val_f1": vf,
                    "test_acc": tea,
                    "test_f1": tef,
                },
                step=ep,
            )
        # best test metrics (from best val-F1 checkpoint)
        mlflow.log_metrics({"best_test_acc": test_acc, "best_test_f1": test_f1})
        mlflow.log_artifact(curve_path, "plots/curves")
        for p in cosine_init + cosine_trained:
            mlflow.log_artifact(p, "plots/cosine")
        mlflow.log_artifact(ckpt, "model")

    return {"test_acc": test_acc, "test_f1": test_f1, "history": history}


# ─────────────────────────────────────────────────────────────────────────────
# Experiment grid for ONE architecture
# ─────────────────────────────────────────────────────────────────────────────


def run_model_experiments(args, vocab, train_loader, val_loader, test_loader, device, plots_dir):
    """
    Runs all embedding strategies for args.model (gru or bigru).
    Returns a dict  run_name -> {test_acc, test_f1, history}.
    """
    all_results = {}

    # Core 3: random, glove_frozen, glove_finetuned
    for embed_strategy in EMBED_STRATEGIES:
        run_name = f"{args.model}_{embed_strategy}_dim{args.embed_dim}"
        all_results[run_name] = run_experiment(
            run_name=run_name,
            model_name=args.model,
            embed_strategy=embed_strategy,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            vocab=vocab,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            plots_dir=plots_dir,
            dropout=args.dropout,
            patience=args.patience,
        )

    # Ablation 1: embedding dimensionality
    if args.ablation_dims:
        for dim in [50, 100, 200]:
            if dim == args.embed_dim:
                continue
            for strategy in ("random", "glove_finetuned"):
                if strategy == "glove_finetuned" and dim == 50:
                    continue  # gensim has no GloVe-50
                run_name = f"{args.model}_{strategy}_dim{dim}"
                all_results[run_name] = run_experiment(
                    run_name=run_name,
                    model_name=args.model,
                    embed_strategy=strategy,
                    embed_dim=dim,
                    hidden_dim=args.hidden_dim,
                    vocab=vocab,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    plots_dir=plots_dir,
                    dropout=args.dropout,
                    patience=args.patience,
                )

    # Ablation 2: OOV handling strategy
    if args.ablation_oov:
        for oov in ("random", "zero"):
            run_name = f"{args.model}_glove_finetuned_dim{args.embed_dim}_oov_{oov}"
            all_results[run_name] = run_experiment(
                run_name=run_name,
                model_name=args.model,
                embed_strategy="glove_finetuned",
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                vocab=vocab,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                plots_dir=plots_dir,
                oov_strategy=oov,
                dropout=args.dropout,
                patience=args.patience,
            )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN mode
# ─────────────────────────────────────────────────────────────────────────────


def mode_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    print(f"Device : {device}")
    print(f"Model  : {args.model.upper()}")
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(args.experiment_name)

    # ── Data ──────────────────────────────────────────────────────
    print("\nLoading SST-5 ...")
    dataset = load_dataset("SetFit/sst5")
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # Build / reuse vocab (shared across both models so results are comparable)
    vocab_file = "checkpoints/vocab.pkl"
    if os.path.exists(vocab_file):
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        print(f"Vocab loaded from cache: {len(vocab)} tokens")
    else:
        vocab = build_vocab(train_data, args.vocab_size)
        with open(vocab_file, "wb") as f:
            pickle.dump(vocab, f)
        print(f"Vocab built: {len(vocab)} tokens")

    train_ds = SSTDataset(train_data, vocab)
    val_ds = SSTDataset(val_data, vocab)
    test_ds = SSTDataset(test_data, vocab)

    kw = dict(batch_size=args.batch_size, collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    # ── Run experiments ────────────────────────────────────────────
    all_results = run_model_experiments(args, vocab, train_loader, val_loader, test_loader, device, plots_dir)

    # ── Summary plots ──────────────────────────────────────────────
    summary_paths = make_summary_plots(all_results, args.model, plots_dir)

    with mlflow.start_run(run_name=f"{args.model}_summary"):
        for p in summary_paths:
            mlflow.log_artifact(p, "plots/summary")

        lines = [
            f"\n{'=' * 60}",
            f"  {args.model.upper()} RESULTS SUMMARY",
            f"{'=' * 60}",
            f"  {'Run':<48} {'TestAcc':>8} {'TestF1':>8}",
            f"  {'-' * 66}",
        ]
        for name, res in sorted(all_results.items()):
            lines.append(f"  {name:<48} {res['test_acc']:>8.4f} {res['test_f1']:>8.4f}")
        table = "\n".join(lines)
        print(table)

        txt_path = os.path.join(plots_dir, f"{args.model}_results_table.txt")
        Path(txt_path).write_text(table, encoding="utf-8")
        mlflow.log_artifact(txt_path, "summary")

    print(f"\nTraining complete for {args.model.upper()}.")
    print("Run `mlflow ui` to explore all results in the browser.")
    print("\nCheckpoints saved in plots/:")
    for s in EMBED_STRATEGIES:
        p = checkpoint_path(args.model, s, args.embed_dim, plots_dir)
        exists = "OK" if os.path.exists(p) else "missing"
        print(f"  [{exists}]  {p}")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT mode
# ─────────────────────────────────────────────────────────────────────────────


def mode_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load vocab ─────────────────────────────────────────────────
    vocab_file = "checkpoints/vocab.pkl"
    if not os.path.exists(vocab_file):
        raise FileNotFoundError(
            "checkpoints/vocab.pkl not found. Train a model first:\n  python train.py --model gru --epochs 5"
        )
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    print(f"Vocab loaded : {len(vocab)} tokens")

    # ── Resolve checkpoint ─────────────────────────────────────────
    ckpt = checkpoint_path(args.model, args.embed_strategy, args.embed_dim, plots_dir="plots")
    if not os.path.exists(ckpt):
        available = sorted(Path("plots").glob("*_model.pt")) if Path("plots").exists() else []
        hint = ("\nAvailable checkpoints:\n" + "\n".join(f"  {p}" for p in available)) if available else ""
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}{hint}\n\nTrain it first:\n  python train.py --model {args.model} --epochs 5"
        )

    # ── Build model shell & load weights ───────────────────────────
    emb = nn.Embedding(len(vocab), args.embed_dim, padding_idx=0)
    bidir = args.model == "bigru"
    model = GRUModel(emb, args.hidden_dim, num_classes=5, bidirectional=bidir)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)

    print(f"Model loaded : {ckpt}")
    print(f"Architecture : {args.model.upper()}  embed={args.embed_strategy}  dim={args.embed_dim}\n")

    # ── Predict ────────────────────────────────────────────────────
    label, probs = predict_text(model, vocab, args.predict, device)
    print_prediction(args.predict, label, probs)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="SST-5 sentiment classifier  -  GRU / BiGRU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Core flag (required) ───────────────────────────────────────
    parser.add_argument("--model", choices=["gru", "bigru"], required=True, help="Architecture to train or use for prediction")

    # ── Predict flag ───────────────────────────────────────────────
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="TEXT",
        help="Sentence to classify. When set, runs prediction instead of training.",
    )

    # ── Shared hyper-params ────────────────────────────────────────
    parser.add_argument("--embed_dim", type=int, default=100, help="Embedding size (100 or 200 for GloVe; default 100)")
    parser.add_argument("--hidden_dim", type=int, default=64)

    # ── Embedding strategy for --predict ──────────────────────────
    parser.add_argument(
        "--embed_strategy",
        choices=EMBED_STRATEGIES,
        default="glove_finetuned",
        help="Which trained checkpoint to load for prediction (default: glove_finetuned)",
    )

    # ── Train-only flags ───────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--experiment_name", type=str, default="sst5_gru_experiments")
    parser.add_argument("--ablation_dims", action="store_true", help="Ablation over embed_dim in {50, 100, 200}")
    parser.add_argument("--ablation_oov", action="store_true", help="Ablation over OOV strategy {random, zero}")
    parser.add_argument("--dropout", type=float, default=0.6, help="Dropout rate for embedding and FC layers (default: 0.6)")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience — stop if val F1 does not improve for N epochs (default: 3)",
    )

    args = parser.parse_args()

    if args.predict:
        mode_predict(args)
    else:
        mode_train(args)


if __name__ == "__main__":
    main()
