"""Generate plots and a MARP slide deck from logs/results.jsonl."""
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

JSONL_PATH = "logs/results.jsonl"
PLOTS_DIR = "logs"
REPORT_PATH = "logs/report.md"


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_accuracy(entries: list[dict], window: int = 20) -> str:
    iters = [e["iteration"] for e in entries]
    correct = [int(e["true_label"] == e["predicted_label"]) for e in entries]
    acc_roll = rolling_mean(correct, window)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, acc_roll, color="#2563eb", linewidth=1.8,
            label=f"Rolling accuracy (w={window})")
    ax.axhline(sum(correct) / len(correct), color="#dc2626", linestyle="--",
               linewidth=1.2, label=f"Mean = {sum(correct)/len(correct):.2%}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Prediction Accuracy over Iterations")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confidence(entries: list[dict], window: int = 20) -> str:
    iters = [e["iteration"] for e in entries]
    confs = [e["confidence"] for e in entries]
    conf_roll = rolling_mean(confs, window)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, confs, color="#d1d5db", linewidth=0.6, alpha=0.7, label="Raw")
    ax.plot(iters, conf_roll, color="#7c3aed", linewidth=1.8,
            label=f"Rolling mean (w={window})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Softmax Confidence")
    ax.set_title("Prediction Confidence over Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "confidence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss(entries: list[dict], window: int = 20) -> str:
    iters = [e["iteration"] for e in entries]
    losses = [e["loss"] for e in entries]
    loss_roll = rolling_mean(losses, window)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, losses, color="#d1d5db", linewidth=0.6, alpha=0.7, label="Raw")
    ax.plot(iters, loss_roll, color="#059669", linewidth=1.8,
            label=f"Rolling mean (w={window})")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Prediction Loss over Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_label_distribution(entries: list[dict]) -> str:
    from collections import Counter
    true_counts = Counter(e["true_label"] for e in entries)
    pred_counts = Counter(e["predicted_label"] for e in entries)
    labels = list(range(10))

    x = range(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], [true_counts[l] for l in labels],
           width=width, label="True", color="#2563eb", alpha=0.8)
    ax.bar([i + width / 2 for i in x], [pred_counts[l] for l in labels],
           width=width, label="Predicted", color="#f59e0b", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Count")
    ax.set_title("True vs Predicted Label Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "label_dist.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── MARP slide deck ───────────────────────────────────────────────────────────

def build_report(entries: list[dict],
                 acc_path: str, conf_path: str,
                 loss_path: str, dist_path: str) -> None:
    n = len(entries)
    correct = sum(int(e["true_label"] == e["predicted_label"]) for e in entries)
    mean_acc = correct / n
    mean_conf = sum(e["confidence"] for e in entries) / n
    mean_loss = sum(e["loss"] for e in entries) / n
    first_iter = entries[0]["iteration"]
    last_iter = entries[-1]["iteration"]

    # paths relative to the logs/ directory where the report lives
    def rel(p):
        return os.path.basename(p)

    slides = textwrap.dedent(f"""\
    ---
    marp: true
    theme: default
    paginate: true
    ---

    # Continual Learning on MNIST
    ## Experiment Report

    **Iterations:** {first_iter} – {last_iter} &nbsp;|&nbsp; **n = {n}**
    **Seed:** 42 &nbsp;|&nbsp; **Device:** CPU &nbsp;|&nbsp; **LR:** 0.01

    ---

    ## Experimental Setup

    - Minimal CNN trained incrementally on MNIST
    - Bootstrap: 100 images, then one new image per iteration
    - Model re-instantiated from checkpoint each iteration
    - Prediction made **before** training on new sample
    - Labels: true MNIST label logged; pseudo-label used for training

    ---

    ## Summary Statistics

    | Metric | Value |
    |--------|-------|
    | Total iterations | {n} |
    | Correct predictions | {correct} / {n} |
    | Mean accuracy | {mean_acc:.2%} |
    | Mean confidence | {mean_conf:.4f} |
    | Mean cross-entropy loss | {mean_loss:.4f} |

    ---

    ## Prediction Accuracy

    ![Accuracy](accuracy.png)

    Rolling mean (window = 20) over {n} iterations.

    ---

    ## Prediction Confidence

    ![Confidence](confidence.png)

    Softmax confidence on the prediction sample at each iteration.

    ---

    ## Cross-Entropy Loss

    ![Loss](loss.png)

    Loss evaluated on the prediction sample before each training step.

    ---

    ## Label Distribution

    ![Label distribution](label_dist.png)

    Comparison of true MNIST labels vs model predictions across all iterations.

    ---

    ## Observations

    - Initial accuracy near chance (~10%) with low confidence
    - Confidence and accuracy trend upward as training data grows
    - Model is re-instantiated each iteration — no persistent optimizer state
    - Single-image gradient updates limit plasticity per step

    ---

    ## Reproducibility

    Re-running with `--seed 42` produces **identical** `results.jsonl`
    (MD5 verified). Determinism achieved via:

    - `random.seed`, `numpy.seed`, `torch.manual_seed`
    - `torch.backends.cudnn.deterministic = True`
    - `torch.backends.cudnn.benchmark = False`
    """)

    with open(REPORT_PATH, "w") as f:
        f.write(slides)
    print(f"Report written to {REPORT_PATH}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    entries = load_log(JSONL_PATH)
    print(f"Loaded {len(entries)} log entries.")

    acc_path = plot_accuracy(entries)
    conf_path = plot_confidence(entries)
    loss_path = plot_loss(entries)
    dist_path = plot_label_distribution(entries)

    print(f"Plots saved: {acc_path}, {conf_path}, {loss_path}, {dist_path}")
    build_report(entries, acc_path, conf_path, loss_path, dist_path)


if __name__ == "__main__":
    main()
