"""Required figures: (i) max Q vs episode, (ii) reward + 100-episode moving average, (iii) 500-roll-out histogram.

  python plots.py cartpole --seed 1
  python plots.py pacman
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "results", "figures")
os.makedirs(FIG, exist_ok=True)


def ma(x, w=100):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0))
    n = np.arange(1, len(x) + 1)
    return (c[n] - c[np.maximum(n - w, 0)]) / np.minimum(n, w)


def three_panel(df, scores, game, fname, q_ref=None, bins=None):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.3))
    axs[0].plot(df.episode, df.mean_maxQ, lw=0.7, color="#3182ce", label="mean over episode of max_a Q(s_t,a)")
    axs[0].plot(df.episode, df.start_maxQ, lw=0.7, color="#dd6b20", alpha=0.7, label="max_a Q(s_0,a) at episode start")
    if q_ref:
        axs[0].axhline(q_ref, color="k", ls="--", lw=1, label=f"1/(1-γ) = {q_ref:.0f}")
    axs[0].set_xlabel("Training episode"); axs[0].set_ylabel("Max Q-value"); axs[0].legend(frameon=False, fontsize=8)
    axs[0].set_title(f"{game}: (i) maximum Q-value"); axs[0].grid(alpha=0.3)
    axs[1].plot(df.episode, df.score, lw=0.5, color="#90cdf4", label="Episode reward")
    axs[1].plot(df.episode, ma(df.score), lw=2, color="#2b6cb0", label="Moving average (last 100)")
    axs[1].set_xlabel("Training episode"); axs[1].set_ylabel("Episode reward"); axs[1].legend(frameon=False)
    axs[1].set_title(f"{game}: (ii) episode reward"); axs[1].grid(alpha=0.3)
    axs[2].hist(scores, bins=bins if bins is not None else 30, color="#48bb78", edgecolor="white")
    axs[2].axvline(scores.mean(), color="#2f855a", ls="--")
    axs[2].set_title(f"{game}: (iii) 500 roll-outs of trained net\nmean = {scores.mean():.1f}, std = {scores.std():.1f}")
    axs[2].set_xlabel("Episode reward"); axs[2].set_ylabel("Count"); axs[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, fname), dpi=150); plt.close(fig)


def cartpole(seed):
    d = os.path.join(HERE, "results", "cartpole", "cartpole", f"seed{seed}")
    df = pd.read_csv(os.path.join(d, "episodes.csv"))
    three_panel(df, np.load(os.path.join(d, "eval_q_best.npy")), "CartPole-v1", f"cartpole_seed{seed}.png",
                q_ref=20, bins=np.arange(0, 525, 25))
    # all seeds
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for f in sorted(glob.glob(os.path.join(HERE, "results", "cartpole", "cartpole", "seed*", "episodes.csv"))):
        s = f.split(os.sep)[-2]; x = pd.read_csv(f)
        axs[0].plot(x.episode, ma(x.score), label=s); axs[1].plot(x.episode, x.mean_maxQ, lw=0.6, label=s)
    axs[0].set_ylabel("Reward (MA 100)"); axs[1].set_ylabel("Mean max Q"); axs[1].axhline(20, color="k", ls="--")
    for ax in axs:
        ax.set_xlabel("Training episode"); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8)
    fig.suptitle("CartPole-v1 DQN: 5 seeds"); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "cartpole_all_seeds.png"), dpi=150); plt.close(fig)


def pacman(run):
    d = os.path.join(HERE, "results", run)
    df = pd.read_csv(os.path.join(d, "episodes.csv"))
    three_panel(df, np.load(os.path.join(d, "eval_q_final.npy")), "MsPacman-v5", f"{run}.png")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(df.episode, df.epsilon, label="ε"); ax2 = ax.twinx()
    ax2.plot(df.episode, df.lr, color="#dd6b20", label="learning rate")
    ax.set_xlabel("Training episode"); ax.set_ylabel("ε"); ax2.set_ylabel("learning rate", color="#dd6b20")
    ax.set_title("MsPacman: ε and learning-rate schedules"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"{run}_schedules.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["cartpole", "pacman"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--run", default="pacman")
    a = p.parse_args()
    cartpole(a.seed) if a.what == "cartpole" else pacman(a.run)
    print("figures in", FIG)
