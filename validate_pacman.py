"""Validation experiments for DQN on MsPacman-v5 (run after training).

  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python validate_pacman.py --run pacman
"""
import argparse
import glob
import os
import re

import ale_py
import gymnasium as gym
import numpy as np
import torch

from dqn_pacman import RESULTS, QNet, make_envs, preprocess_batch, preprocess_observation

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

gym.register_envs(ale_py)
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
results = []


def check(name, ok, ev):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {ev}", flush=True)


def load(run, ckpt):
    q = QNet().to(DEV)
    q.load_state_dict(torch.load(os.path.join(RESULTS, run, ckpt), map_location=DEV))
    return q


def pacman_xy(frame):
    """Pac-Man's centre in the raw 210x160 frame (pixels of colour (210,164,74) in the maze area)."""
    m = np.all(frame[1:172] == (210, 164, 74), axis=2)
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean() + 1) if len(xs) else None


@torch.no_grad()
def play(q, seed, eps=0.01, max_steps=5000):
    """One episode with full per-step records."""
    env = gym.make("ALE/MsPacman-v5")
    obs, info = env.reset(seed=seed)
    f = preprocess_observation(obs.copy())[..., 0]
    stack = np.repeat(f[None], 4, axis=0)
    lives, rec = info["lives"], []
    rng = np.random.default_rng(seed)
    for t in range(max_steps):
        qv = q(torch.from_numpy(stack[None]).to(DEV))[0].cpu().numpy()
        a = int(qv.argmax()) if rng.random() >= eps else int(rng.integers(9))
        xy = pacman_xy(obs)
        obs, r, te, tr, info = env.step(a)
        lost = info["lives"] < lives
        lives = info["lives"]
        rec.append(dict(q=qv, a=a, r=r, lost=lost or te, xy=xy, te=te))
        f = preprocess_observation(obs.copy())[..., 0]
        stack = np.concatenate([stack[1:], f[None]])
        if te or tr:
            break
    return rec


def test_preprocessing():
    """M1: the batched preprocessing equals the provided function; the uint64 wrap-around of
    `img // 3 - 128` is harmless because the int8 cast keeps the low byte (value mod 256)."""
    env = gym.make("ALE/MsPacman-v5")
    obs, _ = env.reset(seed=0)
    F = []
    for _ in range(300):
        obs, *_ = env.step(env.action_space.sample()); F.append(obs.copy())
    F = np.stack(F)
    ref = np.stack([preprocess_observation(x.copy())[..., 0] for x in F])
    pac = np.mean([(ref[i] == -128).sum() for i in range(len(ref))])
    check("M1 batched preprocessing == provided function", np.array_equal(ref, preprocess_batch(F)),
          f"300 frames identical; value range [{ref.min()}, {ref.max()}]; pixels at -128 (Pac-Man after the contrast step, plus black areas): {int(pac)} per frame")


def test_snapshots(run, n=20):
    """M2: held-out evaluation of training snapshots (eps=0.01, fresh seeds) must improve over training,
    starting from the random-policy level."""
    snaps = sorted(glob.glob(os.path.join(RESULTS, run, "q_ep*.pt")), key=lambda f: int(re.search(r"ep(\d+)", f).group(1)))
    snaps.append(os.path.join(RESULTS, run, "q_final.pt"))
    envs = make_envs(n)
    xs, ys, es = [], [], []
    for path in snaps:
        q = load(run, os.path.basename(path))
        obs, _ = envs.reset(seed=4242)
        stack = np.repeat(preprocess_batch(obs)[:, None], 4, axis=1)
        ret, done_mask, scores = np.zeros(n), np.zeros(n, bool), np.zeros(n)
        rng = np.random.default_rng(0)
        while not done_mask.all():
            with torch.no_grad():
                act = q(torch.from_numpy(stack).to(DEV)).argmax(1).cpu().numpy()
            rnd = rng.random(n) < 0.01
            act[rnd] = rng.integers(0, 9, rnd.sum())
            obs, r, te, tr, _ = envs.step(act)
            ret += r * ~done_mask
            fin = (te | tr) & ~done_mask
            scores[fin] = ret[fin]; done_mask |= fin
            stack = np.concatenate([stack[:, 1:], preprocess_batch(obs)[:, None]], axis=1)
        m = re.search(r"ep(\d+)", path)
        xs.append(int(m.group(1)) if m else -1); ys.append(scores.mean()); es.append(scores.std() / np.sqrt(n))
    envs.close()
    xs[-1] = max(xs[:-1]) + 500 if len(xs) > 1 else 0
    rho = np.corrcoef(xs, ys)[0, 1]
    check("M2 snapshot evaluation improves with training", ys[-1] > ys[0] * 1.5 and rho > 0.5,
          ", ".join(f"ep{x}: {y:.0f}±{e:.0f}" for x, y, e in zip(xs, ys, es)) + f" (last = final); corr = {rho:.2f}")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3)
    ax.set_xlabel("Training episode of snapshot"); ax.set_ylabel(f"Mean score ({n} held-out episodes)")
    ax.set_title("MsPacman: held-out evaluation of training snapshots"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, run, "val_snapshots.png"), dpi=150); plt.close(fig)


def test_q_calibration(run, eps_runs=6, gamma=0.99):
    """M3: Q(s,a) estimates the discounted *clipped* return with life loss as terminal. Compute that
    Monte-Carlo return along real trajectories and compare. Double DQN should be roughly calibrated
    (vanilla DQN is known to over-estimate)."""
    q = load(run, "q_final.pt")
    qa, G = [], []
    recs = []
    for k in range(eps_runs):
        rec = play(q, seed=777 + k)
        recs.append(rec)
        g = 0.0
        out = []
        for x in reversed(rec):
            if x["lost"]:
                g = 0.0
            g = np.clip(x["r"], -1, 1) + gamma * g
            out.append(g)
        G += out[::-1]; qa += [x["q"][x["a"]] for x in rec]
    qa, G = np.array(qa), np.array(G)
    ratio = qa.mean() / G.mean()
    corr = np.corrcoef(qa, G)[0, 1]
    check("M3 Q-values are calibrated against Monte-Carlo returns", 0.6 < ratio < 1.6 and corr > 0.3,
          f"{len(qa)} steps: mean Q(s,a) = {qa.mean():.2f}, mean MC return = {G.mean():.2f} (ratio {ratio:.2f}); "
          f"corr = {corr:.2f}; theoretical cap 1/(1-γ) = {1/(1-gamma):.0f}")
    return recs, qa, G


def plot_life_loss(run, recs):
    """M4 + non-standard plot: Q along an episode with life losses marked. Since life loss is terminal
    for the target, max Q should fall in the steps right before Pac-Man dies."""
    pre, base = [], []
    for rec in recs:
        mq = np.array([x["q"].max() for x in rec])
        deaths = [t for t, x in enumerate(rec) if x["lost"]]
        for t in deaths:
            if t >= 40:
                pre.append(mq[t - 5:t].mean()); base.append(mq[t - 40:t - 20].mean())
    check("M4 max Q drops just before a life is lost", np.mean(pre) < np.mean(base),
          f"{len(pre)} deaths: mean max Q 5 steps before death = {np.mean(pre):.2f} vs 20-40 steps before = {np.mean(base):.2f}")
    rec = recs[0]
    mq = np.array([x["q"].max() for x in rec])
    score = np.cumsum([x["r"] for x in rec])
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(mq, lw=1, color="#2b6cb0", label="max_a Q(s_t, a)")
    for t, x in enumerate(rec):
        if x["lost"]:
            ax.axvline(t, color="#c53030", ls="--", lw=1)
        if x["r"] >= 200:
            ax.scatter(t, mq[t], color="#d69e2e", zorder=3, s=25)
    ax.set_xlabel("Step in episode"); ax.set_ylabel("max Q")
    ax2 = ax.twinx(); ax2.plot(score, color="#718096", lw=1, alpha=0.6); ax2.set_ylabel("Cumulative score", color="#718096")
    ax.set_title("MsPacman: Q-value timeline (red dashed = life lost, gold dots = ghost eaten)")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, run, "nonstandard_q_timeline.png"), dpi=150); plt.close(fig)


def plot_q_map(run, recs):
    """Non-standard plot: where in the maze does the agent think it is doing well? Mean max-Q binned by
    Pac-Man's position, drawn over a maze frame."""
    env = gym.make("ALE/MsPacman-v5"); bg, _ = env.reset(seed=0)
    H, W, B = 172, 160, 8
    S, C = np.zeros((H // B + 1, W // B + 1)), np.zeros((H // B + 1, W // B + 1))
    for rec in recs:
        for x in rec:
            if x["xy"] is not None:
                i, j = int(x["xy"][1] // B), int(x["xy"][0] // B)
                S[i, j] += x["q"].max(); C[i, j] += 1
    M = np.where(C > 0, S / np.maximum(C, 1), np.nan)
    fig, ax = plt.subplots(figsize=(5.5, 6))
    ax.imshow(bg[:H], alpha=0.45)
    im = ax.imshow(M, extent=(0, (W // B + 1) * B, (H // B + 1) * B, 0), cmap="viridis", alpha=0.75)
    plt.colorbar(im, ax=ax, label="mean max Q at this position")
    ax.set_title("MsPacman: value map by Pac-Man position"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS, run, "nonstandard_q_map.png"), dpi=150); plt.close(fig)
    print(f"       Q map: {int((C > 0).sum())} visited cells")


def first_steps(run):
    """M5: first 10 steps under the trained net: Q-values, action, reward."""
    q = load(run, "q_final.pt")
    rec = play(q, seed=1, eps=0.0, max_steps=10)
    names = gym.make("ALE/MsPacman-v5").unwrapped.get_action_meanings()
    print("       first 10 steps (greedy):")
    for t, x in enumerate(rec):
        print(f"       t{t}: pacman={None if x['xy'] is None else tuple(round(v) for v in x['xy'])} "
              f"Q=[{' '.join(f'{v:.2f}' for v in x['q'])}] a={names[x['a']]} r={x['r']:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", default="pacman")
    a = ap.parse_args()
    test_preprocessing()
    first_steps(a.run)
    recs, qa, G = test_q_calibration(a.run)
    plot_life_loss(a.run, recs)
    plot_q_map(a.run, recs)
    test_snapshots(a.run)
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed")
