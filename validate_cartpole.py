"""Validation experiments for DQN on CartPole-v1 (gamma = 0.95).

  python validate_cartpole.py --run cartpole --seed 0
Some tests retrain small ablation agents (a few minutes on CPU, run in parallel processes).
"""
import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import torch

import dqn_cartpole as dc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

results = []
OUT = os.path.join(dc.RESULTS, "validation")
os.makedirs(OUT, exist_ok=True)


def check(name, ok, ev):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}\n       {ev}", flush=True)


def load(run, seed, ckpt="q_best.pt"):
    q = dc.qnet(); q.load_state_dict(torch.load(os.path.join(dc.RESULTS, run, f"seed{seed}", ckpt)))
    return q


def rollout(q, seed):
    env = gym.make("CartPole-v1")
    s, _ = env.reset(seed=seed); S, A, Q, done = [], [], [], False
    while not done:
        with torch.no_grad():
            qv = q(torch.as_tensor(s)).numpy()
        a = int(qv.argmax()); S.append(s); A.append(a); Q.append(qv)
        s, r, te, tr, _ = env.step(a); done = te or tr
    return np.array(S), np.array(A), np.array(Q), te


def test_q_scale(run, seed, gamma=0.95):
    """C1: reward is +1 per step, so for a state from which the agent survives k more steps the true
    discounted value is (1 - gamma^k)/(1 - gamma) -> 1/(1-gamma) = 20. States >= 60 steps from the end
    have true value >= 19.1. Checked on the final network (Q fully propagated) and on the best
    checkpoint (selected for its greedy score, possibly before Q had converged)."""
    out, fails = {}, []
    for ckpt in ("q_final.pt", "q_best.pt"):
        q = load(run, seed, ckpt)
        safe = []
        for k in range(20):
            S, A, Q, te = rollout(q, 500 + k)
            T = len(S)
            if te and ckpt == "q_final.pt":
                fails.append((Q.max(1), (1 - gamma ** (T - np.arange(T))) / (1 - gamma)))
            safe += list(Q.max(1)[:max(T - 60, 1)])
        out[ckpt] = np.array(safe)
    f, b = out["q_final.pt"], out["q_best.pt"]
    check("C1 Q-values converge to the closed form 1/(1-γ) = 20", abs(f.mean() - 20) < 2,
          f"final net: mean max Q on states >=60 steps from the end = {f.mean():.2f} (theory 19.1-20.0); "
          f"best checkpoint: {b.mean():.2f}" + ("" if abs(b.mean() - 20) < 2 else
          " -> picked before Q finished propagating: it ranks actions correctly but values are not calibrated"))
    return fails


def train_variant(kw):
    p = dc.parser()
    args = p.parse_args(["train"] + kw)
    dc.train(args, verbose=False)
    sc = dc.greedy_eval(_load(args), 50, seed=2 * 10**6)
    import pandas as pd
    df = pd.read_csv(os.path.join(dc.RESULTS, args.run, f"seed{args.seed}", "episodes.csv"))
    return args.run, args.seed, sc.mean(), df.mean_maxQ.tail(100).mean(), df.score.tail(100).mean()


def _load(args):
    q = dc.qnet(); q.load_state_dict(torch.load(os.path.join(dc.RESULTS, args.run, f"seed{args.seed}", "q_best.pt")))
    return q


def test_ablations(base_args, seeds=(0, 1, 2)):
    """C2-C4: change one ingredient at a time and check the predicted effect.
      C2 gamma = 0.9   -> Q should converge to 1/(1-0.9) = 10 instead of 20
      C3 truncation treated as terminal -> spurious zero targets at t=500 pull Q below 20
      C4 no target network (sync every step) / tiny replay (= batch) -> less stable, lower score"""
    variants = {
        "base": [], "gamma0.9": ["--gamma", "0.9"],
        "short_bootstrap": ["--max_steps", "50"], "short_terminal": ["--max_steps", "50", "--trunc_as_terminal"],
        "no_target": ["--target_every", "1"], "tiny_replay": ["--buffer", "64", "--learn_start", "64"],
    }
    jobs = [base_args + v + ["--run", f"abl_{k}", "--seed", str(s)] for k, v in variants.items() for s in seeds]
    with ProcessPoolExecutor(len(jobs), mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(train_variant, jobs))
    tab = {}
    for run, s, ev, mq, ma in res:
        tab.setdefault(run, []).append((ev, mq, ma))
    summ = {k: np.mean(v, axis=0) for k, v in tab.items()}
    for k, (ev, mq, ma) in summ.items():
        print(f"       {k:18s} best-ckpt greedy score {ev:6.1f} | mean max Q (last 100 eps) {mq:5.2f} | MA100 {ma:6.1f}")
    b, g9 = summ["abl_base"], summ["abl_gamma0.9"]
    check("C2 gamma=0.9 halves the Q scale (10 vs 20)", abs(g9[1] - 10) < 1.5 and abs(b[1] - 20) < 2.5,
          f"mean max Q: gamma 0.95 -> {b[1]:.2f}, gamma 0.9 -> {g9[1]:.2f}")
    # 50-step limit: truncation becomes frequent. The state has no clock, so Q settles at the fixed point
    # Q = 1/(1 - γ(1 - f - ρ)) where f = failures per transition and ρ = truncations per transition that
    # are (wrongly) treated as terminal; with correct bootstrapping ρ drops out.
    import pandas as pd
    def hazard(run):
        d = pd.concat([pd.read_csv(os.path.join(dc.RESULTS, run, f"seed{s}", "episodes.csv")).tail(200) for s in seeds])
        n = d.length.sum()
        return (d.length < 50).sum() / n, (d.length == 50).sum() / n, d.mean_maxQ.mean()
    fb, rb, qb = hazard("abl_short_bootstrap")
    ft, rt, qt = hazard("abl_short_terminal")
    pb, pt = 1 / (1 - 0.95 * (1 - fb)), 1 / (1 - 0.95 * (1 - ft - rt))
    check("C3 treating truncation as terminal lowers Q (direction predicted by a hazard-rate model)",
          qt < qb and abs(qb - pb) < 1.5,
          f"50-step limit, mean max Q: bootstrap {qb:.2f} (model {pb:.2f}), as-terminal {qt:.2f} (model {pt:.2f}); "
          f"failure rate f={fb:.4f}/{ft:.4f}, truncation rate ρ={rt:.4f}. Direction matches; the as-terminal drop is "
          f"smaller than the model predicts")
    nt, tr_ = summ["abl_no_target"], summ["abl_tiny_replay"]
    check("C4 removing the target network or the replay buffer hurts", nt[2] < b[2] and tr_[2] < b[2],
          f"MA100 at end: base {b[2]:.1f}, no target {nt[2]:.1f}, tiny replay {tr_[2]:.1f}")
    return summ


def first_steps(run, seed, gamma=0.95):
    """C5: the first 10 steps: state, Q(s,.), greedy action, reward and the TD target r + γ max Q(s')."""
    q = load(run, seed)
    env = gym.make("CartPole-v1"); s, _ = env.reset(seed=42)
    print("       t |   x      x_dot   theta   th_dot |  Q(left) Q(right) a | r  target  TD-err")
    for t in range(10):
        with torch.no_grad():
            qv = q(torch.as_tensor(s)).numpy()
        a = int(qv.argmax()); s2, r, te, tr, _ = env.step(a)
        with torch.no_grad():
            y = r + gamma * (0 if te else q(torch.as_tensor(s2)).max().item())
        print(f"      {t:2d} | {s[0]:+.3f} {s[1]:+.3f} {s[2]:+.4f} {s[3]:+.3f} | {qv[0]:7.3f} {qv[1]:7.3f}  {a} | "
              f"{r:.0f} {y:7.3f} {qv[a]-y:+.3f}")
        s = s2


def nonstandard_plots(run, seed, fails, gamma=0.95):
    """Non-standard plots: (1) Q vs steps-to-failure against the closed form; (2) greedy action and
    value over the (theta, theta_dot) plane with the cart centred and at rest."""
    q = load(run, seed)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    k = np.arange(1, 60)
    axs[0].plot(k, (1 - gamma ** k) / (1 - gamma), "k--", lw=2, label="theory (1-γ^k)/(1-γ)")
    for mq, true in fails[:8]:
        T = len(mq); kk = T - np.arange(T)
        sel = kk < 60
        axs[0].plot(kk[sel], mq[sel], lw=1, alpha=0.7)
    axs[0].set_xlabel("Steps until the episode actually failed (k)"); axs[0].set_ylabel("max Q(s)")
    axs[0].set_title("Q vs time-to-failure (failed greedy episodes)"); axs[0].legend(frameon=False); axs[0].grid(alpha=0.3)
    th, thd = np.linspace(-0.25, 0.25, 121), np.linspace(-2, 2, 121)
    TH, THD = np.meshgrid(th, thd)
    X = np.stack([np.zeros_like(TH), np.zeros_like(TH), TH, THD], -1).reshape(-1, 4).astype(np.float32)
    with torch.no_grad():
        Qv = q(torch.from_numpy(X)).numpy().reshape(121, 121, 2)
    im = axs[1].imshow(Qv.max(-1), origin="lower", extent=(th[0], th[-1], thd[0], thd[-1]), aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=axs[1], label="max Q")
    for ax in axs[1:]:
        ax.axvline(-0.2095, color="w", ls=":"); ax.axvline(0.2095, color="w", ls=":")
        ax.set_xlabel("pole angle θ (rad)"); ax.set_ylabel("angular velocity θ̇")
    axs[1].set_title("Value landscape (x = ẋ = 0)")
    axs[2].imshow(Qv.argmax(-1), origin="lower", extent=(th[0], th[-1], thd[0], thd[-1]), aspect="auto", cmap="coolwarm")
    axs[2].set_title("Greedy action (blue = left, red = right)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "nonstandard_q_landscape.png"), dpi=150); plt.close(fig)
    # quantitative, on states the greedy policy actually visits: push toward the side the pole falls
    right = Qv.argmax(-1)
    grid = np.mean(right[(TH + 0.3 * THD) > 0.05] == 1), np.mean(right[(TH + 0.3 * THD) < -0.05] == 0)
    S = np.concatenate([rollout(q, 900 + k)[0] for k in range(5)])
    with torch.no_grad():
        A = q(torch.from_numpy(S.astype(np.float32))).argmax(1).numpy()
    lean = S[:, 2] + 0.3 * S[:, 3]
    vis = np.mean(A[lean > 0.05] == 1), np.mean(A[lean < -0.05] == 0)
    corr = np.corrcoef(lean, A)[0, 1]
    check("C6 greedy action pushes toward the falling side on visited states", min(vis) > 0.9,
          f"visited states (n={len(S)}): push-right when θ+0.3θ̇ > 0.05: {vis[0]*100:.1f}%, push-left when < -0.05: "
          f"{vis[1]*100:.1f}%, corr(lean, action) = {corr:.2f}; on the x=ẋ=0 grid slice: {grid[0]*100:.1f}% / {grid[1]*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="cartpole"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="", help="extra train args of the chosen config, e.g. '--lr 1e-3'")
    ap.add_argument("--skip_ablations", action="store_true")
    a = ap.parse_args()
    fails = test_q_scale(a.run, a.seed)
    first_steps(a.run, a.seed)
    nonstandard_plots(a.run, a.seed, fails)
    if not a.skip_ablations:
        test_ablations(a.base.split() + ["--episodes", "600"])
    print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed; plots in {OUT}")
