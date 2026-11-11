"""DQN on CartPole-v1 (gamma = 0.95). Double DQN, Huber loss, replay buffer, hard-updated target network.

  python dqn_cartpole.py train --run cartpole --seed 0
  python dqn_cartpole.py eval  --run cartpole --seed 0     # 500 roll-outs with the trained network
"""
import argparse
import copy
import csv
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "cartpole")


def qnet():
    return nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))


class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, 4), np.float32); self.s2 = np.zeros((cap, 4), np.float32)
        self.a = np.zeros(cap, np.int64); self.r = np.zeros(cap, np.float32); self.d = np.zeros(cap, np.float32)

    def add(self, s, a, r, d, s2):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.d[j], self.s2[j] = s, a, r, d, s2
        self.i = (self.i + 1) % self.cap; self.n = min(self.n + 1, self.cap)

    def sample(self, b):
        idx = np.random.randint(0, self.n, b)
        t = torch.from_numpy
        return t(self.s[idx]), t(self.a[idx]), t(self.r[idx]), t(self.d[idx]), t(self.s2[idx])


def greedy_eval(q, n, seed, eps=0.0):
    env = gym.make("CartPole-v1")
    scores = []
    for k in range(n):
        s, _ = env.reset(seed=seed + k); done, R = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s)).argmax()) if np.random.rand() >= eps else env.action_space.sample()
            s, r, te, tr, _ = env.step(a); R += r; done = te or tr
        scores.append(R)
    return np.array(scores)


def train(a, verbose=True):
    out = os.path.join(RESULTS, a.run, f"seed{a.seed}")
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    # --max_steps is only used by a validation test (makes truncation frequent); default keeps the 500 limit
    env = gym.make("CartPole-v1", **({"max_episode_steps": a.max_steps} if a.max_steps else {}))
    env.action_space.seed(a.seed)
    q = qnet(); tq = copy.deepcopy(q)
    opt = torch.optim.Adam(q.parameters(), lr=a.lr)
    buf = Replay(a.buffer)
    rows, step, best = [], 0, (-1, None)
    for ep in range(1, a.episodes + 1):
        if a.lr_end is not None:  # linear learning-rate decay over training
            for g in opt.param_groups:
                g["lr"] = a.lr + (a.lr_end - a.lr) * (ep - 1) / max(a.episodes - 1, 1)
        s, _ = env.reset(seed=a.seed * 100000 + ep)
        done, R, qs, q0, L = False, 0.0, [], None, 0
        while not done:
            eps = max(a.eps_end, 1.0 - step / a.eps_steps * (1.0 - a.eps_end))
            with torch.no_grad():
                qv = q(torch.as_tensor(s))
            qs.append(qv.max().item()); q0 = qs[0]
            act = env.action_space.sample() if np.random.rand() < eps else int(qv.argmax())
            s2, r, te, tr, _ = env.step(act)
            # terminated (pole fell / cart left) -> no bootstrap; truncated at 500 -> bootstrap from s2
            d = float(te) if not a.trunc_as_terminal else float(te or tr)
            buf.add(s, act, r, d, s2)
            s, R, L, done, step = s2, R + r, L + 1, te or tr, step + 1
            if buf.n >= a.learn_start:
                bs, ba, br, bd, bs2 = buf.sample(a.batch)
                with torch.no_grad():
                    if a.double:
                        a2 = q(bs2).argmax(1, keepdim=True)
                        nq = tq(bs2).gather(1, a2).squeeze(1)
                    else:
                        nq = tq(bs2).max(1).values
                    y = br + a.gamma * (1 - bd) * nq
                pred = q(bs).gather(1, ba[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(pred, y)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(q.parameters(), 10); opt.step()
                if step % a.target_every == 0:
                    tq.load_state_dict(q.state_dict())
        ev = np.nan
        if ep % a.eval_every == 0:  # greedy check on fresh seeds; keep the best network (DQN can forget)
            ev = greedy_eval(q, 10, seed=10**6 + ep).mean()
            # 10 episodes is noisy: confirm promising nets on 50 more episodes before keeping them
            score = greedy_eval(q, 50, seed=2 * 10**6 + ep * 50).mean() if ev >= min(475, best[0]) else ev
            if score >= best[0]:
                best = (score, copy.deepcopy(q.state_dict()))
        rows.append((ep, R, L, np.mean(qs), q0, eps, step, ev))
        if verbose and ep % 50 == 0:
            ma = np.mean([x[1] for x in rows[-100:]])
            print(f"[{a.run} s{a.seed}] ep {ep:4d} step {step:6d} eps {eps:.3f} MA100 {ma:6.1f} "
                  f"maxQ {np.mean(qs):5.2f} eval {ev}", flush=True)
    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "score", "length", "mean_maxQ", "start_maxQ", "epsilon", "step", "greedy_eval10"])
        w.writerows(rows)
    torch.save(q.state_dict(), os.path.join(out, "q_final.pt"))
    torch.save(best[1], os.path.join(out, "q_best.pt"))
    return q


def evaluate(a):
    out = os.path.join(RESULTS, a.run, f"seed{a.seed}")
    q = qnet()
    q.load_state_dict(torch.load(os.path.join(out, a.ckpt)))
    np.random.seed(0)
    sc = greedy_eval(q, 500, seed=10**7, eps=a.eval_eps)
    np.save(os.path.join(out, f"eval_{os.path.splitext(a.ckpt)[0]}.npy"), sc)
    print(f"[{a.run} s{a.seed}] EVAL {a.ckpt}: 500 eps mean {sc.mean():.2f} std {sc.std():.2f} "
          f"solved(500) {np.mean(sc == 500)*100:.1f}%")
    return sc


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--run", default="cartpole")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--lr_end", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=50_000)
    p.add_argument("--learn_start", type=int, default=1000)
    p.add_argument("--target_every", type=int, default=500)
    p.add_argument("--eps_steps", type=int, default=20_000)
    p.add_argument("--eps_end", type=float, default=0.02)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--no_double", dest="double", action="store_false")
    p.add_argument("--trunc_as_terminal", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--ckpt", default="q_best.pt")
    p.add_argument("--eval_eps", type=float, default=0.0)
    return p


if __name__ == "__main__":
    a = parser().parse_args()
    train(a) if a.mode == "train" else evaluate(a)
