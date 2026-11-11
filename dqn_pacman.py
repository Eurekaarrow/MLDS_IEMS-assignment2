"""DQN on ALE/MsPacman-v5 (gamma = 0.99, 9 actions).

Double DQN + Huber loss + target network + replay buffer, 4 stacked preprocessed frames as the state,
16 parallel envs, linear epsilon and learning-rate schedules.

  python dqn_pacman.py train --run pacman --hours 5
  python dqn_pacman.py eval  --run pacman            # 500 roll-outs
"""
import argparse
import csv
import os
import time

import ale_py
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gym.register_envs(ale_py)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

mspacman_color = 210 + 164 + 74


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


def preprocess_batch(obs):
    """Same as preprocess_observation for a batch (N,210,160,3) -> (N,88,80) int8."""
    img = obs[:, 1:176:2, ::2].sum(axis=3)
    img[img == mspacman_color] = 0
    return (img // 3 - 128).astype(np.int8)


class QNet(nn.Module):
    def __init__(self, n_actions=9):
        super().__init__()
        self.c1 = nn.Conv2d(4, 32, 8, stride=4)
        self.c2 = nn.Conv2d(32, 64, 4, stride=2)
        self.c3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(64 * 7 * 6, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = x.float() / 128.0
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.relu(self.c3(x))
        return self.out(F.relu(self.fc(x.flatten(1))))


class Replay:
    """Stores the 4-frame state and only the newest frame of the next state (s' = s[1:] + new frame)."""
    def __init__(self, cap):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, 4, 88, 80), np.int8)
        self.f2 = np.zeros((cap, 88, 80), np.int8)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.t = np.zeros(cap, np.int64)  # global step when stored

    def add(self, s, a, r, d, f2, step):
        for k in range(len(a)):
            j = self.i
            self.s[j], self.f2[j], self.a[j], self.r[j], self.d[j], self.t[j] = s[k], f2[k], a[k], r[k], d[k], step
            self.i = (self.i + 1) % self.cap
            self.n = min(self.n + 1, self.cap)

    def sample(self, b, device):
        idx = np.random.randint(0, self.n, b)
        s = self.s[idx]
        s2 = np.concatenate([s[:, 1:], self.f2[idx][:, None]], axis=1)
        to = lambda x: torch.from_numpy(x).to(device)
        return to(s), to(self.a[idx]), to(self.r[idx]), to(self.d[idx]), to(s2)


def make_envs(n):
    def thunk():
        import ale_py as _a
        gym.register_envs(_a)
        return gym.make("ALE/MsPacman-v5")
    return gym.vector.AsyncVectorEnv([thunk] * n, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)


def linear(start, end, t, t_end, t_start=0):
    frac = np.clip((t - t_start) / max(t_end - t_start, 1), 0, 1)
    return start + frac * (end - start)


def epsilon(step):
    # 1.0 -> 0.1 over the first 1M transitions, then 0.1 -> 0.01 by 3M
    return linear(1.0, 0.1, step, 1_000_000) if step < 1_000_000 else linear(0.1, 0.01, step, 3_000_000, 1_000_000)


def train(a):
    out = os.path.join(RESULTS, a.run)
    os.makedirs(out, exist_ok=True)
    dev = torch.device(a.device)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    q, tq = QNet().to(dev), QNet().to(dev)
    tq.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=a.lr, eps=1.5e-4)
    buf = Replay(a.buffer)
    envs = make_envs(a.num_envs)
    N = a.num_envs

    obs, info = envs.reset(seed=a.seed)
    f = preprocess_batch(obs)
    stack = np.repeat(f[:, None], 4, axis=1)  # episode start: repeat the first frame 4 times
    lives = info["lives"].copy()
    ep_ret, ep_len, ep_q, ep_q0 = np.zeros(N), np.zeros(N, int), np.zeros(N), np.full(N, np.nan)

    log = open(os.path.join(out, "episodes.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["episode", "score", "length", "mean_maxQ", "start_maxQ", "epsilon", "step", "lr", "wall_s"])
    ulog = open(os.path.join(out, "updates.csv"), "w", newline="")
    uw = csv.writer(ulog)
    uw.writerow(["update", "step", "loss", "mean_q_batch", "mean_target", "wall_s"])
    with open(os.path.join(out, "config.txt"), "w") as fh:
        fh.write(str(vars(a)) + "\n")

    step, updates, episode, t0, losses = 0, 0, 0, time.time(), []
    while episode < a.episodes and time.time() - t0 < a.hours * 3600:
        eps = epsilon(step)
        with torch.no_grad():
            qv = q(torch.from_numpy(stack).to(dev)).cpu().numpy()
        act = qv.argmax(1)
        rand = np.random.rand(N) < eps
        act[rand] = np.random.randint(0, 9, rand.sum())
        maxq = qv.max(1)
        ep_q += maxq
        ep_q0 = np.where(ep_len == 0, maxq, ep_q0)

        obs, r, term, trunc, info = envs.step(act)
        done = term | trunc
        f = preprocess_batch(obs)
        f2 = f.copy()
        if done.any():  # with SAME_STEP autoreset, obs is already the reset frame -> use final_obs for s'
            for i in np.flatnonzero(done):
                f2[i] = preprocess_batch(info["final_obs"][i][None])[0]
        life_lost = (info["lives"] < lives) & ~done
        # bootstrap through truncation, but not through game over or a lost life
        target_done = (term | life_lost).astype(np.float32)
        buf.add(stack, act, np.clip(r, -1, 1), target_done, f2, step)

        ep_ret += r; ep_len += 1
        lives = info["lives"].copy()
        stack = np.concatenate([stack[:, 1:], f[:, None]], axis=1)
        for i in np.flatnonzero(done):
            episode += 1
            w.writerow([episode, ep_ret[i], ep_len[i], ep_q[i] / ep_len[i], ep_q0[i], round(eps, 4), step,
                        opt.param_groups[0]["lr"], round(time.time() - t0, 1)])
            stack[i] = np.repeat(f[i][None], 4, axis=0)
            ep_ret[i], ep_len[i], ep_q[i] = 0, 0, 0
            if episode % 20 == 0:
                log.flush()
                import pandas as pd
                d = pd.read_csv(os.path.join(out, "episodes.csv")).tail(100)
                print(f"ep {episode:5d} step {step/1e6:.2f}M eps {eps:.3f} | MA100 {d.score.mean():7.1f} "
                      f"| maxQ {d.mean_maxQ.mean():6.2f} | loss {np.mean(losses[-500:]) if losses else 0:.4f} "
                      f"| {(time.time()-t0)/60:.1f} min", flush=True)
            if episode % 500 == 0:
                torch.save(q.state_dict(), os.path.join(out, f"q_ep{episode}.pt"))
        step += N

        if buf.n >= a.learn_start:
            for _ in range(N // a.replay_ratio):
                s, ac, rr, dd, s2 = buf.sample(a.batch, dev)
                with torch.no_grad():  # Double DQN: online net picks a', target net evaluates it
                    a2 = q(s2).argmax(1, keepdim=True)
                    y = rr + a.gamma * (1 - dd) * tq(s2).gather(1, a2).squeeze(1)
                qs = q(s).gather(1, ac[:, None]).squeeze(1)
                loss = F.smooth_l1_loss(qs, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10)
                opt.step()
                updates += 1
                losses.append(loss.item()) if updates % 10 == 0 else None
                if updates % a.target_every == 0:
                    tq.load_state_dict(q.state_dict())
                if updates % 1000 == 0:
                    uw.writerow([updates, step, loss.item(), qs.mean().item(), y.mean().item(), round(time.time() - t0, 1)])
                    ulog.flush()
            for g in opt.param_groups:  # learning-rate schedule: linear decay to lr_end by 4M steps
                g["lr"] = linear(a.lr, a.lr_end, step, 4_000_000)
    torch.save(q.state_dict(), os.path.join(out, "q_final.pt"))
    log.close(); ulog.close(); envs.close()
    print(f"done: {episode} episodes, {step} steps, {updates} updates")


@torch.no_grad()
def evaluate(a):
    out = os.path.join(RESULTS, a.run)
    dev = torch.device(a.device)
    q = QNet().to(dev)
    q.load_state_dict(torch.load(os.path.join(out, a.ckpt), map_location=dev))
    n_env, per_env = 20, a.episodes // 20  # fixed episodes per env: avoids over-sampling short episodes
    envs = make_envs(n_env)
    np.random.seed(a.seed + 1)
    obs, _ = envs.reset(seed=a.seed + 1000)
    f = preprocess_batch(obs)
    stack = np.repeat(f[:, None], 4, axis=1)
    count, ret, scores = np.zeros(n_env, int), np.zeros(n_env), []
    while (count < per_env).any():
        act = q(torch.from_numpy(stack).to(dev)).argmax(1).cpu().numpy()
        rand = np.random.rand(n_env) < a.eval_eps
        act[rand] = np.random.randint(0, 9, rand.sum())
        obs, r, term, trunc, _ = envs.step(act)
        f = preprocess_batch(obs)
        ret += r
        stack = np.concatenate([stack[:, 1:], f[:, None]], axis=1)
        for i in np.flatnonzero(term | trunc):
            if count[i] < per_env:
                scores.append(ret[i]); count[i] += 1
            ret[i] = 0
            stack[i] = np.repeat(f[i][None], 4, axis=0)
    scores = np.array(scores)
    np.save(os.path.join(out, f"eval_{os.path.splitext(a.ckpt)[0]}.npy"), scores)
    print(f"EVAL {a.ckpt}: {len(scores)} episodes, mean {scores.mean():.1f}, std {scores.std():.1f}")
    envs.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("--run", default="pacman")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr_end", type=float, default=5e-5)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--buffer", type=int, default=300_000)
    p.add_argument("--learn_start", type=int, default=20_000)
    p.add_argument("--target_every", type=int, default=2500, help="gradient updates between target syncs")
    p.add_argument("--replay_ratio", type=int, default=4, help="env transitions per gradient update")
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--episodes", type=int, default=6000)
    p.add_argument("--hours", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--ckpt", default="q_final.pt")
    p.add_argument("--eval_eps", type=float, default=0.01)
    a = p.parse_args()
    a.episodes = a.episodes if a.mode == "train" else 500
    train(a) if a.mode == "train" else evaluate(a)
