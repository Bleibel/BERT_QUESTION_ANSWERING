# Advanced RL: PPO with Self-Critical Baseline

> "The same algorithm that powers ChatGPT's alignment, applied to a 1.8M-parameter QA model."

This document explains the Proximal Policy Optimization (PPO) training pipeline and why it represents a significant step beyond standard supervised learning and vanilla REINFORCE.

---

## 1. The Problem with Supervised Learning

Standard QA training uses **maximum likelihood estimation (MLE)**:

```
Loss = -log P(answer_tokens | question, passage)
```

**Why this is suboptimal:**
- We train on token-level cross-entropy, but we evaluate on **span-level F1**
- The model learns to copy labels, not to maximize the metric we care about
- **Exposure bias:** Training sees ground-truth tokens; inference sees its own predictions

**Reinforcement Learning fixes this** by directly optimizing the evaluation metric.

---

## 2. Why REINFORCE Isn't Enough

Our basic RL script (`train_rl.py`) uses REINFORCE:

```
∇J = E[ ∇log π(a|s) × (R(a) - b) ]
```

**Problems with REINFORCE:**
1. **High variance** — a single bad sample can cause a massive gradient update
2. **Unstable training** — rewards oscillate, learning is erratic
3. **Sample inefficient** — needs many rollouts to estimate reliable gradients

**PPO solves all three problems.**

---

## 3. Proximal Policy Optimization (PPO)

PPO is the industry standard RL algorithm used by:
- **OpenAI** (ChatGPT, InstructGPT)
- **Anthropic** (Claude's RLHF)
- **DeepMind** (AlphaStar, Sparrow)

### 3.1 The Core Idea: Clipped Surrogate Objective

Instead of allowing unconstrained policy updates, PPO **clips** the probability ratio:

```
ratio(θ) = π_θ(a|s) / π_θ_old(a|s)

Loss_CLIP(θ) = -E[ min(
    ratio(θ) × A,
    clip(ratio(θ), 1-ε, 1+ε) × A
) ]
```

Where:
- `π_θ` = current policy
- `π_θ_old` = policy used to collect rollouts
- `A` = advantage estimate
- `ε` = clip parameter (typically 0.2)

**Why clipping works:**
- If the new policy becomes too different from the old one, the clipped term kicks in
- This prevents the catastrophic policy collapses that plague vanilla policy gradient
- The model can improve, but not too fast — a form of "training wheels" for RL

### 3.2 Generalized Advantage Estimation (GAE)

PPO uses GAE to compute stable advantage estimates:

```
Â_t = Σ (γλ)^l × δ_{t+l}

where δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

For our QA task (non-sequential), GAE simplifies to:

```
Â = Reward - Value(s)
```

The critic network learns `V(s)` — the expected reward for a given (question, passage).

### 3.3 Actor-Critic Architecture

Our system has **two networks:**

| Component | Role | Architecture |
|-----------|------|--------------|
| **Actor** | Chooses answer spans | Micro-BERT + QA heads |
| **Critic** | Estimates expected reward | Linear layer on [CLS] token |

The actor improves the policy; the critic provides stable baseline estimates.

---

## 4. Self-Critical Baseline

PPO already reduces variance via clipping. We go further with **Self-Critical Sequence Training (SCST)**.

### 4.1 The Idea

Instead of a learned critic baseline, use the model's own **greedy decoding** as the baseline:

```
Advantage = Reward(sampled) - Reward(greedy)
```

**Why this is brilliant:**
- The greedy output represents "the best the model can do without sampling"
- If a sampled answer beats the greedy answer, it gets positive credit
- If it's worse, it gets negative credit — regardless of absolute F1
- **Zero-variance baseline** relative to the model's current capability

### 4.2 Example

| Ground Truth | Greedy Output | Sampled Output | Reward | Baseline | Advantage |
|-------------|---------------|----------------|--------|----------|-----------|
| "Berlin" | "Berlin" | "Berlin" | 1.0 | 1.0 | 0.0 |
| "3.7 million" | "3. 7 million" | "3.7" | 0.4 | 0.4 | 0.0 |
| "Oxygen" | "dioxide" | "Oxygen" | 1.0 | 0.0 | **+1.0** |

In the last row, the sampled answer is **better than greedy** — so it gets strong positive reinforcement, even though greedy was wrong.

---

## 5. Full Loss Function

The PPO training objective combines four terms:

```
L_total = L_policy + c_vf × L_value + c_ent × L_entropy
```

### 5.1 Policy Loss (Clipped PPO)

```
L_policy = -E[ min(ratio × A, clip(ratio, 0.8, 1.2) × A) ]
```

- Encourages actions with positive advantage
- Prevents overly aggressive policy changes

### 5.2 Value Loss (MSE)

```
L_value = E[ (V(s) - R)^2 ]
```

- Trains the critic to predict actual rewards
- Stabilizes advantage estimates

### 5.3 Entropy Bonus

```
L_entropy = -E[ H(π(·|s)) ]
```

- `H` = entropy of the action distribution
- Encourages exploration by rewarding the model for being uncertain
- Prevents premature convergence to a single answer strategy

---

## 6. Comparison of Methods

| Method | Variance | Stability | Sample Efficiency | Impressiveness |
|--------|----------|-----------|-------------------|----------------|
| Supervised (MLE) | Low | High | High | ⭐⭐ |
| REINFORCE | Very High | Low | Very Low | ⭐⭐⭐ |
| **PPO + Self-Critical** | **Low** | **Very High** | **High** | **⭐⭐⭐⭐⭐** |

---

## 7. Training Pipeline

```
Step 1: Supervised Pre-training
    ├── Train on SQuAD with cross-entropy
    └── Result: Decent policy that understands language

Step 2: PPO Fine-tuning
    ├── Collect rollouts (sample + greedy decode)
    ├── Compute rewards (F1) and advantages (sample - greedy)
    ├── Run 4 epochs of clipped PPO updates
    ├── Update both actor (policy) and critic (value)
    └── Result: Policy optimized for F1, not just log-likelihood
```

---

## 8. Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `clip_eps` | 0.2 | PPO clipping radius |
| `ppo_epochs` | 4 | Update epochs per rollout batch |
| `rollout_size` | 16 | Examples collected before update |
| `lr` | 5e-6 | Conservative LR (smaller than supervised) |
| `vf_coef` | 0.5 | Weight of value loss |
| `entropy_coef` | 0.01 | Exploration bonus |

**Why the learning rate is smaller:**
RL is more fragile than supervised learning. A smaller LR prevents the policy from collapsing to a bad local optimum.

---

## 9. When to Use Which RL Method

| Situation | Recommended Method |
|-----------|-------------------|
| Quick course project demo | REINFORCE (`train_rl.py`) |
| **Impressing a professor** | **PPO (`train_ppo.py`)** |
| Research publication | PPO + learned reward model (RLHF) |
| Production deployment | PPO with extensive hyperparameter tuning |

---

## 10. The ChatGPT Connection

ChatGPT's training pipeline (InstructGPT) uses **exactly this architecture**:

```
1. Pre-train GPT on internet text          ←→ Our supervised training
2. Fine-tune on human demonstrations       ←→ Our SQuAD fine-tuning
3. Train reward model on human preferences ←→ Our F1 reward function
4. PPO to optimize against reward model    ←→ Our PPO fine-tuning
```

The difference is scale (175B vs. 1.8M parameters) and the reward source (human judges vs. F1 score). **The algorithm is the same.**

---

## 11. Running PPO Training

```bash
# Step 1: Supervised checkpoint (required)
python train.py --epochs 200 --batch_size 4 --learning_rate 1e-3

# Step 2: PPO fine-tuning
python train_ppo.py \
  --actor checkpoints/micro-bert-qa \
  --dataset data/sample_squad.json \
  --epochs 50 \
  --rollout_size 16 \
  --ppo_epochs 4 \
  --learning_rate 5e-6
```

**Expected output:**
```
Epoch 1/50  | Reward: 0.4231 | Greedy: 0.4200 | Advantage: +0.0031 | PPO Loss: -0.1523
Epoch 2/50  | Reward: 0.4456 | Greedy: 0.4350 | Advantage: +0.0106 | PPO Loss: -0.1987
...
Epoch 50/50 | Reward: 0.5234 | Greedy: 0.5100 | Advantage: +0.0134 | PPO Loss: -0.2845
```

---

## 12. References

1. **Schulman, J., et al. (2017).** *Proximal Policy Optimization Algorithms.* arXiv:1707.06347
2. **Schulman, J., et al. (2015).** *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR.
3. **Rennie, S. J., et al. (2017).** *Self-Critical Sequence Training for Image Captioning.* CVPR.
4. **Ziegler, D. M., et al. (2019).** *Fine-Tuning Language Models from Human Preferences.* arXiv:1909.08593
5. **Ouyang, L., et al. (2022).** *Training Language Models to Follow Instructions with Human Feedback.* NeurIPS (InstructGPT/ChatGPT).
