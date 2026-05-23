# Reinforcement Learning for Micro-BERT QA

This document explains the REINFORCE-based RL training extension added to the project.

---

## Why RL for QA?

Standard supervised training optimizes the cross-entropy loss between predicted and ground-truth answer spans. This is effective but has limitations:

1. **Mismatched objective:** We train on token-level cross-entropy, but we actually care about **F1 score** or **Exact Match**
2. **Spans vs. labels:** The model learns to copy the label, not to optimize answer quality
3. **Exposure bias:** During training, the model always sees teacher-forced labels; at inference, it sees its own predictions

**Reinforcement Learning directly optimizes the metric we care about** (F1, EM) by treating answer selection as a policy decision.

---

## Algorithm: REINFORCE with Baseline

We use the classic **REINFORCE** (policy gradient) algorithm:

```
Loop:
  1. Model samples an answer span from its probability distribution
  2. We compute Reward = F1(sampled_answer, ground_truth)
  3. Loss = -log_prob(sampled_span) × (Reward - Baseline)
  4. Backpropagate and update weights
```

**Baseline:** A running average of past rewards that reduces variance and stabilizes training.

---

## Reward Design

The reward function combines multiple signals:

```python
Reward = 0.5 × Exact_Match + 0.5 × F1_Score
```

| Component | Weight | Purpose |
|-----------|--------|---------|
| Exact Match | 0.5 | Encourages perfect answers |
| F1 Score | 0.5 | Provides continuous feedback for partial matches |

**Alternative rewards you can try:**
- `F1 - λ × len(answer)` — penalizes overly long answers
- `1 if F1 > 0.8 else 0` — threshold-based sparse reward
- `cosine_similarity(answer_embedding, gold_embedding)` — semantic reward

---

## How to Run

### Prerequisites
You need a supervised checkpoint first. RL fine-tuning works best when starting from a decent model, not random weights.

```bash
# Step 1: Train supervised model (or use existing checkpoint)
python train.py --epochs 200 --batch_size 4 --learning_rate 1e-3

# Step 2: Fine-tune with RL
python train_rl.py \
  --checkpoint checkpoints/micro-bert-qa \
  --dataset data/sample_squad.json \
  --epochs 20 \
  --batch_size 4 \
  --learning_rate 1e-5 \
  --output_dir checkpoints/micro-bert-qa-rl
```

### Expected Behavior
```
Epoch 1/20  | Avg Reward: 0.4231 | Avg Loss: -0.1523 | Baseline: 0.4200
Epoch 2/20  | Avg Reward: 0.4456 | Avg Loss: -0.1987 | Baseline: 0.4350
...
Epoch 20/20 | Avg Reward: 0.5123 | Avg Loss: -0.2845 | Baseline: 0.5100
```

Reward should gradually increase as the policy learns to prefer high-F1 spans.

---

## Architecture of RL Training

```
┌─────────────────┐
│  Question +     │
│  Passage        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Micro-BERT     │ ←── Policy Network
│  (1.8M params)  │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
Start      End
Logits     Logits
    │         │
    ▼         ▼
 Sample   Sample
    │         │
    └────┬────┘
         │
         ▼
   Answer Span
         │
         ▼
   Compute F1
   vs. Ground Truth
         │
         ▼
   Reward Signal
         │
         ▼
  REINFORCE Loss
         │
         ▼
   Update Policy
```

---

## Theoretical Background

### Policy Gradient Theorem

The gradient of the expected reward with respect to policy parameters θ is:

```
∇J(θ) = E[ ∇log π(a|s) × R(s,a) ]
```

Where:
- `π(a|s)` = probability of selecting answer span `a` given input `s`
- `R(s,a)` = reward for that answer
- We approximate the expectation by sampling answers from the current policy

### Baseline Subtraction

The raw policy gradient has high variance. We subtract a baseline `b`:

```
∇J(θ) = E[ ∇log π(a|s) × (R(s,a) - b) ]
```

This doesn't change the expected gradient (the baseline is constant with respect to the action), but it dramatically reduces variance and speeds up learning.

---

## Limitations & When RL Helps

### When RL Works Well
- You have a **decent supervised checkpoint** to start from
- Your reward function is **smooth and informative** (F1 is better than binary EM)
- You have **enough compute** for many sampling iterations

### When RL Struggles
- Starting from **random weights** (exploration is too hard)
- **Sparse rewards** (binary EM gives almost no signal)
- **Very small models** (1.8M params may not have capacity to exploit reward signals)
- **Small datasets** (RL needs many rollouts to estimate gradients)

### For This Project
RL is best used as a **second-stage fine-tuning** after supervised pre-training on SQuAD. It may provide small improvements (2–5% F1) by aligning the model more closely with the evaluation metric.

---

## Possible Extensions

1. **PPO (Proximal Policy Optimization)**
   - More stable than REINFORCE
   - Clips policy updates to prevent collapse
   - Implementation: `from trl import PPOTrainer`

2. **Reward Shaping**
   - Add intermediate rewards for predicting the right sentence
   - Use semantic similarity (SBERT embeddings) instead of token F1

3. **Self-Critical Sequence Training (SCST)**
   - Use greedy decoding as the baseline instead of a moving average
   - Common in image captioning, applicable to QA

4. **Human Feedback (RLHF)**
   - Train a reward model on human preferences
   - Too complex for this project, but state-of-the-art for LLMs

---

## References

1. Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. *Machine Learning*.
2. Rennie, S. J., et al. (2017). Self-Critical Sequence Training for Image Captioning. *CVPR*.
3. Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. *arXiv*.
4. Ziegler, D. M., et al. (2019). Fine-Tuning Language Models from Human Preferences. *arXiv*.
