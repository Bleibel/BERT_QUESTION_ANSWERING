# Training Pipeline Guide

This guide covers the complete 3-stage training pipeline for Micro-BERT QA.

## Architecture

- **Model**: Custom Micro-BERT
- **Parameters**: ~10,120,450 (4 layers, 256 hidden, 4 heads, 512 intermediate)
- **Size**: ~3% of BERT-large

---

## Stage 1: Supervised Pre-training

Train from scratch on SQuAD 1.1 using cross-entropy loss.

```bash
python train.py \
    --dataset squad \
    --epochs 10 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --fp16 \
    --device 0
```

**Colab (T4 GPU):**
```bash
%cd /content/BERT_QUESTION_ANSWERING
!python train.py --dataset squad --epochs 10 --batch_size 32 --fp16 --device 0
```

**Expected time:** ~25–35 minutes  
**Output:** `checkpoints/micro-bert-qa`

---

## Stage 2: Knowledge Distillation (Optional but Recommended)

Distill from a large teacher model to boost accuracy.

```bash
python train_distillation.py \
    --dataset squad \
    --teacher_model csarron/bert-base-uncased-squad-v1 \
    --output_dir checkpoints/micro-bert-qa-distilled \
    --epochs 5 \
    --batch_size 16 \
    --alpha 0.5 \
    --temperature 2.0 \
    --fp16 \
    --device 0
```

**Expected boost:** +10–20% F1  
**Output:** `checkpoints/micro-bert-qa-distilled`

---

## Stage 3: GRPO Fine-tuning (Optional, Recommended)

Align the model with F1-based rewards using **Group Relative Policy Optimization** — the same RL algorithm DeepSeek-R1 used.

GRPO is simpler than PPO: it **removes the critic network** and uses the average reward of a sampled group as the baseline.

```bash
python train_grpo.py \
    --checkpoint checkpoints/micro-bert-qa \
    --dataset squad \
    --epochs 5 \
    --group_size 8 \
    --device 0
```

**Expected boost:** +3–8% F1  
**Output:** `checkpoints/micro-bert-qa-grpo`

### Alternative: PPO Fine-tuning

If you prefer the classic PPO with actor-critic:

```bash
python train_ppo.py \
    --actor checkpoints/micro-bert-qa \
    --dataset squad \
    --epochs 5 \
    --rollout_size 32 \
    --ppo_epochs 4 \
    --device 0
```

**Output:** `checkpoints/micro-bert-qa-ppo`

---

## Quick Evaluation

After any stage, evaluate on SQuAD validation:

```bash
python run_evaluation.py --dataset squad --max-examples 1000 --device 0
```

---

## Training Tips

| Tip | Why |
|-----|-----|
| Use `--fp16` on GPU | 2× faster training, less memory |
| Batch size 32+ | More stable gradients |
| 10 epochs supervised | Convergence for ~10M params |
| Distill from strong teacher | Biggest single improvement |
| PPO after supervised | Aligns loss with F1 metric |
