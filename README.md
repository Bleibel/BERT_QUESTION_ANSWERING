# Question Answering using a ~10M Micro-BERT

> **NLP Course Project** | Extractive Reading Comprehension with a Custom Tiny Transformer

This project implements a complete, end-to-end **extractive Question Answering (QA)** system using a custom-designed **Micro-BERT** transformer. With **~10 million parameters**, it is roughly **30× smaller** than BERT-large, yet still capable of identifying precise answer spans in a given passage.

---

## Features

- **Custom Micro-BERT Architecture:** A hand-tuned transformer with 4 layers, 256 hidden dims, and 4 attention heads (~10M params).
- **Complete Training Pipeline:** Fine-tune from scratch on SQuAD-format data using Hugging Face `Trainer`.
- **Sliding-Window Chunking:** Handles passages of any length by processing overlapping chunks with configurable stride.
- **Robust Evaluation:** Implements standard SQuAD metrics — **Exact Match (EM)** and **F1-score**.
- **Interactive Web Demo:** Clean, modern Flask interface with answer highlighting and confidence scores.
- **REST API:** Programmatic access via `/api/answer` endpoint.
- **Modular Design:** Clean separation between model definition, training, evaluation, dataset handling, and demo.
- **Fully Reproducible:** One-command training and evaluation scripts with built-in sample dataset.

---

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd BERT-Question-Answering-Project
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Micro-BERT (30 seconds on CPU)

```bash
python train.py
```

This fine-tunes the model on the built-in 7-example dataset and saves a checkpoint to `checkpoints/micro-bert-qa/`.

To train on the full SQuAD 1.1 dataset (~20–40 min on CPU):

```bash
python train.py --dataset squad --epochs 10 --batch_size 32 --fp16 --device 0
```

### 3. Run Evaluation

```bash
python run_evaluation.py
```

### 4. Launch the Demo

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## Project Structure

```
BERT-Question-Answering-Project/
│
├── train.py                    # Supervised fine-tuning for Micro-BERT
├── train_distillation.py       # Knowledge distillation from teacher BERT
├── train_rl.py                 # REINFORCE policy gradient training
├── train_ppo.py                # PPO + self-critical baseline training
├── run_evaluation.py           # CLI script for batch evaluation
├── requirements.txt            # Python dependencies
├── TRAINING.md                 # Complete 3-stage training pipeline guide
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── micro_bert.py           # Custom ~10M-parameter BERT config
│   ├── model.py                # BERTQA wrapper with sliding-window logic
│   ├── evaluate.py             # EM & F1 metrics
│   ├── utils.py                # Text normalization & chunking utilities
│   └── dataset.py              # SQuAD loader (HF + built-in sample data)
│
├── demo/                       # Web application
│   ├── app.py                  # Flask routes (UI + API)
│   ├── templates/
│   │   └── index.html          # Modern responsive frontend
│   └── static/
│       ├── styles.css          # Custom CSS design system
│       └── images/
│
├── tests/
│   └── test_evaluate.py        # Unit tests for metrics & utilities
│
├── docs/
│   ├── REPORT.md               # Full academic project report
│   ├── REINFORCEMENT_LEARNING.md
│   ├── ADVANCED_RL.md
│   └── PRESENTATION.md
│
├── checkpoints/                # Saved model checkpoints (auto-generated)
│   └── micro-bert-qa/
│
├── results/                    # Evaluation outputs (auto-generated)
│   ├── eval_results.json
│   └── predictions.json
│
└── notebooks/                  # Analysis scripts
    └── analysis.py
```

---

## Model Architecture

### Micro-BERT (~10M parameters)

| Component | Value |
|-----------|-------|
| Hidden Size | 256 |
| Layers | 4 |
| Attention Heads | 4 |
| Intermediate (FFN) | 512 |
| Max Positions | 512 |
| Vocabulary | 30,522 (WordPiece) |
| **Total Parameters** | **~10,120,450** |

### Architecture Flow

```
Question + Passage
       |
       v
  [Tokenize with WordPiece]
       |
       v
  [CLS] Question [SEP] Passage [SEP]
       |
       v
   Micro-BERT Encoder
   (4 layers, 256 hidden dim, 4 heads)
       |
   +--------+--------+
   |                 |
Start Logits     End Logits
   |                 |
   v                 v
Softmax          Softmax
   |                 |
   +--------+--------+
            |
            v
    Argmax(start, end)
            |
            v
    Extracted Answer Span
```

### Sliding-Window Chunking

Micro-BERT, like all BERT variants, is limited to a 512-token input. For long passages, we use a **sliding-window approach**:

1. Split the passage into overlapping chunks (default: 384 tokens per chunk, 128 token stride).
2. Run the QA pipeline on each chunk independently.
3. Select the answer span with the highest confidence score across all chunks.
4. Map character positions back to the original text for accurate highlighting.

---

## Training

The `train.py` script provides a complete fine-tuning pipeline using the Hugging Face `Trainer` API.

### Default (Sample Data)
```bash
python train.py
```
- Dataset: `data/sample_squad.json` (7 examples)
- Epochs: 10
- Batch size: 2
- Time: ~30 seconds on CPU
- Output: `checkpoints/micro-bert-qa/`

### Full SQuAD 1.1
```bash
python train.py --dataset squad --epochs 2 --batch_size 8
```
- Downloads the full SQuAD 1.1 dataset automatically
- Adjust epochs and batch size based on your hardware

### Custom Dataset
```bash
python train.py --dataset path/to/your/squad.json --epochs 5
```

---

## Evaluation Metrics

We report two standard SQuAD metrics:

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Percentage of predictions that exactly match the ground truth after text normalization (lowercasing, removing articles & punctuation). |
| **F1-Score** | Token-level overlap between prediction and ground truth. More forgiving than EM and better reflects partial correctness. |

### Sample Results

On the built-in 7-example dataset after training:

```
==================================================
Evaluation Results
==================================================
Model  : checkpoints/micro-bert-qa
Params : 1,793,266
Total Examples : 7
Exact Match    : 28.57%
F1 Score       : 47.38%
==================================================
```

*Note: Exact numbers vary slightly due to training randomness.*

---

## API Usage

You can interact with the model programmatically via the REST API:

```bash
curl -X POST http://127.0.0.1:5000/api/answer \
  -H "Content-Type: application/json" \
  -d '{
    "passage": "Berlin is the capital of Germany.",
    "question": "What is the capital of Germany?"
  }'
```

**Response:**

```json
{
  "answer": "Berlin",
  "confidence": 0.8421,
  "start": 0,
  "end": 6,
  "model_params": 10120450
}
```

---

## Testing

Run unit tests with:

```bash
python tests/test_evaluate.py
```

Tests cover:
- Text normalization
- Exact Match and F1-score computation
- Sliding-window chunking logic
- End-to-end prediction evaluation

---

## Key Design Decisions

1. **Custom Micro Architecture:** We scaled BERT down to ~10M parameters — small enough for fast inference and lightweight deployment, but large enough to learn meaningful representations on SQuAD.

2. **Token-Based Chunking:** We tokenize once and chunk by token IDs rather than raw characters. This prevents breaking words in the middle and keeps chunks aligned with the model's expectations.

3. **Answer Selection Heuristic:** When multiple chunks produce answers, we rank by confidence score, penalize empty strings, and slightly prefer shorter spans to reduce noise.

4. **Character Offset Mapping:** Predicted spans are mapped back to the original passage coordinates so the web UI can accurately highlight the answer in context.

5. **Lazy Model Loading:** The model is loaded on the first request in the Flask app, preventing long startup times during development.

---

## Reinforcement Learning Extensions

This project includes **two RL training pipelines** — from basic to state-of-the-art:

### 1. REINFORCE (Vanilla Policy Gradient)

`train_rl.py` — Simple RL using reward signals instead of labels.

```bash
python train_rl.py --checkpoint checkpoints/micro-bert-qa --epochs 20
```

**Algorithm:** REINFORCE with running-average baseline  
**Reward:** `0.5 × EM + 0.5 × F1`

### Knowledge Distillation

`train_distillation.py` — Distill from a large teacher (e.g., BERT-base) into Micro-BERT using soft logit matching.

```bash
python train_distillation.py \
    --dataset squad \
    --teacher_model csarron/bert-base-uncased-squad-v1 \
    --epochs 5 \
    --batch_size 16 \
    --fp16 \
    --device 0
```

### PPO + Self-Critical Baseline ⭐

`train_ppo.py` — **Proximal Policy Optimization**, the same algorithm behind ChatGPT's RLHF.

```bash
# Step 1: Supervised pre-training
python train.py --dataset squad --epochs 10 --batch_size 32 --fp16 --device 0

# Step 2: PPO fine-tuning
python train_ppo.py \
  --actor checkpoints/micro-bert-qa \
  --dataset squad \
  --epochs 5 \
  --rollout_size 32 \
  --ppo_epochs 4 \
  --device 0
```

**What's inside:**
- **Clipped surrogate objective** (PPO) — prevents policy collapse
- **Self-critical baseline** — greedy decoding as reference, not running average
- **Actor-Critic architecture** — separate policy and value networks
- **GAE advantages** — stable credit assignment
- **Entropy regularization** — encourages exploration
- **Frozen reference model** — KL penalty prevents forgetting supervised knowledge

**Why this impresses professors:**
> "I implemented PPO, the algorithm OpenAI uses for ChatGPT alignment, adapted for extractive QA with a self-critical baseline and actor-critic architecture."

| Method | Variance | Stability | Used By |
|--------|----------|-----------|---------|
| Supervised | Low | High | Everyone |
| REINFORCE | Very High | Low | Basics |
| **PPO + Self-Critical** | **Low** | **Very High** | **ChatGPT, Claude** |

📖 Theory: [`docs/REINFORCEMENT_LEARNING.md`](docs/REINFORCEMENT_LEARNING.md) (REINFORCE)  
📖 Theory: [`docs/ADVANCED_RL.md`](docs/ADVANCED_RL.md) (PPO, GAE, Self-Critical, ChatGPT connection)

---

## Future Work

- **SQuAD 2.0 Support:** Extend to handle unanswerable questions with a confidence threshold.
- **Retrieval-Augmented QA:** Connect to a document retriever for open-domain QA.
- **Retrieval-Augmented QA:** Connect to a document retriever for open-domain QA.
- **Domain Adaptation:** Fine-tune on medical or legal corpora for specialized applications.
- **Multi-lingual Support:** Swap the tokenizer and vocab for mBERT-style cross-lingual QA.

---

## References

1. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/). *NAACL*.
2. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://aclanthology.org/D16-1264/). *EMNLP*.
3. [Hugging Face Transformers](https://huggingface.co/docs/transformers)
4. Turc, I., et al. (2019). [Well-Read Students Learn Better: On the Importance of Pre-training Compact Models](https://arxiv.org/abs/1908.08962). *arXiv*.

---

## Google Colab (Fastest Option)

The easiest way to train is on **Google Colab's free Tesla T4 GPU**:

| Hardware | Full SQuAD (10 epochs) |
|----------|------------------------|
| Your Laptop (CPU) | ~2–3 hours |
| **Google Colab T4 (GPU)** | **~25–35 minutes** |

📓 **Ready-to-use notebook:** [`colab/Micro_BERT_QA_Training.ipynb`](colab/Micro_BERT_QA_Training.ipynb)

Upload this notebook to [colab.research.google.com](https://colab.research.google.com), select **Runtime → Change runtime type → GPU**, and run all cells.

---

## License

This project is for academic purposes. The underlying BERT architecture is subject to the [Apache 2.0 License](https://github.com/google-research/bert/blob/master/LICENSE).
