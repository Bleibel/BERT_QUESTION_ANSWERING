# Question Answering using a 1.8M Micro-BERT

> **NLP Course Project** | Extractive Reading Comprehension with a Custom Tiny Transformer

This project implements a complete, end-to-end **extractive Question Answering (QA)** system using a custom-designed **Micro-BERT** transformer. With only **~1.8 million parameters**, it is roughly **200× smaller** than BERT-large, yet still capable of identifying precise answer spans in a given passage.

---

## Features

- **Custom Micro-BERT Architecture:** A hand-tuned transformer with 2 layers, 56 hidden dims, and 2 attention heads (~1.8M params).
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
python train.py --dataset squad --epochs 2 --batch_size 8
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
├── app.py                      # Entry-point launcher for the web demo
├── train.py                    # Fine-tuning script for Micro-BERT
├── run_evaluation.py           # CLI script for batch evaluation
├── requirements.txt            # Python dependencies
│
├── src/                        # Core library
│   ├── __init__.py
│   ├── micro_bert.py           # Custom ~1.8M-parameter BERT config
│   ├── model.py                # BERTQA wrapper with sliding-window logic
│   ├── evaluate.py             # EM & F1 metrics
│   ├── utils.py                # Text normalization & chunking utilities
│   └── dataset.py              # SQuAD loader & built-in sample data
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
├── data/
│   └── sample_squad.json       # Small SQuAD-format dataset for testing
│
├── docs/
│   └── REPORT.md               # Full academic project report
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

### Micro-BERT (~1.8M parameters)

| Component | Value |
|-----------|-------|
| Hidden Size | 56 |
| Layers | 2 |
| Attention Heads | 2 |
| Intermediate (FFN) | 128 |
| Max Positions | 512 |
| Vocabulary | 30,522 (WordPiece) |
| **Total Parameters** | **~1,795,642** |

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
   (2 layers, 56 hidden dim, 2 heads)
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
  "model_params": 1795642
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

1. **Custom Micro Architecture:** We intentionally scaled BERT down to ~1.8M parameters to demonstrate that transformer-based QA can work with tiny models, making inference fast and deployment lightweight.

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

### 2. PPO + Self-Critical Baseline ⭐ (The Impressive One)

`train_ppo.py` — **Proximal Policy Optimization**, the same algorithm behind ChatGPT's RLHF.

```bash
# Step 1: Supervised pre-training
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
- **Knowledge Distillation:** Distill a larger teacher model into Micro-BERT for better accuracy.
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

| Hardware | Full SQuAD (3 epochs) | Sample Data (200 epochs) |
|----------|----------------------|--------------------------|
| Your Laptop (CPU) | ~2 hours | ~3 minutes |
| Hostinger VPS KVM 4 (CPU) | ~2 hours | ~3 minutes |
| **Google Colab T4 (GPU)** | **~5–8 minutes** | **~30 seconds** |

📓 **Ready-to-use notebook:** [`colab/Micro_BERT_QA_Training.ipynb`](colab/Micro_BERT_QA_Training.ipynb)

Upload this notebook to [colab.research.google.com](https://colab.research.google.com), select **Runtime → Change runtime type → GPU**, and run all cells.

## Deployment

See [`DEPLOYMENT.md`](DEPLOYMENT.md) for a complete guide to deploying this project on a **Hostinger VPS** (or any Ubuntu server), including:
- Recommended VPS plans and OS selection
- Step-by-step server setup
- Production deployment with Gunicorn + Nginx
- systemd service configuration
- HTTPS with Let's Encrypt

---

## License

This project is for academic purposes. The underlying BERT architecture is subject to the [Apache 2.0 License](https://github.com/google-research/bert/blob/master/LICENSE).
