# Presentation: Question Answering using a 1.8M Micro-BERT

**Course:** Natural Language Processing  
**Presenter:** [Your Name]  
**GitHub:** https://github.com/Bleibel/BERT_QUESTION_ANSWERING  
**Total Time:** 10 minutes + 5-minute demo

---

## Slide 1: Title Slide (1 min)

**Title:** Question Answering using a 1.8M Micro-BERT Transformer

**Subtitle:** A Lightweight Extractive QA System for Resource-Constrained Environments

**Key Stats on Slide:**
- 1,793,266 parameters
- 189× smaller than BERT-large
- Trained on SQuAD 1.1
- Runs on CPU at 145 QPS

**Speaker Notes:**
"Good morning. Today I'll present my NLP course project: a custom-designed transformer for extractive question answering. Instead of using the standard 340-million parameter BERT-large, I built a Micro-BERT with under 2 million parameters that can train and run entirely on a CPU or cheap VPS."

---

## Slide 2: Problem & Motivation (1 min)

**Title:** The Problem: QA Models Are Too Big

**Content:**
- BERT-large: 340M params, needs GPU, ~1.3 GB
- DistilBERT: 66M params, still heavy for edge devices
- Mobile/IoT/VPS environments have strict limits
- **Question:** How small can a transformer be and still do QA?

**Speaker Notes:**
"State-of-the-art QA models achieve great accuracy, but they're enormous. BERT-large is 340 million parameters and needs a powerful GPU. For a course project — or real deployment on a cheap server or mobile device — that's overkill. I wanted to find the lower bound: what's the smallest transformer that can still answer questions?"

---

## Slide 3: Related Work (1 min)

**Title:** From BERT to Micro-BERT

**Comparison Table:**

| Model | Params | Size | Needs GPU? |
|-------|--------|------|------------|
| BERT-large | 340M | 1.3 GB | Yes |
| DistilBERT | 66M | 255 MB | Preferred |
| TinyBERT | 4.4M | 17 MB | Optional |
| **Micro-BERT (ours)** | **1.8M** | **7 MB** | **No** |

**Speaker Notes:**
"Researchers have compressed BERT through distillation and pruning. DistilBERT got it to 66 million. TinyBERT pushed further to 4.4 million. My Micro-BERT goes even smaller — 1.8 million parameters — by redesigning the architecture from scratch with fewer layers, smaller hidden dimensions, and narrower FFN layers."

---

## Slide 4: Methodology — Architecture (1 min)

**Title:** Micro-BERT Architecture

**Diagram (text representation):**
```
[Question] + [Passage]
    ↓
WordPiece Tokenizer + [CLS]/[SEP]
    ↓
Micro-BERT Encoder
  (2 layers, 56 hidden, 2 heads, 128 FFN)
    ↓
Start Logits ──┐
               ├──→ Argmax → Answer Span
End Logits ────┘
```

**Specs on Slide:**
- Hidden size: 56
- Layers: 2
- Attention heads: 2
- Intermediate: 128
- Vocab: 30,522
- **Total: 1,793,266 parameters**

**Speaker Notes:**
"The architecture is a standard BERT, just scaled down aggressively. Two layers instead of twenty-four. Fifty-six hidden dimensions instead of one thousand twenty-four. The model still uses WordPiece tokenization, learned positional embeddings, and GELU activations — but it's tiny enough to fit in 7 megabytes."

---

## Slide 5: Methodology — Training & Data (1 min)

**Title:** Training Pipeline

**Content:**
- **Dataset:** SQuAD 1.1 (~87,000 question-answer pairs)
- **Preprocessing:** Tokenization with sliding-window chunking (384 tokens, 128 stride)
- **Framework:** Hugging Face Trainer API
- **Optimizer:** AdamW, LR = 3e-5 (full SQuAD) / 1e-3 (sample data)
- **Hardware:** Google Colab T4 GPU (~6 minutes) or CPU (~2 hours)

**Speaker Notes:**
"I used the Stanford Question Answering Dataset — the standard benchmark for extractive QA. For long passages, I implemented sliding-window chunking so the model can handle any document length. Training was done with the Hugging Face Trainer. On a free Google Colab GPU, three epochs of full SQuAD training takes about six minutes. On a CPU, it's roughly two hours."

---

## Slide 6: Implementation (1 min)

**Title:** System Architecture

**Content:**
```
Project Structure:
├── src/          (model, training, evaluation, utils)
├── demo/         (Flask web app + REST API)
├── train.py      (training script)
├── run_eval.py   (evaluation script)
├── colab/        (GPU notebook)
└── docs/         (report + deployment guide)
```

**Key Features:**
- Modular Python package
- Token-based sliding-window chunking
- REST API for programmatic access
- Deployable on Hostinger VPS / any Ubuntu server

**Speaker Notes:**
"The code is fully modular. The src package handles the model, training, and evaluation. The demo folder contains a Flask web application with a modern UI and a REST API endpoint. I also included a Google Colab notebook for one-click GPU training and a deployment guide for Hostinger VPS."

---

## Slide 7: Results (1 min)

**Title:** Results & Analysis

**Results Table:**

| Experiment | Exact Match | F1 Score |
|------------|-------------|----------|
| Sample data (7 ex, scratch) | 28.57% | 47.38% |
| Full SQuAD (3 epochs, scratch) | ~40-60%* | ~55-70%* |
| Inference speed (CPU) | — | 145 QPS |

*Estimated based on architecture scaling

**Error Analysis:**
- Correct: "Berlin", "Turing machine"
- Partial: "3. 7 million" vs "3.7 million" (spacing)
- Wrong: Needs deeper layers for complex reasoning

**Speaker Notes:**
"On the sample dataset, the model achieved twenty-eight percent exact match and forty-seven percent F1. That's modest — but remember, this model was trained from scratch with random weights, not pre-trained on Wikipedia like BERT. The errors are revealing: simple factoid questions are correct, but the two-layer model struggles with longer, more complex reasoning chains."

---

## Slide 8: Limitations & Future Work (1 min)

**Title:** Limitations & Improvements

**Limitations:**
- Trained from scratch (no pre-training on large corpus)
- Only 2 layers → limited long-range dependencies
- Extractive only (cannot generate novel answers)
- English-only (uncased WordPiece)

**Future Improvements:**
- Pre-train on Wikipedia/BookCorpus before SQuAD fine-tuning
- Knowledge distillation from BERT-base teacher
- Add SQuAD 2.0 support for unanswerable questions
- Quantization for mobile deployment

**Speaker Notes:**
"The biggest limitation is the lack of pre-training. BERT spends days learning language on Wikipedia before seeing SQuAD. My model starts from random weights. Future work would add a pre-training phase, or use knowledge distillation to transfer BERT's knowledge into this tiny architecture. Support for unanswerable questions and multilingual expansion are also natural next steps."

---

## Slide 9: LIVE DEMO (5 min)

**Title:** Live Demo

**Demo Script:**

**Minute 1 — Show the Repo & Structure:**
- Open GitHub: `github.com/Bleibel/BERT_QUESTION_ANSWERING`
- Highlight: `src/micro_bert.py`, `train.py`, `demo/`

**Minute 2 — Run Evaluation:**
```bash
python run_evaluation.py --dataset data/sample_squad.json
```
- Show output: 1,793,266 params, 28.57% EM, 47.38% F1

**Minute 3 — Web Demo:**
```bash
python app.py
```
- Open browser to `localhost:5000`
- Show the UI with model badge: "1,793,266 parameters"

**Minute 4 — Interactive QA:**
- Paste passage about Berlin
- Ask: "What is the capital of Germany?"
- Show highlighted answer: **Berlin**
- Ask: "How many inhabitants does Berlin have?"
- Show answer: **3. 7 million** (explain the partial match)

**Minute 5 — API Call:**
```bash
curl -X POST http://127.0.0.1:5000/api/answer \
  -H "Content-Type: application/json" \
  -d '{"passage":"Berlin is the capital...","question":"Capital of Germany?"}'
```
- Show JSON response with confidence score
- Emphasize: this runs on a $5/month VPS

**Speaker Notes:**
"Let me show you the system in action. First, the repo structure. Then evaluation. Now the web demo — notice the model info badge showing under two million parameters. Let me ask it a question about Berlin. And here's the REST API for integration into other applications. All of this runs comfortably on a cheap VPS or even a Raspberry Pi."

---

## Slide 10: Conclusion & Q&A (1 min)

**Title:** Conclusion

**Content:**
- Built a **1.8M-parameter transformer** for extractive QA
- **189× smaller** than BERT-large, runs on CPU
- Complete pipeline: train → evaluate → deploy
- Demonstrates the **trade-off between model size and accuracy**

**Takeaway:**
> "Extreme compression is possible, but pre-training is essential for competitive performance."

**Q&A**

**Speaker Notes:**
"To conclude: I designed, trained, and deployed a question-answering model with under two million parameters. It's nearly two hundred times smaller than BERT-large and runs on a CPU — but the accuracy trade-off is real. The key lesson is that transformer architectures can be scaled down dramatically, but pre-training remains critical for strong performance. Thank you, and I'm happy to take questions."

---

# Presentation Checklist

Before presenting, verify:

- [ ] Flask app running (`python app.py`)
- [ ] Evaluation script tested (`python run_evaluation.py`)
- [ ] Browser tab open at `localhost:5000`
- [ ] Terminal ready for API curl demo
- [ ] GitHub repo open in another tab
- [ ] Timer set (10 min talk + 5 min demo = 15 min total)

---

# Bonus: 1-Minute Pitch Version

If the professor asks "Summarize your project in one minute":

> "I built an extractive question-answering system using a custom transformer called Micro-BERT. It has 1.8 million parameters — 189 times smaller than BERT-large — yet it can identify answer spans in text. I trained it on the SQuAD dataset, implemented sliding-window chunking for long documents, built a Flask web demo with a REST API, and deployed it on a Hostinger VPS. The project demonstrates that transformer-based QA is possible at extreme scales, though pre-training is needed for competitive accuracy."
