# AI Prompt: Generate NLP Course Presentation

Copy and paste the block below into ChatGPT, Claude, or any AI assistant to generate a complete presentation outline.

---

## COPY THIS PROMPT:

```
I need a 10-slide academic presentation for a university NLP course project.

PROJECT DETAILS:
- Title: "Question Answering using a 1.8M Micro-BERT Transformer"
- Course: Natural Language Processing
- Team size: Individual (or specify your team size)

PROJECT SUMMARY:
This project implements an extractive Question Answering system. Instead of using the standard 340M-parameter BERT-large model, I designed and trained a custom Micro-BERT architecture with only ~1.79M parameters (189x smaller). The model uses 2 transformer layers, 56 hidden dimensions, 2 attention heads, and 128 intermediate FFN dimensions. It was fine-tuned on the SQuAD 1.1 dataset for extractive QA.

KEY FEATURES:
- Custom Micro-BERT config (~1.8M parameters)
- Complete training pipeline using Hugging Face Trainer API
- Sliding-window chunking for long passages
- Standard SQuAD evaluation: Exact Match (EM) and F1-score
- Interactive Flask web demo with REST API
- Google Colab notebook for GPU training
- Deployable on low-resource hardware (VPS, edge devices)

DATASET:
- Primary: SQuAD 1.1 (Stanford Question Answering Dataset, ~87k examples)
- Testing: Built-in sample dataset (7 diverse examples covering geography, biology, history)

RESULTS:
- Sample dataset (trained 200 epochs on 7 examples): EM=28.57%, F1=47.38%
- Full SQuAD training (3 epochs): Estimated EM=40-60%, F1=55-70%
- Inference speed: ~145 questions/second on CPU, ~6ms per question
- Model disk size: ~7 MB
- Training time: ~2 hours on laptop CPU, ~6 minutes on Google Colab T4 GPU

TECH STACK:
- PyTorch + Transformers (Hugging Face)
- Flask (web demo)
- WordPiece tokenization
- Sliding-window chunking (384 tokens, 128 stride)
- AdamW optimizer

COMPARISON TABLE:
- BERT-large: 340M params, 1.3 GB, GPU required
- DistilBERT: 66M params, 255 MB
- TinyBERT: 4.4M params, 17 MB
- Micro-BERT (ours): 1.8M params, 7 MB, runs on CPU

DEMO:
The live demo is a Flask web app where users paste a passage, ask a question, and the model highlights the answer span with a confidence score. Also includes a REST API endpoint.

PROJECT LINKS:
- GitHub: https://github.com/Bleibel/BERT_QUESTION_ANSWERING
- Colab: (GPU training notebook included in repo)

PRESENTATION REQUIREMENTS:
1. EXACTLY 10 slides maximum
2. Each slide should cover ~1 minute of talking
3. Must include a 5-minute demo section
4. Must follow this structure:
   - Introduction (problem, motivation, objective)
   - Methodology (dataset, preprocessing, model architecture, tools, implementation, evaluation)
   - Results (experiments, analysis, comparison, limitations, improvements)
   - Conclusion + Demo plan

SLIDE RULES:
- Use minimal text (bullet points, key numbers)
- Include diagrams descriptions where helpful (architecture flow, model size comparison chart)
- Add speaker notes for each slide (what to say)
- Make the demo slide clearly marked (Slide 9 or 10)
- Include a "Thank you / Q&A" slide as the last slide

OUTPUT FORMAT:
Give me the presentation as:
1. Slide-by-slide outline with title + content + speaker notes
2. A summary of what to show during the 5-minute demo
3. Suggested talking points for transitions between slides
```

---

## Tips for Using the Prompt

1. **Paste into ChatGPT-4 or Claude** for best results
2. **If the output is too long**, ask: *"Give me slides 1-5 first, then 6-10"*
3. **To convert to PowerPoint**, copy the output into PowerPoint Designer or Google Slides
4. **For the demo**, open your Flask app (`python app.py`) in a browser tab and screen-share it

---

## Ready-Made Demo Script (5 Minutes)

Use this during your presentation demo:

**Minute 1:** Show the GitHub repo and project structure  
**Minute 2:** Run `python run_evaluation.py` and show the metrics  
**Minute 3:** Open the web demo (`python app.py`), show the UI  
**Minute 4:** Paste a Wikipedia paragraph, ask a question, show the highlighted answer  
**Minute 5:** Show the REST API response with `curl` or Postman, mention 1.8M params vs 340M

---

## Suggested Slide Titles (If You Want to Build It Yourself)

| Slide | Title | Time |
|-------|-------|------|
| 1 | Title + Team | 1 min |
| 2 | Problem & Motivation | 1 min |
| 3 | Related Work (BERT, DistilBERT, etc.) | 1 min |
| 4 | Methodology: Model Architecture | 1 min |
| 5 | Methodology: Training & Dataset | 1 min |
| 6 | Implementation & Tools | 1 min |
| 7 | Results & Analysis | 1 min |
| 8 | Comparison & Limitations | 1 min |
| 9 | **LIVE DEMO** | **5 min** |
| 10 | Conclusion + Q&A | 1 min |
