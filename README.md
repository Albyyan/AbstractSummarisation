# 📝 COMP9444 Project: Abstractive Document Summarisation

This project explores abstractive summarisation using two transformer-based models: **T5** and **BART**, fine-tuned on the **XSum dataset**. The goal is to generate short, coherent summaries of documents that go beyond extractive methods and instead produce paraphrased outputs similar to human-written summaries.

---

## 👥 Group Members

- Abhishek Moramganti (z5421958)  
- Arnav Badrish (z5476224)  
- Sebastien Demoiseau (z5363375)  
- Tin Nguyen (z5256386)  
- Xiaohan Yan (z5478088)  

---

## 📚 Background

Text summarisation is a key task in NLP with applications in research, search, news aggregation, and more. This project focuses on **abstractive summarisation**, where the model generates novel sentences rather than copying directly from the source.

The **XSum dataset** is used for training and evaluation. It features short, focused summaries with high abstraction and low extractive overlap, making it ideal for this task.

---

## 🔍 Related Work

- **Narayan et al. (2018)**: Proposed topic-aware CNNs for extreme summarisation on XSum, showing that summaries require higher abstraction and novel phrasing.
- **Koh et al. (2022)**: Provided empirical comparisons across summarisation datasets, showing XSum’s high compression ratio and low extractive coverage/density, indicating strong abstraction requirements.

---

## 🛠️ Methods

We fine-tuned the following models using HuggingFace Transformers:
- `t5-small` by Google
- `bart-base` by Facebook

Key techniques:
- **Prefixing** inputs with `"summarize:"` for T5
- **Token truncation/padding** to limit processing time
- **Hyperparameter tuning** (learning rate, weight decay)
- **ROUGE metrics** used for evaluation:
  - ROUGE-1: unigram overlap
  - ROUGE-2: bigram overlap
  - ROUGE-L: longest common subsequence

---

## ⚙️ Experimental Setup

- Batch size: 4
- Epochs: 5
- Weight decay: 0.01  
- Learning rates tested:
  - T5: `1e-4`, `3e-4`, `2e-5` → Best: `3e-4`
  - BART: `1e-5`, `2e-5`, `3e-5` → Best: `2e-5`
- Generation parameters:
  - Length penalty: `0.8`
  - Beam search: `4`
  - Max summary length: `128`

---

## 📊 Results

### 📘 Fine-Tuned T5
- ROUGE-1: 34.34%
- ROUGE-2: 12.68%
- ROUGE-L: 27.14%

### 📙 Fine-Tuned BART
- ROUGE-1: 41.84%
- ROUGE-2: 19.12%
- ROUGE-L: 34.06%

🔎 **BART outperformed T5** on all ROUGE metrics, generating summaries closer to reference summaries in both content and structure.

---

## ⚠️ Observations & Limitations

- Both models outperform their non-finetuned versions significantly.
- Common issues:
  - **Hallucination**: Generating content not in the original article
  - **Coherence**: T5 showed more sentence-level incoherence
- Likely due to model size constraints and dataset noise

---

## 🔮 Future Work

- 🔄 Use larger models (e.g., `t5-large`, `bart-large`)
- 💡 Try specialised models like **Pegasus**
- ✏️ Explore **prompt engineering** and **retrieval-augmented generation**
- 🌍 Expand to **multilingual** and **multimodal** summarisation
- 🗨️ Apply to other domains (e.g., dialogue summarisation with SamSum)

---

## 📦 Dataset

- **XSum**: Extreme summarisation dataset with short, abstractive summaries  
  [https://huggingface.co/datasets/xsum](https://huggingface.co/datasets/xsum)

---

## 📈 Evaluation Metric

We used the ROUGE (Recall-Oriented Understudy for Gisting Evaluation) metric to evaluate model performance. This includes:
- ROUGE-1: Overlap of individual words (unigrams)
- ROUGE-2: Overlap of word pairs (bigrams)
- ROUGE-L: Overlap of longest subsequences

---

## ✅ Conclusion

This project demonstrates the effectiveness of transformer-based models like T5 and BART for abstractive summarisation. While BART showed stronger results overall, both models highlight the tradeoffs between abstraction, coherence, and factual correctness. With further tuning and experimentation, these models can form the basis for powerful summarisation tools across many domains.

