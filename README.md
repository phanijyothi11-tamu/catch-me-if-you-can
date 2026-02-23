# 📰 Catch Me If You Can: News Sentiment Across Domains & Languages

A comprehensive study on **Target-Dependent Sentiment Classification (TSC)** in multilingual news, focusing on improving cross-domain, cross-lingual, and multi-entity sentiment understanding using advanced transformer-based architectures.

---

## 🚀 Overview

Target-Dependent Sentiment Classification (TSC) identifies sentiment expressed toward a **specific entity within a sentence**, rather than the overall sentence sentiment.

This project builds upon MAD-TSC and NewsMTSC datasets and proposes improvements to handle:

* Cross-domain generalization
* Multilingual sentiment understanding
* Multi-entity sentence complexity

Our work introduces novel techniques such as **Domain-Adversarial Training, dual-encoder fusion, and entity-localized context modeling** to enhance model robustness. 

---

## 💡 Motivation

* News sentiment is often **implicit and neutral-heavy**
* Models struggle with:

  * Domain shift (NewsMTSC ↔ MAD-TSC)
  * Low-resource languages
  * Multi-entity sentences
* Need for **context-aware, robust, multilingual models**

---

## 🧠 Key Contributions

* 🌍 **Cross-Domain Learning**

  * Domain-Adversarial Neural Network (DANN) for domain-invariant features

* 🌐 **Multilingual Performance**

  * Improved results across multiple languages including Spanish and Dutch

* 🧩 **Multi-Entity Context Modeling**

  * Dynamic context window extraction
  * Position-based weighting for entity importance

* 🔁 **Dataset Extensions**

  * Expanded MAD-TSC with:

    * Cross-domain news data
    * Hindi language support

---

## 📊 Datasets

* **MAD-TSC**

  * Multilingual aligned dataset (8 languages)
  * Complex geopolitical and multi-entity sentences

* **NewsMTSC**

  * English political news dataset
  * Subtle and implicit sentiment expressions

---

## 🏗️ Models & Approaches

### 🔹 Baselines

* TD Model (Target-dependent BERT)
* SPC Model ([CLS] sentence [SEP] target [SEP])
* Prompt-based models
* Base model (without target)

### 🔹 Proposed Methods

* **Domain-Adversarial Training (DANN)**

  * Gradient Reversal Layer (GRL)
  * Domain-invariant feature learning

* **Monolingual + Multilingual Fusion**

  * Dual encoder architecture
  * Combines language-specific and multilingual representations

* **Zero-shot via Machine Translation**

  * Translate inputs to English for inference

* **Dynamic Entity Context Modeling**

  * Local context windows around entities
  * Distance-based weighting
  * BiLSTM + Attention for contextual learning

---

## 📈 Evaluation

### Metrics

* Macro F1-score (primary)
* Per-class F1-score
* Accuracy

### Experimental Settings

* In-domain
* Cross-domain
* Cross-lingual
* Zero-shot (translation-based)

---

## ⚡ Results

* Improved cross-domain generalization
* Enhanced multilingual performance (notably Spanish & Dutch)
* Significant gains in multi-entity sentiment prediction
* Machine translation proved effective for cross-lingual transfer

---

## 🧩 Challenges Addressed

* Domain shift between datasets
* Low-resource language performance
* Multi-entity sentence ambiguity
* Sensitivity to sentence complexity

---

## 🛠 Tech Stack

* Python
* PyTorch / Transformers
* BERT / RoBERTa / DeBERTa
* NumPy, Pandas

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/catch-me-if-you-can.git

cd catch-me-if-you-can

# Install dependencies
pip install -r requirements.txt

# Run training / experiments
python setup.py
```

---

## 🔗 Applications

* Political sentiment analysis
* News analytics and media bias detection
* Cross-lingual NLP systems
* Recommendation and search systems

---

## 🔮 Future Work

* Improve performance for low-resource languages
* Better handling of long and complex sentences
* Real-time deployment for news analytics

---

## 👩‍💻 Authors

* Phani Jyothi Kurada
