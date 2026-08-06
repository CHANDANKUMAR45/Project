# 📚 Topic Classification using Custom Deep Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

# 📌 Project Overview

This project implements a **Topic Classification System** from scratch using **Deep Learning**.

The objective is to classify text into its corresponding topic using:

- Custom Tokenizer
- Custom Vocabulary
- BiLSTM
- Attention Mechanism
- Fully Connected Neural Network

No pretrained models (BERT, GPT, RoBERTa, etc.) were used.

---

# 📂 Dataset

Dataset Format

```
Parquet
```

Columns

| Column | Description |
|---------|-------------|
| DATA | Input Text |
| TOPIC | Output Label |

Dataset Size

```
10 Million Rows
```

---

# 🚀 Features

- Custom Tokenizer
- Custom Vocabulary
- PyTorch Dataset
- BiLSTM
- Attention Layer
- TF-IDF Baseline Models
- Model Evaluation
- Inference
- Confusion Matrix
- Error Analysis

---

# 🧠 Model Architecture

```
Input Text
     │
     ▼
Tokenizer
     │
     ▼
Vocabulary Encoding
     │
     ▼
Embedding Layer
     │
     ▼
BiLSTM
     │
     ▼
Attention
     │
     ▼
Dropout
     │
     ▼
Fully Connected Layer
     │
     ▼
Topic Prediction
```

---

# 📁 Project Structure

```
Topic-Classification/

│

├── dataset/

├── src/
│   ├── tokenizer.py
│   ├── vocabulary.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── utils.py

│

├── final_models/

├── report.pdf

├── README.md

├── requirements.txt

└── .gitignore
```

---

# ⚙ Installation

Clone repository

```bash
git clone https://github.com/CHANDANKUMAR45/Topic-Classification.git
```

Go inside project

```bash
cd Topic-Classification
```

Install requirements

```bash
pip install -r requirements.txt
```

---

# ▶ Training

```bash
cd src

python train.py
```

---

# 📈 Evaluation

```bash
python evaluate.py
```

---

# 🔍 Inference

```bash
python inference.py
```

Example

```
Input

India defeated Australia in World Cup Final.

Prediction

Sports
```

---

# 📊 Metrics

The following metrics are computed:

- Accuracy
- Precision
- Recall
- F1 Score

---

# 📂 Output Files

```
topic_classifier.pth

vocabulary.pkl

metrics.json

classification_report.txt

confusion_matrix.png
```

---

# 🛠 Tech Stack

- Python
- PyTorch
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- PyArrow

---

# 📌 Future Improvements

- Transformer trained from scratch
- Mixed Precision Training
- Distributed Training
- Hyperparameter Optimization

---



---