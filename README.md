# Multi-Class Text Classification Using Deep Learning

This project implements a deep learning-based approach for multi-class classification of text entries composed of titles and descriptions. It uses PyTorch to construct and train a neural model that predicts one of several output classes.

## 📰 Dataset

- Columns: `Class`, `Title`, `Description`
- Combined into a `FullText` field
- Contains both `train.csv` and `test.csv` files

## 🔧 Preprocessing

- Concatenate `Title` and `Description`
- Tokenization using `nltk`
- Removal of punctuation and stopwords
- Word frequency analysis and word clouds
- Sequence padding for neural input

## 🧠 Model Architecture

- Built using **PyTorch**
- Input: Tokenized and padded sequences
- Layers:
  - Embedding layer
  - RNN (or similar) or fully connected layers
- Optimization:
  - Adam optimizer
  - Loss: CrossEntropyLoss
  - Learning rate scheduling with `ReduceLROnPlateau`

## 📊 Evaluation Metrics

- Accuracy
- Precision, Recall, F1 Score
- ROC Curve and AUC (macro average)

## 📈 Visuals

- Loss and accuracy curves
- Word cloud from training text
- ROC curve visualization

## 🧰 Requirements

```bash
pip install pandas numpy matplotlib seaborn nltk torch torchinfo wordcloud scikit-learn
