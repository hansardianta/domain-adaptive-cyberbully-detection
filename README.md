# 🛡️ Cross-Platform Cyberbullying Detection: Domain Adaptation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)
![Status](https://img.shields.io/badge/Status-Completed-green)

## 📌 Project Overview

This project focuses on detecting cyberbullying and hate speech in Indonesian text. The core challenge addressed in this research is **Cross-Platform Generalization**. Models trained on one social media platform (e.g., Twitter/X) often perform poorly when applied to another (e.g., YouTube) due to linguistic and distributional shifts.

To overcome this, we employ **Domain Adaptation Techniques**, specifically **Domain-Adaptive Pretraining (DAPT)** using Masked Language Modeling (MLM), to bridge the gap between Twitter (Source Domain) and YouTube (Target Domain).

## 📂 Repository Structure

The repository is organized into `cyberbully-detection-project` directory, which contains two main directories: `DATASET` and `NOTEBOOKS`.
```text
cyberbully-detection-project/
├── DATASET/
│   ├── final_data_twitter.csv
│   ├── final_data_yt.csv
│   └── dapt_corpus.txt
│
├── NOTEBOOKS/
│   ├── EDA and Preprocessing/
│   ├── Modelling BERT/
│   ├── Modelling Deep Learning/
│   └── Modelling Machine Learning/
````

-----

## 💾 Dataset Information

The data used in this project involves sourced datasets from Twitter and scraped/labeled data from YouTube.

| File Name | Description |
| :--- | :--- |
| `final_data_twitter.csv` | **Source Domain Data.** Preprocessed Twitter dataset containing hate speech and abusive language. <br>Original Source: [Kaggle - Indonesian Abusive and Hate Speech](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text) |
| `final_data_yt.csv` | **Target Domain Data (Labeled).** Preprocessed YouTube comments with manually annotated labels for evaluation. |
| `dapt_corpus.txt` | **Target Domain Data (Unlabeled).** A large corpus of preprocessed YouTube comments used solely for **Masked Language Modeling (MLM)** to adapt the BERT model to the YouTube linguistic style. |

-----

## 🚀 Methodology & Notebooks

### 1\. EDA and Preprocessing

This folder contains notebooks for data cleaning, normalization, and preparation.

  * `EDA_cleaning_twitter.ipynb`: Cleans raw Twitter data. **Output:** `final_data_twitter.csv`.
  * `EDA_cleaning_youtube.ipynb`: Cleans raw labeled YouTube data. **Output:** `final_data_yt.csv`.
  * `EDA_cleaning_dapt.ipynb`: Prepares the unlabeled corpus for the Domain Adaptation process. **Output:** `dapt_corpus.txt`.

### 2\. Modelling BERT (Transformer Models)

This section contains the core experiments regarding Domain Adaptation and Transfer Learning using **IndoBERT** and **IndoBERTweet**.

  * **Domain Adaptation Process:**

      * `ProsesDAPTIndoBERTweet.ipynb`: Performs **Masked Language Modeling (MLM)** on the `IndoBERTweet` model using the `dapt_corpus.txt`. This creates a domain-adapted version of the model.

  * **Experimental Scenarios:**

      * **Direct Transfer (Zero-Shot Cross-Domain):**
          * `indobert.ipynb` & `indobertweet.ipynb`: Models are fine-tuned on **Twitter Data** and directly tested on **YouTube Data** to measure baseline generalization performance.
      * **In-Domain Supervised Fine-Tuning:**
          * `indobert_yt.ipynb` & `indobertweet_nodapt_yt.ipynb`: Models are fine-tuned on **YouTube Data** and tested on **YouTube Data**. This represents the ideal performance with labeled target data.
      * **Domain Adaptation (DAPT):**
          * `indobertweet_dapt.ipynb`: Uses the MLM-pretrained (adapted) model, fine-tunes it on **YouTube Data**, and predicts on **YouTube Data**.

### 3\. Modelling Deep Learning

Baseline comparisons using classical Deep Learning architectures with word embeddings.

  * `cnn.ipynb`: Convolutional Neural Network (CNN) implementation.
  * `lstm.ipynb`: Long Short-Term Memory (LSTM) implementation.
  * `bilstm.ipynb`: Bidirectional LSTM implementation.

### 4\. Modelling Machine Learning

Compares SVM, Naive Bayes, and XGBoost with 4 Text Representation Methods (TF-IDF, GloVe, FastText, Word2Vec)

  * `ML_fullTwitter.ipynb`: Trained with full Twitter Data only.
  * `ML_combinedTraining.ipynb`: Trained with YouTube-Twitter combined data.

-----

## 📊 Results & Performance

We compared various architectures and adaptation strategies. The detailed results, including Precision, Recall, F1-Score, and Accuracy metrics for all experiments, can be accessed via the link below:

👉 **[View Full Experiment Results (Google Sheets)](https://docs.google.com/spreadsheets/d/14P8xnQyvMH-A03VKyBInDMnN95sVx2S7p5GgeZIu_pA/edit?usp=sharing)**
