# HAN_project

This project implements a **Hierarchical Attention Network (HAN)** for document classification, based on the architecture proposed by Yang et al. (2016). It leverages the hierarchical structure of documents—words forming sentences, and sentences forming documents—using attention mechanisms at both levels.

---

## 📁 Project Structure

- **`han_model.py`**: Defines the HAN architecture with word-level and sentence-level attention.
- **`train.py`**: Script to train the model with configuration from `config.yaml`.
- **`evaluate.py`**: Script to evaluate the model's performance on a test set.
- **`data_loaders.py`**: Utilities for loading and preprocessing datasets.
- **`config.yaml`**: YAML configuration file for all parameters and paths.
- **`.idea/`**: IDE project files (optional, used with PyCharm).

---

## ⚙️ Requirements

- Python 3.x
- PyTorch
- NumPy
- PyYAML
- tqdm
- Any others listed in **`requirements.txt`**

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## 🚀 How to Use

1. **Clone the repository:**
   ```bash
   git clone https://github.com/digesaremarco/HAN_project.git
   cd HAN_project
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure your experiment: Edit the **`config.yaml`** file to set paths, hyperparameters, and other options.**
4. **Train the model:**
    ```bash
   python train.py config.yaml
   ```

## 📄 Dataset

The following datasets were used in this project for document classification tasks:

- **IMDb Large Movie Reviews** – Sentiment analysis (positive vs. negative).
- **Stanford Sentiment Treebank (SST)** – Fine-grained and binary sentiment classification.
- **20 Newsgroups (subset)** – Binary topic classification (e.g., hockey vs. baseball).
- **AG News (subset)** – Binary topic classification (e.g., world vs. business).

Make sure all datasets are preprocessed and tokenized. Hierarchical structure (documents → sentences → words) is expected.

The project uses **Stanford CoreNLP** for sentence and word tokenization. Ensure the tokenizer is configured as specified in the `config.yaml`.

## 📚 Reference

- Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). [Hierarchical Attention Networks for Document Classification](https://www.aclweb.org/anthology/N16-1174/). *NAACL 2016*.
