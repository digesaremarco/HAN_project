# HAN_project

The Hierarchical Attention Network (HAN) is a document classification model that leverages the inherent hierarchical structure of documents: words forming sentences, and sentences forming documents. It employs attention mechanisms at two levels to identify key words within sentences and key sentences within documents, focusing on the most relevant information for classification.

### 🧱 Architecture in Brief

1.  **Word Encoder:** Processes each sentence, transforming words into vector embeddings using a bidirectional recurrent neural network (Bi-RNN) to capture context.
2.  **Word Attention Layer:** Assigns an importance weight to each word in the sentence, highlighting those most significant to the sentence's meaning within the document's context. Produces a weighted sentence representation vector.
3.  **Sentence Encoder:** Processes the sequence of sentence vectors using another Bi-RNN to understand the relationships between sentences in the document.
4.  **Sentence Attention Layer:** Assigns an importance weight to each sentence in the document, identifying those most crucial for classification. Produces a weighted document representation vector.
5.  **Classification Layer:** Uses the document vector to predict the document's class via a fully connected layer and an activation function (e.g., softmax).

### 💡 Why is it Effective?

* **Hierarchical Structure:** Explicitly models the organization of documents.
* **Interpretable Attention:** Indicates which words and sentences are most important.
* **Focus on the Essential:** Ignores less relevant information.
* **End-to-End Learning:** Optimizes the entire network for classification.

---

## 📁 Project Structure

This project is organized into the following key modules:

-   **`han_model.py`**: Contains the definition of the Hierarchical Attention Network (HAN) architecture, including the implementation of word-level and sentence-level attention mechanisms.
-   **`train.py`**: Provides the script for training the HAN model. It loads configuration parameters from the `config.yaml` file and orchestrates the training process.
-   **`evaluate.py`**: Implements the script used to evaluate the trained model's performance on a designated test dataset.
-   **`data_loaders.py`**: Includes utility functions and classes responsible for loading and preprocessing the datasets used for training and evaluation. This module likely handles tokenization and batching.
-   **`config.yaml`**: Serves as the central configuration file, storing all relevant parameters such as file paths, hyperparameters for the model, and training settings.

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
