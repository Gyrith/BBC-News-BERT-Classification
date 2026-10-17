# BBC News Classification with BERT

Classifies BBC news articles into five categories (business, entertainment, politics, sport, tech) by fine-tuning BERT. Built as a lab for NewsInsight, a fictional media analytics company, to replace slow and inconsistent manual categorization.

## Approaches
The notebook implements two versions of the same task:
- **TensorFlow:** Small BERT from TensorFlow Hub, fully fine-tuned
- **Hugging Face:** `bert-base-uncased` with a frozen encoder and a trained classification head

## Dataset
BBC News Archive (`bbc_news_data.csv`, tab-separated): 2,225 articles with `category`, `filename`, `title` and `content` columns.

## Workflow
1. Explore the data (category balance, article length, missing values)
2. Stratified split: 75% train+validation, 25% test, with validation taken from the training portion
3. Tokenize (128 tokens) and build the BERT classifier
4. Fine-tune for 5 epochs with early stopping
5. Evaluate on the test set: accuracy, precision, recall, F1, confusion matrix
6. Predict the category of new articles with an inference function

## Results
| Model | Test accuracy |
|---|---|
| TensorFlow (Small BERT, fully fine-tuned) | 95.5% |
| Hugging Face (`bert-base-uncased`, frozen encoder) | 63.6% |

Sport was the easiest category to classify and business the hardest. The Hugging Face model trains only the classification head, which explains its lower accuracy.

## Setup
Run in Google Colab with a GPU (Runtime > Change runtime type > T4 GPU).

```bash
pip install tensorflow-hub tf-keras tensorflow-text transformers datasets accelerate scikit-learn seaborn
```
> `tensorflow-text` does not install on Windows. Use Colab or WSL2 for the TensorFlow part.

## Files
- `bert_classification_lab.ipynb`: the lab notebook
- `bbc_news_data.csv`: the dataset

## Tools
Python, TensorFlow, Hugging Face Transformers, PyTorch, scikit-learn, pandas, matplotlib, seaborn