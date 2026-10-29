# Spam Email Detection

A supervised machine-learning project that classifies text messages as **spam** or **ham** using TF-IDF features and a Multinomial Naive Bayes classifier.

[![CI checks](https://github.com/Z3raki444/Spam_Email_Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/Z3raki444/Spam_Email_Detection/actions/workflows/ci.yml)

## Project at a glance

| Stage | Approach |
| --- | --- |
| Task | Binary text classification: spam vs. ham |
| Features | TF-IDF text vectorization |
| Model | Multinomial Naive Bayes |
| Evaluation | Train/test split, precision, recall, F1, confusion matrix |
| Quality | Offline tests and headless training/evaluation in GitHub Actions |

**Start with:** [Training and inference code](main.py) · [Unit tests](tests/) · [Environment requirements](requirements.txt)

> **Limitation:** Results are based on one train/test split of a historical dataset, not cross-validation or a deployed email filter.

## Project Overview

The workflow covers the core stages of a text-classification pipeline:

1. load and clean labelled message data
2. encode spam/ham labels
3. split data into training and test sets
4. transform text using TF-IDF
5. train a Multinomial Naive Bayes model
6. evaluate predictions using accuracy, classification metrics and a confusion matrix
7. classify new unseen messages through a reusable prediction function

## Technologies

- Python
- Pandas
- scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Matplotlib
- Seaborn

## Key ML Concepts Demonstrated

- supervised classification
- train/test splitting
- text vectorization
- feature extraction
- model training
- confusion matrices
- precision, recall and F1-score
- inference on new text

## Repository Structure

```text
Spam_Email_Detection/
├── main.py
├── spam.csv
├── output/
├── LICENSE
└── README.md
```

## Run the Project

Install dependencies:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Then run:

```bash
python main.py
```

The script prints model evaluation metrics, displays the confusion matrix and class distribution, and predicts whether a sample message is spam.

## Future Improvements

- add cross-validation
- compare multiple classifiers
- tune model hyperparameters
- save the trained model and vectorizer
- expose predictions through a FastAPI endpoint
- containerize the application with Docker

## Reproducible setup and checks

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

The CSV uses `latin-1` encoding and has the `v1` (ham/spam) and `v2` (message) columns. The [GitHub Actions workflow](.github/workflows/ci.yml) validates this data contract and runs the training/evaluation script in headless mode.

**Evaluation limitations:** This is one random train/test split, not cross-validation or a production deployment. Accuracy alone can be misleading with class imbalance; inspect the precision, recall, F1 and confusion matrix. Messages and labels should not be treated as a representative sample of all current email.
