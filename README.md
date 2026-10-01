# Spam Email Detection

A supervised machine-learning project that classifies text messages as **spam** or **ham** using TF-IDF features and a Multinomial Naive Bayes classifier.

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
