import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from CSV file downloaded from Kaggle
# Selecting only relevant columns 'v1' (label) and 'v2' (message)
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns for clarity

# Map textual labels 'ham' and 'spam' to binary numerical values 0 and 1 respectively
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Split the dataset into training and test sets
# 80% data used for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# Convert text data into numerical feature vectors using TF-IDF vectorization
# Stop words (common words) in English are removed to improve performance
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_test_tfidf = vectorizer.transform(X_test)        # Transform test data

# Initialize and train a Multinomial Naive Bayes classifier using the training data
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict labels for the test set messages
y_pred = model.predict(X_test_tfidf)

# Evaluate the model performance by calculating accuracy and detailed classification metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Compute confusion matrix to visualize true vs predicted labels
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix as a heatmap for better understanding of model errors
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Spam (Ham)', 'Spam'],
            yticklabels=['Not Spam (Ham)', 'Spam'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Spam Detection')
plt.show()

# Plot the distribution of classes (ham vs spam) in the original dataset as a bar chart
plt.figure(figsize=(5,4))
df['label'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Class Distribution in Dataset')
plt.xlabel('Label')
plt.ylabel('Count')
plt.show()

# Define a function to predict whether a new message is spam or not
def predict_spam(text):
    text_vec = vectorizer.transform([text])  # Vectorize the input text
    pred = model.predict(text_vec)[0]        # Predict label
    return "Spam" if pred == 1 else "Not Spam"

# Test the prediction function on a sample message
sample_text = "Congratulations! You won a lottery. Claim now!"
print(f"Sample message prediction: {predict_spam(sample_text)}")
