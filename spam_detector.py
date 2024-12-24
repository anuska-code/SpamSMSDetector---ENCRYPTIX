import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]  # Keep only the label and text columns
df.columns = ['label', 'text']  # Rename columns for clarity

# Convert spam/ham to binary labels
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Convert text to TF-IDF features
tfidf = TfidfVectorizer(max_features=3000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Initialize classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(random_state=42),
    'Linear SVM': LinearSVC(random_state=42)
}

# Train and evaluate each classifier
results = {}
for name, clf in classifiers.items():
    # Train the classifier
    clf.fit(X_train_tfidf, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_tfidf)
    
    # Store results
    results[name] = {
        'report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }

# Print results
print("SMS Spam Detection Results\n")
for name, result in results.items():
    print(f"\n{name} Results:")
    print("-" * 50)
    print("Classification Report:")
    print(result['report'])
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.close()

# Function to test custom messages
def predict_spam(message, classifier='Logistic Regression'):
    # Transform the message using the same TF-IDF vectorizer
    message_tfidf = tfidf.transform([message])
    
    # Make prediction
    prediction = classifiers[classifier].predict(message_tfidf)[0]
    
    return "SPAM" if prediction == 1 else "HAM"

# Test some example messages
example_messages = [
    "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18",
    "Hi, how are you? Let's meet for coffee tomorrow?",
    "FREE UNLIMITED CALLS! Call now to activate your free unlimited calling plan!",
    "Meeting at 3pm in the conference room. Don't forget to bring your laptop."
]

print("\nTesting Example Messages:")
print("-" * 50)
for message in example_messages:
    result = predict_spam(message)
    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")
