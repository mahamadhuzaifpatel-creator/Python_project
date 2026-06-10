import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from data_preprocessing import clean_resume_text

def train_model(dataset_path: str = "data/resume_dataset.csv", models_dir: str = "models"):
    """
    Loads the resume dataset, preprocesses it, trains a Logistic Regression model,
    saves the vectorizer and classifier models, and prints evaluation metrics.
    """
    print(f"Loading dataset from: {dataset_path}...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run generate_dataset.py first.")
        
    df = pd.read_csv(dataset_path)
    
    # Check if necessary columns exist
    if "Category" not in df.columns or "Resume" not in df.columns:
        raise ValueError("Dataset must contain 'Category' and 'Resume' columns.")
        
    print("Preprocessing resume texts (this might take a few seconds)...")
    # Apply cleaning to the Resume column
    df["Cleaned_Resume"] = df["Resume"].apply(clean_resume_text)
    
    # Separate features and labels
    X = df["Cleaned_Resume"]
    y = df["Category"]
    
    # Split the dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Extracting features using TF-IDF...")
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=2500, min_df=2, max_df=0.95)
    
    # Fit and transform on training data, transform testing data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Logistic Regression classifier...")
    # Initialize and train Logistic Regression model
    classifier = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    classifier.fit(X_train_tfidf, y_train)
    
    # Predict on test set
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Completed successfully!")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed Classification Report
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the vectorizer and classifier
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
    classifier_path = os.path.join(models_dir, "resume_classifier.pkl")
    
    print(f"Saving vectorizer to {vectorizer_path}...")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"Saving classifier to {classifier_path}...")
    with open(classifier_path, "wb") as f:
        pickle.dump(classifier, f)
        
    return accuracy, classification_report(y_test, y_pred, output_dict=True)

if __name__ == "__main__":
    # Test model training
    train_model()
