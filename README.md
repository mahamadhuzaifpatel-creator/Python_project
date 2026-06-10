# AI-Based Resume Screening System

An advanced Python for Data Science and Natural Language Processing (NLP) mini project that automates the screening and classification of resumes into various job categories, extracts technical skills, and displays interactive analytics. Suitable for college mini projects and built with a premium Streamlit web interface.

---

## 📝 Project Abstract

In the modern corporate recruiting environment, human resource departments are inundated with hundreds or thousands of resumes for a single job opening. Manually screening each profile to match candidates with appropriate roles is an extremely labor-intensive, time-consuming, and error-prone process. 

This project presents an **AI-Based Resume Screening System** designed to solve this problem by automating the initial stages of resume filtering. The system utilizes Natural Language Processing (NLP) to clean and preprocess raw text from uploaded PDF resumes (handling lowercasing, punctuation removal, stripping hyperlinks, emails, hashtags, and removing NLTK English stopwords). Features are extracted using the **Term Frequency-Inverse Document Frequency (TF-IDF)** vectorization model. A supervised machine learning classifier using **Logistic Regression** is trained on a labeled dataset of resumes to classify applicants into six major job roles: 
1. **Data Science**
2. **Web Development**
3. **Java Developer**
4. **Python Developer**
5. **Testing**
6. **HR**

A rule-based keyword matching algorithm checks word boundaries to extract specific technical skills corresponding to each job family without false positives (such as matching "Java" within "JavaScript"). The application provides two operating interfaces:
- **CLI (Command Line Interface)** for model training, validation reporting, static plot generation, and raw text prediction.
- **Web Dashboard (Streamlit)** which offers a premium user interface with file upload (PDF support), real-time classification probability (matching score), skills tagging, and dataset analytics.

---

## 📂 Project Folder Structure

```text
AI_Resume_Screening_System/
│
├── data/
│   └── resume_dataset.csv          # Labeled resume dataset CSV (180 entries)
│
├── models/
│   ├── resume_classifier.pkl       # Trained Logistic Regression classifier
│   └── tfidf_vectorizer.pkl        # Fitted TF-IDF Vectorizer
│
├── plots/
│   ├── category_distribution.png   # Job category distribution plot
│   ├── common_skills.png          # Visual frequency chart of skills
│   └── model_accuracy.png          # Model metrics (Precision/Recall/F1) chart
│
├── data_preprocessing.py            # Text cleaning utilities (lower, regex, nltk)
├── generate_dataset.py              # Script to programmatically generate data
├── model_training.py                # Train model, save .pkl artifacts, print classification report
├── resume_predictor.py              # Resume Predictor class and Skill Extractor engine
├── visualization.py                 # Graph plotting script (matplotlib and seaborn)
├── main.py                          # CLI runner script & interactive prompt
├── app.py                           # Streamlit Web Dashboard application
├── requirements.txt                 # Project dependencies list
└── README.md                        # Documentation, abstract, viva questions, output description
```

---

## ⚙️ Requirements & Installation

The project uses Python 3.8+ and standard data science libraries.

### Prerequisites

Clone or download the project files into a folder, open VS Code, and open the project directory.

### Installation Steps

1. **Create a Virtual Environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   # Activate on Windows:
   venv\Scripts\activate
   # Activate on macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Dependencies:**
   Ensure `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `nltk`, `streamlit`, `pypdf`, and `seaborn` are installed.

---

## 🚀 How to Run the Project

### Option A: Command Line Interface (CLI)

Run `main.py`. This script automatically:
- Checks for the dataset (creates one if missing).
- Preprocesses texts and trains the Logistic Regression model.
- Evaluates the model, displaying test accuracy and the classification report.
- Saves trained model files (`models/`) and chart images (`plots/`).
- Enters an interactive loop allowing you to test predictions with default samples or custom text.

```bash
python main.py
```

### Option B: Streamlit Web Dashboard (Recommended)

Start the local Streamlit server to run the graphical web application:

```bash
streamlit run app.py
```

Streamlit will launch a page in your web browser (typically at `http://localhost:8501`).
- **📂 Resume Screener Tab**: Upload a PDF resume or load sample profiles to get instantaneous match prediction, confidence score, and visual tags for extracted skills.
- **📊 Dataset Insights Tab**: View the distribution of classes and most common skills.
- **📈 Model Performance Tab**: Check classification scores (Precision, Recall, F1-Score) and interactive performance charts.

---

## 📸 Expected Output Screenshots Description

When compiling your project report, you can add screenshots matching the following layout details:

1. **CLI Model Training & Accuracy**:
   - Running `python main.py` in the terminal output shows the loading message, the NLTK stopword download logs, a printout of the dataset shape (180 entries, 6 categories), and a text table representing the scikit-learn `classification_report`.
   - The test accuracy should show a high value (typically **94% - 100%** due to well-defined synthetic vocabulary).

2. **CLI Interactive Predictor Loop**:
   - The terminal shows options `(1-4)`. Selecting option `1` prints a snippet of a Data Science resume, followed by an output block with stars:
     - `Predicted Suitable Category: Data Science`
     - `Role Match/Confidence Score: ~95%`
     - `Extracted Skills: Git, Jupyter Notebook, NumPy, Pandas, Python, Scikit-Learn, PyTorch, SQL...`

3. **Streamlit App - Main Screener View**:
   - Displays a dark/light responsive layout featuring a title "AI-Based Resume Screening System" with a background blue-to-red color gradient.
   - Shows the file uploader widget. Uploading a PDF will display a success message: `Successfully extracted text from uploaded PDF!`.
   - Displays two responsive visual cards showing the prediction details: **Predicted Match Role** and **Profile Match Strength (Confidence %)**, accompanied by a clean green progress bar.
   - Features categorized colored tags (purple and blue) representing the extracted technical skills.

4. **Streamlit App - Dataset Insights**:
   - Two side-by-side graphical figures showing a horizontal bar graph of category distribution and a horizontal bar graph of the top 15 most frequent skills.

5. **Streamlit App - Model Performance Tab**:
   - Displays a tabular dataset of metrics (Precision, Recall, F1) for all 6 categories, and a multi-bar chart comparing them side-by-side using red, purple, and green bars.

---

## 🎓 Viva Questions and Answers (Project Prep)

Here are the most common questions asked during academic project presentations (Vivas) for this project:

#### Q1: What is the main objective of the AI-Based Resume Screening System?
**A:** The objective is to automate the initial phase of recruitment by parsing resumes, predicting the most appropriate job category (from a list of target roles) using a trained Machine Learning model, and extracting key technical skills using NLP.

#### Q2: What NLP text preprocessing steps are performed on the resume text, and why?
**A:** The preprocessing steps include:
- **Lowercasing**: To ensure uniformity (e.g., "Python" and "python" are treated identically).
- **Removing URLs/Emails/Tags**: To discard hyperlinks, email addresses, and metadata that do not contribute to classification.
- **Removing Punctuation & Numbers**: To eliminate noise and special characters that are not relevant to semantic content.
- **Tokenization & Stopword Removal**: Breaking text into individual words and removing common connector words (like 'and', 'the', 'is') using the NLTK library so the model only focuses on meaningful keywords and skills.

#### Q3: What is TF-IDF and how does it work?
**A:** TF-IDF stands for **Term Frequency-Inverse Document Frequency**. It is a numerical statistic used to reflect how important a word is to a document in a collection or corpus.
- **Term Frequency (TF)** measures how frequently a term occurs in a document.
- **Inverse Document Frequency (IDF)** measures how important a term is across the entire corpus by reducing the weight of terms that appear very frequently in all documents (like "resume" or "project") and increasing the weight of unique terms (like "scikit-learn" or "onboarding").

#### Q4: Why did you choose Logistic Regression instead of other algorithms?
**A:** Logistic Regression is simple, highly interpretable, and performs exceptionally well on high-dimensional sparse text vectors (TF-IDF features). Furthermore, it provides well-calibrated class probability scores (`predict_proba`) which we use directly to compute the candidate's "Profile Match Strength / Score".

#### Q5: How does the system extract technical skills from the resume?
**A:** We use a rule-based matching mechanism utilizing regular expressions. The script matches terms from a curated skills dictionary against the resume text. Crucially, it uses regex word boundaries (`\b` patterns) to prevent false positives (e.g. preventing "Java" from matching inside "JavaScript" or "Go" matching inside "Google").

#### Q6: How are PDF files parsed in Python?
**A:** We use the `pypdf` library. It reads the uploaded file stream, instantiates a `PdfReader` object, iterates over each page of the document, and extracts the text content using `page.extract_text()`, returning a single concatenated string.

#### Q7: What is the significance of the train_test_split?
**A:** We split the dataset into training (80%) and testing (20%) sets to validate the model's accuracy on unseen data. Using test data prevents overfitting and guarantees that our model is learning general patterns rather than just memorizing the training records. We also use `stratify=y` to ensure categories are equally balanced in both splits.

#### Q8: What are Precision, Recall, and F1-Score?
**A:**
- **Precision**: Out of all resumes predicted for a category, how many were actually correct? (Minimizes false positives).
- **Recall**: Out of all actual resumes belonging to a category, how many did the model find? (Minimizes false negatives).
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single metric representing balance.

#### Q9: What is Streamlit and why was it chosen for this project?
**A:** Streamlit is an open-source Python framework used to create interactive web apps for machine learning and data science projects quickly. It allows developers to build visual interfaces using pure Python without needing front-end web development skills (like HTML, CSS, or JavaScript).

#### Q10: How can this system be enhanced or scaled for production?
**A:** Future enhancements include:
1. **Named Entity Recognition (NER)** using SpaCy to extract candidate names, contact numbers, organizations, and years of experience automatically.
2. **Semantic Similarity Matching** (using transformers like BERT or Sentence-BERT) to match resumes directly against a job description instead of just classifying them into generic categories.
3. **OCR (Optical Character Recognition)** support using Tesseract for scanned/image-based resumes.
