import os
import pickle
import re
from typing import Dict, List, Tuple
from data_preprocessing import clean_resume_text

# Define dictionary of skills by category for keyword matching
SKILLS_DICTIONARY = {
    "Data Science": [
        "python", "r language", "sql", "machine learning", "deep learning", "pandas", "numpy", 
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "tableau", "power bi", 
        "natural language processing", "nlp", "computer vision", "statistics", "spark", "hadoop",
        "data analysis", "data visualization", "predictive modeling"
    ],
    "Web Development": [
        "html", "css", "javascript", "js", "react", "angular", "vue", "node.js", "node",
        "express", "mongodb", "typescript", "bootstrap", "tailwind", "git", "github", 
        "rest api", "restful api", "redux", "web services", "webpack", "ajax"
    ],
    "Java Developer": [
        "java", "j2ee", "spring", "spring boot", "hibernate", "microservices", "rest api", 
        "maven", "gradle", "sql", "oracle", "mysql", "jpa", "multithreading", "junit", "mockito", 
        "tomcat", "docker", "oops", "design patterns"
    ],
    "Python Developer": [
        "python", "django", "flask", "fastapi", "postgresql", "mysql", "sqlite", "git", 
        "docker", "aws", "celery", "redis", "rest api", "web scraping", "beautifulsoup", 
        "selenium", "unittest", "pytest", "scripting", "linux"
    ],
    "Testing": [
        "manual testing", "automation testing", "selenium", "test cases", "jira", 
        "bugzilla", "regression testing", "functional testing", "postman", "api testing", 
        "qa", "quality assurance", "software testing", "jmeter", "loadrunner", "test plan"
    ],
    "HR": [
        "recruitment", "employee relations", "onboarding", "talent acquisition", "payroll", 
        "hr policies", "performance management", "employee engagement", "training", "sourcing", 
        "screening", "interviewing", "conflict resolution", "hris", "workday", "compliance"
    ]
}

class ResumePredictor:
    def __init__(self, models_dir: str = "models"):
        """Loads the trained model and vectorizer from the models directory."""
        self.vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
        self.classifier_path = os.path.join(models_dir, "resume_classifier.pkl")
        
        if not os.path.exists(self.vectorizer_path) or not os.path.exists(self.classifier_path):
            raise FileNotFoundError("Model files not found. Please train the model first by running model_training.py.")
            
        with open(self.vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
            
        with open(self.classifier_path, "rb") as f:
            self.classifier = pickle.load(f)

    def predict(self, resume_text: str) -> Tuple[str, float]:
        """
        Cleans the input text, vectorized it, and predicts the job category 
        along with the matching (confidence) score.
        """
        # Preprocess the resume text
        cleaned_text = clean_resume_text(resume_text)
        
        if not cleaned_text:
            return "Unknown", 0.0
            
        # Vectorize the text
        features = self.vectorizer.transform([cleaned_text])
        
        # Predict the category
        prediction = self.classifier.predict(features)[0]
        
        # Calculate confidence score (probability)
        probabilities = self.classifier.predict_proba(features)[0]
        class_idx = list(self.classifier.classes_).index(prediction)
        confidence = probabilities[class_idx]
        
        return prediction, confidence

    def extract_skills(self, resume_text: str) -> Dict[str, List[str]]:
        """
        Extracts skills from the resume by matching keyword patterns.
        Returns a dictionary showing matching skills found, grouped by category,
        as well as a list of all detected skills.
        """
        # Convert text to lowercase for case-insensitive matching
        lower_text = resume_text.lower()
        
        # Clean special formatting but keep text structure
        # Replace newlines/tabs with spaces
        text_for_matching = re.sub(r'\s+', ' ', lower_text)
        
        detected_by_category = {}
        all_detected = set()
        
        for category, skills in SKILLS_DICTIONARY.items():
            matched_skills = []
            for skill in skills:
                # We use word boundaries to avoid partial matches
                # e.g., 'java' matching in 'javascript' is avoided.
                # Escape the skill for regex safety
                escaped_skill = re.escape(skill)
                
                # Check for special characters like .js, C++, C#
                # If skill ends with special chars, word boundary \b might fail, so handle it
                if skill.endswith(('++', '#', '.js')):
                    pattern = rf'\b{escaped_skill}(?!\w)'
                else:
                    pattern = rf'\b{escaped_skill}\b'
                    
                if re.search(pattern, text_for_matching):
                    # Format skill nicely (capitalize first letter/words)
                    formatted_name = skill.title() if len(skill) > 3 else skill.upper()
                    if skill == "sql": formatted_name = "SQL"
                    if skill == "html": formatted_name = "HTML"
                    if skill == "css": formatted_name = "CSS"
                    if skill == "nlp": formatted_name = "NLP"
                    if skill == "hris": formatted_name = "HRIS"
                    if skill == "js": formatted_name = "JavaScript"
                    if skill == "node.js": formatted_name = "Node.js"
                    if skill == "react.js": formatted_name = "React.js"
                    
                    matched_skills.append(formatted_name)
                    all_detected.add(formatted_name)
                    
            if matched_skills:
                detected_by_category[category] = matched_skills
                
        return {
            "by_category": detected_by_category,
            "all_skills": list(all_detected)
        }

if __name__ == "__main__":
    # Quick testing
    # Note: Requires trained model to run successfully.
    try:
        predictor = ResumePredictor()
        test_resume = """
        John Doe - Data Analyst
        Experienced in Python, SQL, and machine learning models. 
        Developed predictive models using pandas, numpy, and scikit-learn.
        Familiar with Tableau and Git.
        """
        category, score = predictor.predict(test_resume)
        skills = predictor.extract_skills(test_resume)
        
        print(f"Predicted Category: {category}")
        print(f"Confidence: {score:.2%}")
        print(f"Skills Extracted: {skills['all_skills']}")
    except FileNotFoundError as e:
        print(e)
