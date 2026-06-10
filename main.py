import os
import pandas as pd
from generate_dataset import create_dataset_csv
from model_training import train_model
from resume_predictor import ResumePredictor
from visualization import (
    plot_category_distribution,
    plot_most_common_skills,
    plot_model_performance
)

# Configuration Paths
DATASET_PATH = "data/resume_dataset.csv"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

def print_separator():
    print("=" * 60)

def print_header(title):
    print_separator()
    print(f" {title.center(58)} ")
    print_separator()

def main():
    print_header("AI-BASED RESUME SCREENING SYSTEM")
    
    # Step 1: Dataset Generation
    if not os.path.exists(DATASET_PATH):
        print(f"[*] Dataset not found at '{DATASET_PATH}'. Generating a new sample dataset...")
        create_dataset_csv(DATASET_PATH, count_per_category=30)
    else:
        print(f"[+] Found existing dataset at '{DATASET_PATH}'")
        
    # Read the dataset for visualization and info
    df = pd.read_csv(DATASET_PATH)
    print(f"[i] Dataset contains {len(df)} resumes across {df['Category'].nunique()} categories.")
    print_separator()
    
    # Step 2: Model Training & Evaluation
    print("[*] Starting model training and evaluation...")
    accuracy, report_dict = train_model(DATASET_PATH, MODELS_DIR)
    print_separator()
    print(f"[+] Model trained successfully! Overall Accuracy: {accuracy * 100:.2f}%")
    print_separator()
    
    # Step 3: Initialize Predictor
    print("[*] Loading trained model artifacts...")
    predictor = ResumePredictor(MODELS_DIR)
    
    # Step 4: Generate Visualizations
    print("[*] Generating visualization plots...")
    plot_category_distribution(df, PLOTS_DIR)
    plot_most_common_skills(df, predictor, PLOTS_DIR)
    plot_model_performance(report_dict, PLOTS_DIR)
    print(f"[+] All charts saved successfully under the '{PLOTS_DIR}/' directory.")
    print_separator()
    
    # Preloaded sample resumes for quick testing
    sample_resumes = {
        "1": {
            "name": "Python/Data Science Resume",
            "text": """
            RAHUL SHARMA - MACHINE LEARNING ENGINEER
            Email: rahul.sharma@example.com | Phone: +91 99999-12345
            
            SUMMARY:
            Experienced Data Scientist and Python Developer with 2+ years of experience 
            building predictive models and data pipelines. Passionate about machine learning, 
            NLP, and data analysis.
            
            TECHNICAL SKILLS:
            - Programming: Python, R, SQL
            - Data Science: Pandas, NumPy, Scikit-Learn, TensorFlow, Keras, PyTorch
            - Tools: Git, Jupyter Notebook, VS Code, Docker, AWS
            
            WORK EXPERIENCE:
            Associate Data Scientist at Google (June 2024 - Present)
            - Developed customer churn predictive models using Random Forest and XGBoost.
            - Cleaned and preprocessed large-scale datasets using pandas and numpy.
            - Designed interactive Tableau dashboards for business intelligence.
            """
        },
        "2": {
            "name": "Human Resources (HR) Resume",
            "text": """
            SNEHA JOSHI - HR MANAGER
            Email: sneha.joshi@example.com | Phone: +91 98888-54321
            
            SUMMARY:
            Dynamic Human Resources professional with expertise in Talent Acquisition, 
            Employee Relations, Onboarding, and Performance Management. Skilled in developing 
            HR policies and resolving workplace conflicts.
            
            KEY SKILLS:
            - HR Operations: Recruitment, Screening, Onboarding, Payroll Management, exit interviews
            - HR Tools: HRIS, Workday, Microsoft Office Suite
            - Interpersonal: Conflict Resolution, Team Leadership, Employee Engagement, Communication
            
            EXPERIENCE:
            HR Specialist at TCS (Jan 2024 - Present)
            - Managed end-to-end recruitment drives, hiring over 50 candidates in 6 months.
            - Conducted onboarding, training, and performance reviews.
            - Updated the employee handbook with current compliance policies.
            """
        }
    }
    
    # Step 5: Interactive Predictor Loop
    print_header("RESUME PREDICTION INTERACTIVE TEST")
    print("Test prediction using preloaded resumes or enter your own text.")
    print("Options:")
    print("1 - Test with Sample Data Science Resume")
    print("2 - Test with Sample HR Resume")
    print("3 - Paste your custom Resume Text")
    print("4 - Exit")
    print_separator()
    
    while True:
        choice = input("Select an option (1-4): ").strip()
        
        if choice == "4":
            print("\nExiting. Thank you for using the AI Resume Screening System!")
            break
            
        resume_text = ""
        resume_name = "Custom Resume"
        
        if choice in sample_resumes:
            resume_text = sample_resumes[choice]["text"]
            resume_name = sample_resumes[choice]["name"]
            print(f"\nEvaluating: {resume_name}")
            print("-" * 40)
            print(resume_text.strip()[:350] + "\n...[truncated]...")
            print("-" * 40)
        elif choice == "3":
            print("\nPaste your resume text below. When finished, press Enter on an empty line or Ctrl+Z (Windows) then Enter:")
            lines = []
            while True:
                try:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
                except EOFError:
                    break
            resume_text = "\n".join(lines)
            if not resume_text.strip():
                print("[-] Empty input. Try again.")
                continue
        else:
            print("[-] Invalid choice. Please select 1, 2, 3, or 4.")
            continue
            
        # Perform prediction
        category, score = predictor.predict(resume_text)
        skills_info = predictor.extract_skills(resume_text)
        
        # Output results
        print("\n" + "*" * 50)
        print(f" SCREENING RESULT FOR: {resume_name.upper()} ")
        print("*" * 50)
        print(f"-> Predicted Suitable Category : {category}")
        print(f"-> Role Match/Confidence Score  : {score * 100:.2f}%")
        print("\n-> Extracted Skills Detected:")
        
        if skills_info["all_skills"]:
            # Print sorted list of skills
            print(f"   {', '.join(sorted(skills_info['all_skills']))}")
            print("\n-> Skills Grouped By Job Domain:")
            for cat, cat_skills in skills_info["by_category"].items():
                print(f"   - {cat}: {', '.join(cat_skills)}")
        else:
            print("   No technical skills matching our dictionary were found.")
        print("*" * 50 + "\n")

if __name__ == "__main__":
    main()
