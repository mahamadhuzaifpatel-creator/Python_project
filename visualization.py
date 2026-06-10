import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from resume_predictor import ResumePredictor

# Set a beautiful visual style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Custom premium palette
COLOR_PALETTE = ["#2B2D42", "#8D99AE", "#EF233C", "#D90429", "#4A4E69", "#9A8C98"]

def plot_category_distribution(df: pd.DataFrame, output_dir: str = "plots") -> str:
    """Generates and saves a bar chart showing the distribution of job categories."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "category_distribution.png")
    
    plt.figure(figsize=(10, 6))
    
    # Calculate counts and sort
    category_counts = df["Category"].value_counts().reset_index()
    category_counts.columns = ["Category", "Count"]
    
    # Draw horizontal bar chart for clean text display
    sns.barplot(
        x="Count", 
        y="Category", 
        data=category_counts, 
        palette="crest", 
        hue="Category",
        legend=False
    )
    
    plt.title("Distribution of Resumes by Job Category", pad=20, fontweight="bold")
    plt.xlabel("Number of Resumes")
    plt.ylabel("Job Category")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Category distribution plot saved to: {filepath}")
    return filepath

def plot_most_common_skills(df: pd.DataFrame, predictor: ResumePredictor, output_dir: str = "plots") -> str:
    """Extracts skills from all resumes, counts frequencies, and plots the top 15 skills."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "common_skills.png")
    
    print("Extracting skills from dataset for visualization...")
    all_skills = []
    for resume_text in df["Resume"]:
        skills_info = predictor.extract_skills(resume_text)
        all_skills.extend(skills_info["all_skills"])
        
    skill_counts = Counter(all_skills)
    top_skills = pd.DataFrame(skill_counts.most_common(15), columns=["Skill", "Frequency"])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Frequency", 
        y="Skill", 
        data=top_skills, 
        palette="viridis", 
        hue="Skill",
        legend=False
    )
    
    plt.title("Top 15 Most Common Skills in Dataset", pad=20, fontweight="bold")
    plt.xlabel("Frequency (Mentions across Resumes)")
    plt.ylabel("Skill / Keyword")
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Common skills plot saved to: {filepath}")
    return filepath

def plot_model_performance(report_dict: dict, output_dir: str = "plots") -> str:
    """Plots evaluation metrics (precision, recall, f1-score) per job category."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "model_accuracy.png")
    
    # Format the classification report dictionary for plotting
    data = []
    for class_name, metrics in report_dict.items():
        if class_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        data.append({
            "Category": class_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-Score": metrics["f1-score"]
        })
        
    df_metrics = pd.DataFrame(data)
    
    # Melt dataframe for easy plotting with Seaborn
    df_melted = pd.melt(
        df_metrics, 
        id_vars="Category", 
        value_vars=["Precision", "Recall", "F1-Score"],
        var_name="Metric", 
        value_name="Score"
    )
    
    plt.figure(figsize=(11, 6))
    sns.barplot(
        x="Category", 
        y="Score", 
        hue="Metric", 
        data=df_melted, 
        palette="muted"
    )
    
    plt.title("Model Classification Metrics by Job Category", pad=20, fontweight="bold")
    plt.xlabel("Job Category")
    plt.ylabel("Score (0.0 - 1.0)")
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Model performance plot saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    # Test visualizations
    # Note: Requires dataset and trained models.
    try:
        df = pd.read_csv("data/resume_dataset.csv")
        pred = ResumePredictor()
        
        # Test report dictionary structure
        test_report = {
            "Data Science": {"precision": 0.95, "recall": 0.90, "f1-score": 0.92},
            "Web Development": {"precision": 0.88, "recall": 0.92, "f1-score": 0.90},
            "Java Developer": {"precision": 0.91, "recall": 0.89, "f1-score": 0.90},
            "Python Developer": {"precision": 0.85, "recall": 0.88, "f1-score": 0.86},
            "Testing": {"precision": 0.96, "recall": 0.95, "f1-score": 0.95},
            "HR": {"precision": 1.00, "recall": 1.00, "f1-score": 1.00},
            "accuracy": 0.92
        }
        
        plot_category_distribution(df)
        plot_most_common_skills(df, pred)
        plot_model_performance(test_report)
    except Exception as e:
        print(e)
