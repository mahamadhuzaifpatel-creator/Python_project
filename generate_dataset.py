import os
import random
import pandas as pd

# Define categories
CATEGORIES = [
    "Data Science",
    "Web Development",
    "Java Developer",
    "Python Developer",
    "Testing",
    "HR"
]

# Vocabulary lists to construct random realistic resumes
NAMES = ["Amit Sharma", "Priya Patel", "Rajesh Kumar", "Anjali Singh", "Sandeep Verma", 
         "Neha Gupta", "Vikram Malhotra", "Kiran Rao", "Rahul Nair", "Sneha Joshi",
         "John Doe", "Jane Smith", "Michael Johnson", "Emily Davis", "David Miller"]

UNIVERSITIES = ["IIT Bombay", "IIT Delhi", "BITS Pilani", "Delhi University", "VIT Vellore", 
                "Stanford University", "MIT", "UC Berkeley", "Carnegie Mellon", "State University"]

COMPANIES = ["Google", "Microsoft", "TCS", "Infosys", "Wipro", "Accenture", "Amazon", 
             "Cognizant", "Capgemini", "Tech Mahindra", "Meta", "Netflix"]

# Keywords by category
SKILLS_POOL = {
    "Data Science": [
        "Python", "R", "SQL", "Machine Learning", "Deep Learning", "Pandas", "NumPy", 
        "Scikit-Learn", "TensorFlow", "Keras", "PyTorch", "Tableau", "Power BI", 
        "Natural Language Processing", "NLP", "Computer Vision", "Data Visualization", 
        "Statistics", "Linear Regression", "Random Forest", "XGBoost", "Data Analysis",
        "Jupyter", "Hadoop", "Spark"
    ],
    "Web Development": [
        "HTML5", "CSS3", "JavaScript", "React.js", "Angular", "Vue.js", "Node.js", 
        "Express.js", "MongoDB", "TypeScript", "Bootstrap", "Tailwind CSS", "Git", "GitHub", 
        "RESTful APIs", "Webpack", "Redux", "SQL", "Front-end Development", 
        "Back-end Development", "Full Stack Development", "Web Services", "JSON"
    ],
    "Java Developer": [
        "Java", "J2EE", "Spring", "Spring Boot", "Hibernate", "Microservices", "REST APIs", 
        "Maven", "Gradle", "SQL", "Oracle", "MySQL", "JPA", "Git", "Multithreading", 
        "Data Structures", "Algorithms", "JUnit", "Mockito", "Apache Tomcat", "Docker",
        "Design Patterns", "OOPs"
    ],
    "Python Developer": [
        "Python", "Django", "Flask", "FastAPI", "PostgreSQL", "MySQL", "SQLite", "Git", 
        "Docker", "AWS", "Celery", "Redis", "RESTful APIs", "OOPs", "Web Scraping", 
        "BeautifulSoup", "Selenium", "Pandas", "NumPy", "Unit Testing", "Pytest", 
        "Linux", "Scripting", "Automation"
    ],
    "Testing": [
        "Manual Testing", "Automation Testing", "Selenium", "Java", "Python", "Test Cases", 
        "Jira", "Bugzilla", "Regression Testing", "Functional Testing", "Postman", 
        "API Testing", "Software Testing Life Cycle", "STLC", "SDLC", "SQL", "Test Plan",
        "Agile Methodology", "LoadRunner", "JMeter", "Test Execution", "Quality Assurance", "QA"
    ],
    "HR": [
        "Recruitment", "Employee Relations", "Onboarding", "Talent Acquisition", 
        "Payroll Management", "HR Policies", "Performance Management", "Employee Engagement", 
        "Training & Development", "Sourcing", "Screening", "Interviewing", "Conflict Resolution", 
        "HRIS", "Workday", "Exit Interviews", "Compliance", "Strategic Planning", 
        "Communication Skills", "HR Operations"
    ]
}

PROJECTS_POOL = {
    "Data Science": [
        "Predictive Modeling for Customer Churn using Random Forest",
        "Natural Language Processing for Sentiment Analysis on Twitter Data",
        "Object Detection System using OpenCV and TensorFlow",
        "Deep Learning Model for Medical Image Classification",
        "Data Visualization Dashboard using Tableau and Python"
    ],
    "Web Development": [
        "E-Commerce Platform using React, Node.js, and MongoDB",
        "Real-Time Chat Application with WebSockets and React",
        "Responsive Portfolio Website using Tailwind CSS and HTML5",
        "Task Management Dashboard with Angular and Firebase",
        "REST API Service using Express.js and PostgreSQL"
    ],
    "Java Developer": [
        "Online Banking System using Spring Boot and Hibernate",
        "Microservices Architecture for Food Delivery Application",
        "Employee Directory API with Spring Data JPA and MySQL",
        "Inventory Management System with Spring Boot and Maven",
        "Multi-threaded Chat Server in Java"
    ],
    "Python Developer": [
        "Web Scraping Tool for Property Rental Data using BeautifulSoup",
        "RESTful API Server for Blogging Platform using Django REST Framework",
        "Background Task Processing Service with FastAPI, Celery, and Redis",
        "Automated PDF Invoice Generator and Emailer",
        "Machine Learning Web App deployed using Flask and AWS"
    ],
    "Testing": [
        "Test Automation Framework for E-Commerce site using Selenium and Java",
        "API Testing Suite for Payment Gateway using Postman and Newman",
        "Performance and Load Testing of Web Portal using JMeter",
        "Mobile App Testing Framework with Appium and Python",
        "Defect Lifecycle Management and Reporting in Jira"
    ],
    "HR": [
        "Campus Recruitment Drive and Hiring Strategy Implementation",
        "Design and Roll-out of Employee Wellness and Engagement Programs",
        "Onboarding Process Redesign for Hybrid Work Environment",
        "Performance Appraisal and Feedback System Implementation",
        "Company Policy Formulation and Employee Handbook Update"
    ]
}

def generate_random_resume(category: str) -> str:
    """Generates a realistic resume text with a randomized list of details and relevant skills."""
    name = random.choice(NAMES)
    email = f"{name.lower().replace(' ', '')}@example.com"
    phone = f"+91 {random.randint(70000, 99999)}-{random.randint(10000, 99999)}"
    uni = random.choice(UNIVERSITIES)
    company1 = random.choice(COMPANIES)
    company2 = random.choice([c for c in COMPANIES if c != company1])
    
    # Select skills
    category_skills = SKILLS_POOL[category]
    num_skills = random.randint(6, 12)
    selected_skills = random.sample(category_skills, k=min(num_skills, len(category_skills)))
    
    # Select projects
    category_projects = PROJECTS_POOL[category]
    num_projects = random.randint(2, 3)
    selected_projects = random.sample(category_projects, k=min(num_projects, len(category_projects)))
    
    # Add a few cross-functional or general skills
    general_skills = ["Git", "Agile", "SQL", "Team Collaboration", "Problem Solving", "Communication"]
    extra_skills = random.sample(general_skills, k=random.randint(2, 4))
    all_skills = list(set(selected_skills + extra_skills))
    
    resume_text = f"""
    {name.upper()}
    Contact: {phone} | Email: {email}
    Address: Bangalore, Karnataka, India
    
    OBJECTIVE
    Detail-oriented and results-driven professional seeking a challenging role as a {category} to utilize my skills and contribute to organizational success.
    
    EDUCATION
    Bachelor of Technology in Computer Science & Engineering
    {uni} - CGPA: {random.uniform(7.5, 9.5):.2f}/10
    
    SUMMARY OF SKILLS
    - Technical Skills: {', '.join(all_skills)}
    - Tools & Technologies: GitHub, Docker, VS Code, Postman
    - Soft Skills: Leadership, Team Player, Analytical Thinking
    
    PROFESSIONAL EXPERIENCE
    Associate Software Engineer | {company1} (June 2024 - Present)
    - Developed and maintained critical software modules under Agile methodologies.
    - Worked closely with senior engineers to implement new product features and resolve bugs.
    - Utilized technologies like {', '.join(random.sample(selected_skills, k=3))} to optimize workflows.
    - Wrote clean, readable, and well-documented code.
    
    Software Engineering Intern | {company2} (Jan 2024 - May 2024)
    - Collaborated with cross-functional teams to design high-quality features.
    - Assisted in deploying applications and writing comprehensive documentation.
    - Hands-on experience with {', '.join(random.sample(selected_skills, k=min(2, len(selected_skills))))}.
    
    PROJECTS
    1. {selected_projects[0]}
       - Description: Developed a solution addressing core domain challenges.
       - Technologies Used: {', '.join(random.sample(selected_skills, k=min(4, len(selected_skills))))}
    2. {selected_projects[1]}
       - Description: Engineered an end-to-end framework, increasing efficiency and responsiveness.
       - Technologies Used: {', '.join(random.sample(selected_skills, k=min(3, len(selected_skills))))}
    
    DECLARATION
    I hereby declare that the information provided above is true to the best of my knowledge.
    """
    return resume_text.strip()

def create_dataset_csv(filepath: str, count_per_category: int = 30):
    """Generates the CSV file with Category and Resume text."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    data = []
    for category in CATEGORIES:
        for _ in range(count_per_category):
            resume = generate_random_resume(category)
            data.append({"Category": category, "Resume": resume})
            
    df = pd.DataFrame(data)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"Dataset successfully created at {filepath} with {len(df)} entries.")

if __name__ == "__main__":
    # Test generation
    dataset_path = os.path.join("data", "resume_dataset.csv")
    create_dataset_csv(dataset_path, count_per_category=30)  # 6 * 30 = 180 entries
