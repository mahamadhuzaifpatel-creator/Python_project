import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Automatically download NLTK datasets if they are not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def clean_resume_text(text: str) -> str:
    """
    Cleans the input resume text by:
    1. Converting to lowercase
    2. Removing URLs, emails, hashtags, and mentions
    3. Removing punctuation and special characters
    4. Removing extra whitespaces
    5. Removing NLTK English stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'www\.\S+\s*', ' ', text)
    
    # 3. Remove Email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # 4. Remove RTF, HTML-like tags, and special characters
    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'rtf[a-zA-Z0-9]*', ' ', text)
    
    # 5. Remove numbers, hashtags, mentions, and symbols
    text = re.sub(r'#\S+', ' ', text)  # remove hashtags
    text = re.sub(r'@\S+', ' ', text)  # remove mentions
    
    # Remove punctuation
    # We replace punctuation characters with spaces
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Remove digits/numbers
    text = re.sub(r'\d+', ' ', text)
    
    # 6. Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    
    cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    
    # Reconstruct the cleaned string
    cleaned_text = ' '.join(cleaned_tokens)
    
    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

if __name__ == "__main__":
    # Test preprocessing
    sample_text = "Hello! My email is test@example.com and portfolio is http://portfolio.com. I know Python, ML, and Java."
    print("Original Text:")
    print(sample_text)
    print("\nCleaned Text:")
    print(clean_resume_text(sample_text))
