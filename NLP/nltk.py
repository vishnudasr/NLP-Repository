import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary nltk resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Take input from the user
text = input("Enter the text to be processed: ")

# Tokenization
tokens = word_tokenize(text)
print(f"\nTokens: {tokens}")

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in tokens]
print(f"\nStemmed Tokens: {stemmed_tokens}")

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
print(f"\nLemmatized Tokens: {lemmatized_tokens}")

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print(f"\nFiltered Tokens (Stop Words Removed): {filtered_tokens}")