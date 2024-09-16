import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Suppress warnings about mixed data types (temporary solution)
warnings.filterwarnings("ignore")

# Load the dataset (replace with your actual file path)
data = pd.read_csv('medicines.csv', low_memory=False)  # Handle potential memory issues

# Combine substitute and side effect information (handle missing values)
data['combined_text'] = data[['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4', 'sideEffect0', 'sideEffect1', 'sideEffect2']].fillna('').apply(lambda x: ' '.join(x), axis=1)

# Create TF-IDF vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

# Recommendation function
def recommend(medicine_name):
    query_vector = vectorizer.transform([medicine_name])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    top_indices = cosine_similarities.argsort()[-5:][::-1]  # Recommend top 5
    recommended_medicines = data.iloc[top_indices]['name'].tolist()
    return recommended_medicines

# Example usage
medicine_to_recommend = 'augmentin 625 duo tablet'
recommendations = recommend(medicine_to_recommend)
print(f"Recommended medicines for '{medicine_to_recommend}':")
print(recommendations)