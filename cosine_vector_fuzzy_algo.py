import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process
import os
import re
import time

# Load the dataset
df = pd.read_csv('medicine_dataset.csv', low_memory=False)

# Feedback file to store user feedback
FEEDBACK_FILE = "feedback.csv"


# Function to ensure feedback file exists
def initialize_feedback_file():
    if not os.path.exists(FEEDBACK_FILE):
        pd.DataFrame(columns=["drug_name", "selected_drug", "feedback_type", "comments"]).to_csv(FEEDBACK_FILE,
                                                                                                 index=False)


# Preprocessing function to clean drug names and user input
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # Remove special characters
    return text


# Function to find the closest matches for a drug name using cosine similarity
def find_closest_drugs(user_input, limit=5, threshold=0.1):
    # Initialize TF-IDF Vectorizer with unigrams only (1-grams)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1))  # Use unigrams only

    # Preprocess drug names and fit-transform them into vectors
    drug_names = df['name'].dropna().unique()
    processed_drug_names = [preprocess(name) for name in drug_names]
    drug_vectors = vectorizer.fit_transform(processed_drug_names)

    # Preprocess user input and transform it into a vector
    processed_user_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_user_input])

    # Calculate cosine similarity between user input and all drug names
    cosine_similarities = cosine_similarity(user_vector, drug_vectors).flatten()

    # Get the indices of the most similar drugs, only if the similarity is above the threshold
    similar_indices = [i for i in range(len(cosine_similarities)) if cosine_similarities[i] > threshold]

    if not similar_indices:
        return []

    # Sort drugs by similarity
    similar_indices.sort(key=lambda i: cosine_similarities[i], reverse=True)

    # Return the closest drugs based on similarity score
    closest_drugs = [drug_names[i] for i in similar_indices[:limit]]
    return closest_drugs


# Fuzzy Matching Function to handle partial matches
def fuzzy_find_closest_drugs(user_input, limit=5):
    drug_names = df['name'].dropna().unique()

    # Use fuzzy matching to get closest matches
    matches = process.extract(user_input, drug_names, scorer=fuzz.partial_ratio, limit=limit)
    return [match[0] for match in matches]


# Function to get drug usage
def get_usage(drug_name):
    drug_info = df[df['name'].str.lower() == drug_name.lower()]
    if not drug_info.empty:
        return [drug_info[f'use{i}'].values[0] for i in range(5) if pd.notna(drug_info[f'use{i}'].values[0])]
    return ["Usage not available."]


# Function to get substitutes
def get_substitutes(drug_name):
    drug_info = df[df['name'].str.lower() == drug_name.lower()]
    if not drug_info.empty:
        return [drug_info[f'substitute{i}'].values[0] for i in range(5) if
                pd.notna(drug_info[f'substitute{i}'].values[0])]
    return ["Substitutes not available."]


# Function to get side effects
def get_side_effects(drug_name):
    drug_info = df[df['name'].str.lower() == drug_name.lower()]
    if not drug_info.empty:
        return [drug_info[f'sideEffect{i}'].values[0] for i in range(10) if
                pd.notna(drug_info[f'sideEffect{i}'].values[0])]
    return ["Side effects not available."]


# Function to collect user feedback
def collect_feedback(drug_name, selected_drug):
    print("\nWe'd love your feedback to improve the results!")
    print("1. Correct Information")
    print("2. Incorrect Information")
    print("3. Missing Information")
    print("4. Other")
    try:
        feedback_type = int(input("Enter the number corresponding to your feedback: ").strip())
        if feedback_type not in [1, 2, 3, 4]:
            print("Invalid input. Feedback skipped.")
            return

        comments = input("Please enter your comments or corrections (optional): ").strip()

        # Save feedback to the file
        feedback_data = pd.DataFrame([{
            "drug_name": drug_name,
            "selected_drug": selected_drug,
            "feedback_type": feedback_type,
            "comments": comments
        }])
        feedback_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
        print("Thank you for your feedback! 🙏")
    except ValueError:
        print("Invalid input. Feedback skipped.")


# Chatbot main conversation
def chatbot_conversation():
    print("\n👋 Welcome to the AI Drug Assistant! How can I help you today?")
    print("You can type 'exit' to end the chat at any time.")

    while True:
        # User Input
        user_input = input("\nPlease enter the drug name or part of it: ").strip()
        if user_input.lower() == 'exit':
            print("👋 Thank you for using the AI Drug Assistant. Have a great day!")
            break

        # Find closest matching drugs using both cosine similarity and fuzzy matching
        closest_drugs_cosine = find_closest_drugs(user_input)
        closest_drugs_fuzzy = fuzzy_find_closest_drugs(user_input)

        # Combine both lists
        closest_drugs = list(set(closest_drugs_cosine + closest_drugs_fuzzy))

        if not closest_drugs:
            print("⚠️ No matching drugs found. Please try again with another name.")
            continue

        # Display options with the closest drugs
        print("\nDid you mean one of these drugs?")
        for idx, drug in enumerate(closest_drugs, start=1):
            print(f"{idx}. {drug}")

        # Adding a slight delay to slow down output
        time.sleep(1)  # Adds a 1-second delay (you can adjust this time)

        try:
            choice = int(input("\nEnter the number corresponding to the correct drug: ").strip())
            if 1 <= choice <= len(closest_drugs):
                selected_drug = closest_drugs[choice - 1]
            else:
                print("⚠️ Invalid choice. Let's try again.")
                continue
        except ValueError:
            print("⚠️ Invalid input. Let's try again.")
            continue

        # Provide details about the selected drug
        print(f"\n--- Details for {selected_drug} ---")
        print("\n💊 Usage:")
        for use in get_usage(selected_drug):
            print(f"- {use}")

        # Adding a delay to allow users to read the details
        time.sleep(2)  # Adds a 2-second delay for each set of details (can adjust this time)

        print("\n🔄 Substitutes:")
        for sub in get_substitutes(selected_drug):
            print(f"- {sub}")

        # Adding a delay to allow users to read the substitutes
        time.sleep(2)

        print("\n⚠️ Side Effects:")
        for effect in get_side_effects(selected_drug):
            print(f"- {effect}")

        # Adding a delay to allow users to read the side effects
        time.sleep(2)

        # Collect Feedback
        collect_feedback(user_input, selected_drug)
        print("\n🔍 You can ask about another drug or type 'exit' to quit.")


# Initialize feedback file and run chatbot
if __name__ == "__main__":
    initialize_feedback_file()
    chatbot_conversation()
