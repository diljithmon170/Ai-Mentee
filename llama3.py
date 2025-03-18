import os
import tempfile
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline

# List of URLs to scrape
urls = [
    "https://www.w3schools.com/",
    "https://www.javatpoint.com/",
    "https://www.tutorialspoint.com/",
    "https://www.freecodecamp.org/",
    "https://www.kaggle.com/learn/intro-to-machine-learning",
    "https://www.geeksforgeeks.org/python/",
    "https://www.geeksforgeeks.org/java/",
    "https://www.geeksforgeeks.org/dbms/",
    "https://www.geeksforgeeks.org/ml/",
    "https://www.codecademy.com/learn/learn-python/",
    "https://www.codecademy.com/learn/learn-java/"
]

# Function to scrape data from a URL using BeautifulSoup
def scrape_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text_data = [para.get_text() for para in paragraphs]
        return text_data
    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return []

# Scrape data from all URLs
all_text_data = []
for url in urls:
    all_text_data.extend(scrape_url(url))

# Save the scraped data to a DataFrame
df_scraped = pd.DataFrame(all_text_data, columns=["text"])

df_scraped.to_csv("scraped_data.csv", index=False)

# Load datasets
dataset1 = load_dataset('text', data_files={'train': 'dataset1.txt'})
dataset2 = load_dataset('text', data_files={'train': 'scraped_data.csv'})

# Combine datasets
dataset_combined = dataset1["train"].concatenate(dataset2["train"])

# Load LLaMA 3 model and tokenizer
model_name = "meta-llama/Llama-3-8b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=2048)

# Encode dataset
encoded_dataset = dataset_combined.map(preprocess_function, batched=True)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama3",
    evaluation_strategy="steps",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=False,
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llama3")
tokenizer.save_pretrained("./fine_tuned_llama3")

# Function to generate structured learning lessons
def generate_learning_lesson(subject, level, topic):
    print("Welcome to AI MENTEE - Your AI Tutor!")
    
    # Format prompt for lesson generation
    prompt = f"Generate a detailed learning lesson on {topic} for {level} level students in {subject}. Include explanations, examples, and exercises."
    
    # Initialize text generation pipeline
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    
    # Generate lesson
    generated_text = generator(prompt, max_length=2048, num_return_sequences=1)[0]["generated_text"]
    
    # Save the lesson to a file
    filename = f"{subject}_{level}_{topic}.txt"
    with open(filename, "w") as file:
        file.write(generated_text)
    
    print(f"Lesson saved as {filename}")
    return filename

