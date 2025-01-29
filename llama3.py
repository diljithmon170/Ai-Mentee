import time
time.sleep(time.sleep(1.5 * 60 * 60))









import tempfile
import pandas as pd
from datasets import load_dataset
from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments, pipeline
import requests
from bs4 import BeautifulSoup
import os

# List of URLs to scrape
urls = [
    "https://www.w3schools.com/",
    "https://www.javatpoint.com/",
    "https://www.tutorialspoint.com/",
    "https://www.freecodecamp.org/",
    "https://www.coursebox.ai/"
    
]

# Function to scrape data from a URL 
#using beautifulsoup
def scrape_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    
    
    paragraphs = soup.find_all()
    text_data = [para.get_text() for para in paragraphs]
    return text_data

# Scrape data from all URLs
all_text_data = []
for url in urls:
    text_data = scrape_url(url)
    all_text_data.extend(text_data) 

# Save the scraped data to a DataFrame
df_scraped = pd.DataFrame(all_text_data, columns=["text"])

# Save the scraped data to a CSV file
df_scraped.to_csv("scraped_data.csv", index=False)

# Load the Kaggle dataset
dataset1 = pd.read_csv("dataset1.csv")  # Replace with your actual Kaggle dataset path

# Convert both CSV datasets to .txt files for use in Hugging Face Trainer
dataset1['text'].to_csv('dataset1.txt', index=False, header=False)
df_scraped['text'].to_csv('dataset2.txt', index=False, header=False)

# Load the datasets as text format
dataset1 = load_dataset('text', data_files={'train': 'dataset1.txt'})
dataset2 = load_dataset('text', data_files={'train': 'dataset2.txt'})

# Combine the datasets
dataset_combined = dataset1['train'].concatenate(dataset2['train'])

# Load pretrained LLaMA 2-7B model and tokenizer
model_name = "meta-llama/Llama-2-7b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# Move the model to CPU
model.to('cpu')

# Pre-processing the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=2048)

# Apply preprocessing to the combined dataset
encoded_dataset = dataset_combined.map(preprocess_function, batched=True)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",  # Output directory for the fine-tuned model
    evaluation_strategy="steps",
    logging_dir="./logs",             # Log directory
    per_device_train_batch_size=2,    # Smaller batch size for CPU
    per_device_eval_batch_size=2,     # Smaller batch size for CPU
    num_train_epochs=3,               # Number of epochs for fine-tuning
    save_steps=500,                   # Save model every 500 steps
    logging_steps=100,                # Log every 100 steps
    save_total_limit=2,               # Keep only the 2 most recent saved models
    remove_unused_columns=False,      # Keep unused columns in the dataset
    fp16=False,                       # Do not use FP16 precision (not needed on CPU)
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,  # Use the combined dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")





def generate_text(subject, level, topic):
    
    
    print("Welcome to AI MENTEE- your tutor!")
     
    
    subject = input("\nEnter the subject (Python, Java, DBMS, AI): ").strip().lower()
    
    
    difficulty = input("\n\nEnter the difficulty level (Beginner, Intermediate, Advanced): ").strip().lower()
    
    
    
    # Using the fine-tuned model for text generation
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 for CPU

    # Generate text with the fine-tuned model
    generated_text = generator(subject,difficulty,num_return_sequences=1)[0]["generated_text"]
    # Dynamically create the filename
    filename = f"{subject}-{level}-{topic}.txt"
    
    # Create a temporary file to save the generated text with dynamic name
    with open(filename, "w") as temp_file:
        temp_file.write(generated_text)

    os.remove(filename)



generate_text(subject, level, topic)

        
        







