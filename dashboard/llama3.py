import os
import json
import jsonlines
import pandas as pd
import requests
import tempfile
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import get_peft_model, LoraConfig, TaskType
import torch

#  Web Scraping & Cleaning Data

urls = [
    "https://www.tutorialspoint.com/",
    "https://www.geeksforgeeks.org/python/",
    "https://www.geeksforgeeks.org/ai/",
    "https://www.geeksforgeeks.org/dbms/",
    "https://www.geeksforgeeks.org/ml/"
    "https://www.tutorialspoint.com/python",
    "https://www.tutorialspoint.com/ai",
    "https://www.tutorialspoint.com/ml",
    "https://www.tutorialspoint.com/dbms",
    "https://www.w3schools.com/python",
    "https://www.w3schools.com/ai",
    "https://www.w3schools.com/ml",
    "https://www.w3schools.com/dbms",
    "https://www.javatpoint.com/python",
    "https://www.javatpoint.com/ai",
    "https://www.javatpoint.com/dbms",
    "https://www.javatpoint.com/ml",
]

def scrape_and_clean(urls):
    all_data = []
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            paragraphs = soup.find_all("p")
            text_data = [para.get_text().strip() for para in paragraphs if para.get_text().strip()]
            all_data.extend(text_data)
        except requests.RequestException as e:
            print(f"Error scraping {url}: {e}")

    return all_data

scraped_data = scrape_and_clean(urls)

# Save cleaned scraped data to JSONL
scraped_jsonl_file = "scraped_data.jsonl"
with jsonlines.open(scraped_jsonl_file, mode='w') as writer:
    for text in scraped_data:
        writer.write({"text": text})


#  Convert Kaggle CSV to JSONL

def convert_csv_to_jsonl(csv_file, jsonl_file):
    df = pd.read_csv(csv_file)
    
    # Ensure the CSV has a 'text' column
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column.")

    with jsonlines.open(jsonl_file, mode='w') as writer:
        for _, row in df.iterrows():
            writer.write({"text": row["text"]})

# Convert Kaggle dataset
csv_file = "kaggle_dataset.csv"  # Update with actual file path
jsonl_file = "kaggle_dataset.jsonl"
convert_csv_to_jsonl(csv_file, jsonl_file)







#  Load & Combine Datasets

dataset1 = load_dataset("json", data_files={"train": jsonl_file})["train"]
dataset2 = load_dataset("json", data_files={"train": scraped_jsonl_file})["train"]

# Combine datasets
full_dataset = Dataset.from_dict({
    "text": dataset1["text"] + dataset2["text"]
})


#  Fine-Tune LLaMA 2 with LoRA

model_name = "meta-llama/Llama-2-13b"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# LoRA Configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, 
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Tokenization
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=2048)

encoded_dataset = full_dataset.map(preprocess_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama_lora",
    evaluation_strategy="steps",
    logging_dir="./logs",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
    remove_unused_columns=False,
    fp16=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset
)

# Train the Model
trainer.train()

# Save Model
model.save_pretrained("./fine_tuned_llama_lora")
tokenizer.save_pretrained("./fine_tuned_llama_lora")


#  Text Generation

def generate_learning_content(subject, level, content_no):
  
    # Format input
    input_text = f"Generate structured educational content for {subject} at {level} level."

    # Load fine-tuned model
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

    # Generate text
    generated_text = generator(input_text, max_length=50000, num_return_sequences=1)[0]["generated_text"]

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    file_name = f"{subject}-{level}-{content_no}.txt"
    file_path = os.path.join(temp_dir, file_name)

    with open(file_path, "w") as file:
        file.write(generated_text)

    print(f"Content saved as {file_path}")
    return file_path

