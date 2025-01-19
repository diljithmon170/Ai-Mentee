from transformers import AutoTokenizer, AutoModelForCausalLM


#Load the Pre-trained Model: Load the LLaMA 2-7B model and tokenizer from the Hugging Face Model Hub.
# Load the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")


#Prepare the Dataset: You need to have a dataset that aligns with your specific fine-tuning task (e.g., text generation, classification). You can use the datasets library to load an existing dataset or prepare your custom dataset.
from datasets import load_dataset

# Load a dataset (example: wikitext)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True)
#Tokenize the Dataset: Use the tokenizer to prepare the data.
# Tokenize the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)
from transformers import DataCollatorForSeq2Seq


#Setup Training Arguments: Define training arguments such as learning rate, number of epochs, batch size, etc. This is where you can adjust various parameters.
# Use a data collator for sequence-to-sequence tasks
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",         # Output directory for saved model
    evaluation_strategy="epoch",    # Evaluation strategy (e.g., after each epoch)
    learning_rate=2e-5,             # Learning rate
    per_device_train_batch_size=2,  # Batch size
    per_device_eval_batch_size=2,   # Evaluation batch size
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Weight decay
    save_total_limit=2,             # Limit the total number of saved checkpoints
    logging_dir="./logs",           # Directory for logs
    logging_steps=500,              # Log every 500 steps
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("./fine_tuned_llama")
tokenizer.save_pretrained("./fine_tuned_llama")

trainer.evaluate()
