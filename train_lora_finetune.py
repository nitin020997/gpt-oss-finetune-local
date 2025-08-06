from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from load_model_dataset import load_model_and_data

# Load model, tokenizer, and tokenized dataset from previous step
model, tokenizer, tokenized_dataset = load_model_and_data()

# Step 1: Set up LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
)

model = get_peft_model(model, peft_config)

# Step 2: Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_gptneo",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    logging_dir="./logs",
)

# Step 3: Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Step 4: Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Step 5: Train the model
trainer.train()

# Step 6: Save the model
trainer.save_model("./finetuned_gptneo")
tokenizer.save_pretrained("./finetuned_gptneo")

print("🎉 Training complete! Fine-tuned model saved.")