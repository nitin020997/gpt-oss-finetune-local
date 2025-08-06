from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the fine-tuned model and tokenizer
model_path = "./finetuned_gptneo"

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Define your input prompt
prompt = "In a DevOps workflow, the purpose of CI/CD is to"

# Tokenize the input with attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

# Generate output
with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=False,  # no randomness
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and print
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n🧠 Generated Output:\n")
print(generated_text)