from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_model_and_data():
    model_name = "EleutherAI/gpt-neo-125M"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # ✅ FIX

    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    print(f"Sample Data: \n{dataset[0]['text'][:200]}...")

    print("Tokenizing dataset...")
    def tokenize_fn(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_fn, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")

    print("✅ Model and dataset loaded and tokenized successfully!")
    return model, tokenizer, tokenized_dataset

if __name__ == "__main__":
    load_model_and_data()