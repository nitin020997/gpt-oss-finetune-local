# 🧠 GPT-OSS Fine-Tuning (Local) with Hugging Face + LoRA

This project demonstrates how to fine-tune an **open-source GPT model locally** using:
- 🦙 Hugging Face Transformers
- 🪶 PEFT (Parameter-Efficient Fine-Tuning) via **LoRA**
- ✅ CPU / MPS on Mac (No GPU or cloud dependencies)

---

## 💡 Use Case

Fine-tuned GPT-Neo on a lightweight dataset to:
- Generate DevOps-related text (e.g., CI/CD summaries)
- Automate test case drafts
- Build local LLM workflows for project documentation

---

## 📁 Project Structure

| File | Purpose |
|------|---------|
| `load_model_dataset.py` | Loads GPT-Neo and tokenizes dataset (Wikitext) |
| `train_lora_finetune.py` | Fine-tunes the model using LoRA |
| `test_finetuned_model.py` | Generates output from the trained model |

---
## 🚀 How to Run

### 1. Set up Python Environment

```bash
python3 -m venv llm-finetune-env
source llm-finetune-env/bin/activate
pip install torch torchvision torchaudio transformers datasets peft accelerate bitsandbytes


python load_model_dataset.py
python train_lora_finetune.py
python test_finetuned_model.py
