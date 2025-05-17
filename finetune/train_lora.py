from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from datasets import load_dataset
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

# Load model in 4-bit
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

# LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

# Load dataset with correct path
dataset = load_dataset("json", data_files={"train": "/home/n.dholakia002/LLMCDSR/LLMCDSR/data/pet-beauty/pet_to_beauty.jsonl"})["train"]

def tokenize(example):
    full_text = example["prompt"] + "\n" + example["response"]
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

# Data collator
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training args
args = TrainingArguments(
    output_dir="./lora-llama3-pet2beauty",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    logging_steps=10,
    save_strategy="epoch",
    save_steps=1000,
    num_train_epochs=3,
    bf16=True,
    report_to="none"
)


trainer = Trainer(model=model, args=args, train_dataset=tokenized, tokenizer=tokenizer, data_collator=collator)
trainer.train()

# Save LoRA weights
model.save_pretrained("./lora-llama3-pet2beauty")
