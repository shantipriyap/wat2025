import numpy as np
import evaluate, wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import pandas as pd
from datasets import Dataset
import torch


lang_mapping = {"hi": "hindi", "or": "odia", "bn":"bengali", "ml":"malayalam",
                "hin": "hindi", "ori":"odia","ben": "bengali","mly":"malayalam"}

language = "hindi"

wandb_key = "ENTER_WANDB_KEY_HERE"



# Model and tokenizer section
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model_id = "facebook/nllb-200-3.3B"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, dtype=torch.float32, device_map="auto")

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "decoder" in name:
        param.requires_grad = True

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params} / {total_params}")

project_name = f"{base_model_id}-{language}-finetune"

LANG_CODES = {
    "hindi": "hin_Deva",
    "bengali": "ben_Beng",
    "odia": "ory_Orya",
    "malayalam": "mal_Mlym",
    "english": "eng_Latn"
}

src_lang = LANG_CODES.get("english","eng_Latn")
tgt_lang = LANG_CODES.get(language, "hin_Deva")

tokenizer.src_lang = src_lang
tokenizer.tgt_lang = tgt_lang

max_source_length = 512
max_target_length = 512



# Dataset section
wat_dataset_id = f"DebasishDhal99/{language}-visual-genome-instruction-set"

df1 = load_dataset(wat_dataset_id, split = "train").to_pandas()[["english", language]]
df1 = df1.rename(columns={"english": "src", language: "tgt"})

samanantar_dataset_id = "DebasishDhal99/samanantar-subset-translation"

df2 = load_dataset(samanantar_dataset_id, split = language).to_pandas()[['src', 'tgt']]
train_df = pd.concat([df1, df2])

test_df = load_dataset(wat_dataset_id, split = "test").to_pandas()[["english", language]]
test_df = test_df.rename(columns={"english": "src", language: "tgt"})

dev_df = load_dataset(wat_dataset_id, split = "dev").to_pandas()[["english", language]]
dev_df = dev_df.rename(columns={"english": "src", language: "tgt"})

challenge_df = load_dataset(wat_dataset_id, split = "challenge").to_pandas()[["english", language]]
challenge_df = challenge_df.rename(columns={"english": "src", language: "tgt"})

train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)

def preprocess_function(examples):
    inputs = [ex for ex in examples["src"]]
    targets = [ex for ex in examples["tgt"]]
    model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_dev = dev_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Wandb initialization
import wandb
wandb.login(key=wandb_key)

base_project_name = "wat2025"
wandb.init(project=f"{base_project_name}-{project_name}".replace("/", "-"))


# Evaluation Metrics

bleu = evaluate.load("bleu")
sacrebleu = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, tuple):
        preds = preds[0]

    if hasattr(preds, "cpu"):
        preds = preds.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    if labels.ndim == 3:
        labels = np.argmax(labels, axis=-1)

    preds = preds.astype(np.int64)
    labels = labels.astype(np.int64)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    labels = np.where(labels != -100, labels, pad_id)

    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size is not None else getattr(tokenizer, "get_vocab", lambda: {})().__len__()

    if vocab_size:
        preds = np.where((preds >= 0) & (preds < vocab_size), preds, pad_id)
        labels = np.where((labels >= 0) & (labels < vocab_size), labels, pad_id)

    preds_list = [list(p) for p in preds]
    labels_list = [list(l) for l in labels]

    decoded_preds = tokenizer.batch_decode(preds_list, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels_list, skip_special_tokens=True)

    references = [[lab] for lab in decoded_labels]

    bleu_score = bleu.compute(predictions=decoded_preds, references=references)
    sacrebleu_score = sacrebleu.compute(predictions=decoded_preds, references=references)

    wandb.log({
        "bleu": bleu_score.get("bleu", bleu_score.get("score", None)),
        "sacrebleu": sacrebleu_score.get("score", sacrebleu_score.get("bleu", None))
    })

    return {
        "bleu": float(bleu_score.get("bleu", bleu_score.get("score", 0.0))),
        "sacrebleu": float(sacrebleu_score.get("score", sacrebleu_score.get("bleu", 0.0)))
    }




# Training Arguments and Trainer

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results_{language}_lora",
    eval_strategy="steps",
    eval_steps=250,
    save_strategy="steps",
    save_steps=500,
    learning_rate=2e-4,
    per_device_train_batch_size=10, # varies between 4 or 8 or 10 or 16
    per_device_eval_batch_size=10, # varies between 4 or 8 or 10 or 16
    # gradient_accumulation_steps=8,
    num_train_epochs=3,
    weight_decay=0.01,
    predict_with_generate=True,
    logging_dir=f"./logs_{language}_lora",
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="sacrebleu",
    greater_is_better=True,
    generation_max_length=128,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_dev,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
wandb.finish()