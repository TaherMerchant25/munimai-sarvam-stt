# Install dependencies

import json
import numpy as np
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training data — 510+ examples
# In Kaggle: upload intent_training_data.json as a dataset, or paste inline

TRAINING_DATA = {
    "CASH_RECEIVED": [
        "Rs 800 cash mila", "paanch sau rupaye aaye", "aath hazaar cash aaya",
        "customer ne 3000 diye", "abhi 1500 mila cash mein", "Rs 2000 cash receive hua",
        "teen hazaar rupaye aaye aaj", "500 rupiya mila", "Rs 12000 cash aaya hai",
        "customer se 750 mila", "aaj 4500 cash aaya", "do hazaar cash mein mila",
        "1200 rupaye aaye abhi", "Mehra ji ne 5000 diye cash mein", "sawa do sau rupaye mila",
        "dedh hazaar cash income", "paise aaye 6000", "Rs 350 cash sale",
        "nau sau cash mila abhi", "das hazaar cash hua aaj", "2500 mila Sharma ji se",
        "cash 700 aaya", "250 rupaye ka payment aaya", "aath sau cash aaj",
        "3500 cash receive kiya", "Muneem Rs 800 cash mila", "Muneem paanch hazaar mila",
        "Rs 10000 cash aaya Gupta ji se", "1800 rupaye ka cash payment",
        "dedh lakh mila order ka", "50000 cash aaya wholesale order", "Rs 4200 mila",
        "abhi 900 rupaye aaye", "cash sale 1100", "2800 mila counter pe",
        "Rs 650 cash hua", "teen sau cash income hui", "7500 ka cash payment hua",
        "15000 cash aaya festival sale", "Rs 3300 aaya customer se",
    ],
    "EXPENSE_LOG": [
        "Rs 5000 rent diya", "bijli ka bill Rs 2200", "15000 rent ka kharcha",
        "Ramesh ko 12000 salary diya", "45000 stock kharida Gupta Traders se",
        "500 chai-paani kharcha", "12000 transport ka bill", "aaj 3000 ka kharcha hua",
        "2500 ka miscellaneous expense", "Rs 8000 salary diya Priya ko",
        "maal kharida 35000 ka", "dukaan ka rent 18000", "electricity bill 1800",
        "internet ka recharge 599", "phone bill 499 diya", "packing material 2500",
        "cleaning ka kharcha 800", "maintenance 3500", "supplier ko 50000 diya",
        "Rs 7400 stock purchase", "Muneem Rs 5000 rent diya", "Muneem bijli ka bill 2200",
        "aaj 10000 salary diya Raju ko", "Rs 1500 transport diya",
        "kharcha hua 4000 stock ka", "25000 Gupta Traders ko diye",
        "rent de diya 15000", "800 petty cash kharcha", "dhai hazaar ka bill aaya",
        "dedh lakh stock kharida", "insurance premium 3500 diya", "GST payment 8500",
        "packaging material 1200", "labour charge 2000 diya", "delivery boy ko 500 diya",
        "advertising 3000 ka kharcha", "tax consultant 5000 diya", "office supplies 1800",
        "repair kharcha 4500", "staff lunch 600",
    ],
    "UDHARI_CREATE": [
        "Sharma ji ka 8000 udhari", "Tripathi ji ko 12000 diya udhar",
        "udhari likh lo Mehra ji ka 5000", "15000 udhar diya Patel ji ko",
        "Gupta ji ka 3000 baaki hai", "udhari hai Agarwal ji ka 10000",
        "naya udhari 7000 Singh ji", "Rs 4000 credit diya Verma ji ko",
        "Joshi ji ko 6000 udhar pe diya", "Rs 20000 udhari create karo Mishra ji",
        "Pandey ji ka udhari 9000", "2500 udhar diya Yadav ji ko",
        "Chauhan ji ko 11000 udhar", "udhari ban gayi 8500 Saxena ji ki",
        "Rs 1500 credit Singh ji ko", "Muneem Sharma ji ka 8000 udhari",
        "Muneem Rs 12000 udhar Tripathi ji", "Kapoor ji ne udhar liya 5500",
        "naya customer udhari 3000", "aaj 7500 udhar pe diya",
        "credit note banao 4000 Meena ji ka", "udhar pe saaman diya 6500",
        "Sharma ji ko 10000 ka maal udhar diya", "baaki rakh lo 2000",
        "udhari kar do 15000 Gupta ji ka", "dhai hazaar udhar Rajan ji",
        "dedh lakh udhari wholesale customer", "25000 credit diya hai",
        "udhari entry karo 8000", "3500 udhar pe de diya",
    ],
    "UDHARI_SETTLE": [
        "Sharma ji ne 5000 wapas kiya", "Mehra ji ka udhari khatam",
        "Tripathi ji ne 8000 de diya", "udhari settle ho gaya Gupta ji ka",
        "Patel ji ne paisa wapas kiya 3000", "12000 wapas aa gaya Singh ji se",
        "udhari clear Agarwal ji", "Rs 6000 mil gaya Verma ji se",
        "Joshi ji ne poora paisa de diya", "partial payment 5000 Mishra ji",
        "10000 wapas aaya Pandey ji se", "udhari collect ho gaya",
        "Yadav ji ne 2500 diye", "payment aa gaya Chauhan ji ka",
        "Saxena ji ne settle kar diya", "Muneem Sharma ji ne 5000 wapas kiya",
        "Muneem Mehra ji ka udhari khatam hua", "wapas mila 7000 Kapoor ji se",
        "collection ho gayi 4000", "paisa aa gaya 15000", "udhari ka paisa mila",
        "settle 8000 Gupta ji", "return hua 3500", "collection success Sharma ji",
        "Rs 9000 wapas aa gaya",
    ],
    "QUERY_SUMMARY": [
        "aaj kaisa raha", "aaj ka hisaab bata", "aaj ki summary", "today ka report",
        "din kaisa gaya", "aaj kya hua", "subah se ab tak ka hisaab",
        "daily summary bata", "aaj ka overall", "Muneem aaj kaisa raha",
        "bata aaj kya hua", "din bhar ka hisaab", "aaj kitna hua total",
        "summary de do", "hisaab bata do aaj ka", "kaise raha aaj",
        "aaj ka scene kya hai", "report bana do aaj ka", "overall aaj kaisa tha",
        "aaj ki kamaai aur kharcha bata",
    ],
    "QUERY_PROFIT": [
        "profit kitna hua", "munafa bata", "kitna kamaya aaj", "net profit kya hai",
        "margin kitna hai", "paisa kitna bacha", "fayda kitna hua",
        "Muneem profit kitna hua", "aaj ka munafa bata", "total earning kya hai",
        "kamaai kitni hui", "profit margin kya hai aaj", "net income bata",
        "kitna profit hua is hafte", "monthly profit kya hai",
    ],
    "QUERY_EXPENSE": [
        "rent kitna diya", "sabse zyada kharcha kahan hua", "total expense kya hai",
        "kharcha ka breakdown", "kitna kharcha hua aaj", "salary kitni di",
        "is mahine kitna kharcha hua", "expense report bata",
        "Gupta Traders ko kitna diya", "stock pe kitna kharcha hua",
        "Muneem sabse zyada kharcha kahan", "kharcha ki list bata",
        "top expenses kya hain", "is hafte ka kharcha", "rent aur salary mein kitna gaya",
    ],
    "QUERY_CUSTOMER": [
        "Mehra ji ka record", "kitne customer aaye aaj", "top customer kaun hai",
        "Sharma ji kitna kharidta hai", "customer count bata", "kaun kaun aaya aaj",
        "new customers kitne aaye", "returning customers kitne",
        "customer ki list bata", "sabse accha customer kaun hai",
        "Muneem Mehra ji ka record dikha", "customer data bata",
        "aaj kitne log aaye", "regular customer kaun kaun hain", "customer history bata",
    ],
    "COMMAND_REMIND": [
        "Sharma ji ko remind karo", "sab ko yaad dilao", "udhari collect karo",
        "reminder bhejo sab ko", "Tripathi ji ko payment yaad dilao",
        "overdue walo ko remind karo", "collection shuru karo",
        "sab udhari reminder bhej do", "Mehra ji ko bol do paisa bhejne",
        "remind all overdue", "Muneem sab ko remind karo", "Muneem udhari collect karo",
        "haan bhej do reminders", "collection reminders bhejo",
        "sab pending walon ko yaad dilao", "udhari wapas mangwao",
        "paisa wapas lao", "reminder de do Gupta ji ko",
        "payment remind Patel ji", "follow up karo udhari pe",
    ],
    "COMMAND_GST": [
        "GST file karo", "tax kitna bacha", "GST ka status bata",
        "GSTR-3B kab due hai", "GST return file kar do",
        "tax filing ka kya scene hai", "ITC kitna hai", "GST penalty check karo",
        "Muneem GST file karo", "tax ka hisaab bata", "GST prepare karo",
        "filing kab karni hai", "GST deadline kab hai", "tax compliance check",
        "GSTR-1 ready hai kya",
    ],
    "PAYMENT_TAG": [
        "abhi wali payment Sharma ji ki thi", "ye payment Mehra ji ka hai",
        "last payment Gupta ji ki hai", "pichla payment Tripathi ji ka tha",
        "ye wala Patel ji ka hai", "tag karo ye payment Singh ji ko",
        "customer Agarwal ji hai ye", "payment assign karo Verma ji ko",
        "abhi jo payment aaya wo Joshi ji ka hai", "last UPI Mishra ji ka tha",
    ],
    "GENERAL": [
        "namaste", "hello Muneem", "kaise ho", "thank you", "shukriya",
        "kya kar sakte ho", "help chahiye", "kya kya features hain",
        "weather kaisa hai", "time kya hua", "accha", "theek hai",
        "baad mein", "cancel karo", "ruko",
    ],
}

# Flatten to lists
texts = []
labels = []
for intent, examples in TRAINING_DATA.items():
    for ex in examples:
        texts.append(ex)
        labels.append(intent)

print(f"Total examples: {len(texts)}")
print(f"Intent distribution:")
for intent, count in Counter(labels).most_common():
    print(f"  {intent}: {count}")

# Data augmentation: prefix variations
augmented_texts = list(texts)
augmented_labels = list(labels)

for text, label in zip(texts, labels):
    # Add "Muneem, " prefix if not present
    if not text.lower().startswith("muneem"):
        augmented_texts.append(f"Muneem, {text}")
        augmented_labels.append(label)
    # Remove "Muneem, " prefix if present
    if text.lower().startswith("muneem"):
        clean = text.split(",", 1)[-1].strip() if "," in text else text[7:].strip()
        if clean:
            augmented_texts.append(clean)
            augmented_labels.append(label)

print(f"After augmentation: {len(augmented_texts)} examples")
print(f"Distribution:")
for intent, count in Counter(augmented_labels).most_common():
    print(f"  {intent}: {count}")

# Encode labels
unique_labels = sorted(set(augmented_labels))
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
encoded_labels = [label2id[l] for l in augmented_labels]

print(f"Label mapping: {json.dumps(label2id, indent=2)}")

# Train/val split (stratified)
X_train, X_val, y_train, y_val = train_test_split(
    augmented_texts, encoded_labels, test_size=0.15, random_state=42, stratify=encoded_labels
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)}")

# Load tokenizer
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer loaded: {MODEL_NAME}")

# Create HuggingFace datasets
train_dataset = Dataset.from_dict({"text": X_train, "label": y_train})
val_dataset = Dataset.from_dict({"text": X_val, "label": y_val})

def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

print(f"Tokenized. Max length: 64 tokens")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
)
model.to(DEVICE)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

# Compute metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, preds, average="weighted")
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"f1": f1, "macro_f1": macro_f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./intent_classifier_output",
    num_train_epochs=25,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=3e-5,
    eval_strategy="steps",
    eval_steps=25,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    fp16=torch.cuda.is_available(),
    report_to="none",  # Disable wandb
    save_total_limit=3,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Train!
print("🚀 Starting training...")
trainer.train()
print("✅ Training complete!")

# Evaluate
eval_results = trainer.evaluate()
print(f"Validation Results:")
print(f"  Weighted F1: {eval_results['eval_f1']:.4f}")
print(f"  Macro F1:    {eval_results['eval_macro_f1']:.4f}")

# Full classification report
predictions = trainer.predict(val_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
print("\n" + classification_report(
    y_val, preds,
    target_names=[id2label[i] for i in range(len(unique_labels))],
    digits=3,
))

import os
import onnx
from pathlib import Path

EXPORT_DIR = "./munim_intent_classifier_export"
os.makedirs(EXPORT_DIR, exist_ok=True)

# 1. Save the HuggingFace model + tokenizer
trainer.save_model(f"{EXPORT_DIR}/hf_model")
tokenizer.save_pretrained(f"{EXPORT_DIR}/hf_model")
print(f"✅ HuggingFace model saved to {EXPORT_DIR}/hf_model/")

# 2. Save label mappings
label_config = {
    "label2id": label2id,
    "id2label": id2label,
    "num_labels": len(unique_labels),
    "model_name": MODEL_NAME,
    "max_length": 64,
    "metrics": {
        "weighted_f1": float(eval_results['eval_f1']),
        "macro_f1": float(eval_results['eval_macro_f1']),
    }
}
with open(f"{EXPORT_DIR}/label_config.json", "w") as f:
    json.dump(label_config, f, indent=2)
print(f"✅ Label config saved")

# 3. Export to ONNX
dummy_input = tokenizer("Rs 5000 rent diya", return_tensors="pt", padding="max_length", max_length=64)
dummy_input = {k: v.to(DEVICE) for k, v in dummy_input.items()}

torch.onnx.export(
    model.cpu(),
    (dummy_input["input_ids"].cpu(), dummy_input["attention_mask"].cpu()),
    f"{EXPORT_DIR}/intent_classifier.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    },
    opset_version=14,
)
print(f"✅ ONNX model exported")

# 4. Verify ONNX model
onnx_model = onnx.load(f"{EXPORT_DIR}/intent_classifier.onnx")
onnx.checker.check_model(onnx_model)
print(f"✅ ONNX model verified")

# 5. Test ONNX inference
import onnxruntime as ort

session = ort.InferenceSession(f"{EXPORT_DIR}/intent_classifier.onnx")

test_phrases = [
    "Rs 800 cash mila",
    "Rs 5000 rent diya",
    "Sharma ji ka 8000 udhari",
    "aaj kaisa raha",
    "GST file karo",
    "sab ko remind karo",
]

print("\n📋 ONNX Inference Test:")
for phrase in test_phrases:
    tokens = tokenizer(phrase, return_tensors="np", padding="max_length", max_length=64)
    outputs = session.run(None, {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
    })
    logits = outputs[0][0]
    probs = np.exp(logits) / np.sum(np.exp(logits))
    pred_idx = np.argmax(probs)
    pred_label = id2label[pred_idx]
    confidence = probs[pred_idx]
    print(f"  '{phrase}' → {pred_label} ({confidence:.3f})")

# 6. List files to download
print(f"\n📦 Files to download from {EXPORT_DIR}/:")
for f in sorted(Path(EXPORT_DIR).rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.relative_to(EXPORT_DIR)} ({size_mb:.1f} MB)")

print(f"\n🎯 Place these files in: services/ai-engine/models/exported/")
print(f"   Then update services/nlu/intent_classifier.py to load ONNX model")
