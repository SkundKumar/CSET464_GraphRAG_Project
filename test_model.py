import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("--- 1. CHECKING LOCAL GPU ---")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device.upper()}")
if device == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

print("\n--- 2. LOADING YOUR CUSTOM DISTILBERT MODEL ---")
model_path = "./relation_extractor_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

print("Model loaded successfully from local weights!")

print("\n--- 3. RUNNING A TEST PREDICTION ---")
sample_text = "Subject: Interstellar | Object: Christopher Nolan [SEP] Context: Interstellar is a 2014 epic science fiction film directed by Christopher Nolan."
inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=256).to(device)

with torch.no_grad():
    logits = model(**inputs).logits
    predicted_id = logits.argmax().item()

    # Check if the model has label mappings, otherwise print the raw ID
    if hasattr(model.config, "id2label") and model.config.id2label:
        predicted_relation = model.config.id2label[predicted_id]
    else:
        predicted_relation = f"Label ID: {predicted_id}"

print(f"Input: Interstellar -> Christopher Nolan")
print(f"Predicted Relation: {predicted_relation.upper()}")
print("\nSUCCESS! Your custom model is working locally on your laptop.")