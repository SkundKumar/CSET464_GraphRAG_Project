import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("--- 1. LOADING QUANTIZED QWEN 2.5 GENERATOR ---")
# Use "Qwen/Qwen2.5-1.5B-Instruct" here if you need a faster download tonight
model_id = "Qwen/Qwen2.5-3B-Instruct"

# Shrink the model to 4-bit so it fits easily inside the RTX 4050's 6GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": "cuda:0"}
)
print(f"SUCCESS! {model_id} loaded locally in 4-bit precision.")

print("\n--- 2. FEEDING RETRIEVED CONTEXT TO THE LLM ---")
query = "Where was the director of Interstellar born?"

# This simulates what FAISS or Neo4j will hand to the LLM
retrieved_context = (
    "Interstellar is a 2014 epic science fiction film directed by Christopher Nolan. "
    "Christopher Nolan was born in Westminster, London, United Kingdom."
)

prompt = f"Use the following context to answer the question concisely.\n\nContext:\n{retrieved_context}\n\nQuestion: {query}\n\nAnswer:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("Generating answer...")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)

# Slice the output to only show the newly generated text
generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("\n--- 3. FINAL GENERATED ANSWER ---")
print(generated_text.strip())