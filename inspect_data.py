from datasets import load_dataset

print("[*] Downloading the 2WikiMultihopQA dataset from Hugging Face...")
# We use the framolfese mirror which is cleanly formatted for Python
dataset = load_dataset("framolfese/2WikiMultihopQA", split="train")

print(f"[*] Successfully loaded {len(dataset)} real Wikipedia questions.\n")

# Let's look at the very first question in the dataset
sample = dataset[0]

print("--- RAW DATASET SAMPLE ---")
print(f"Question Type : {sample['type']}")
print(f"Question      : {sample['query'] if 'query' in sample else sample['question']}")
print(f"Answer        : {sample['answer']}")
print("\nFirst 3 Entities in the Knowledge Graph Evidence:")
for evidence in sample['evidences'][:3]:
    print(f" - {evidence}")