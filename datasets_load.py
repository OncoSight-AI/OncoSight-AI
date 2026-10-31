from datasets import load_dataset

print("Connecting to Hugging Face dataset...")

# Using streaming=True prevents your laptop from trying to download all 71 GB at once
dataset = load_dataset("MedOtter/LIDC-IDRI", split="train", streaming=True)

# Print out a few samples to verify it works
for i, sample in enumerate(dataset):
    print(f"\n--- Sample {i+1} ---")
    print(f"Case ID: {sample['case_id']}")
    print(f"Patient ID: {sample['patient_id']}")
    print(f"Manufacturer: {sample['manufacturer']}")
    print(f"Number of Slices: {sample['n_slices']}")
    
    if i >= 2:
        break