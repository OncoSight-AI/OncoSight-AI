import json
import random
from pathlib import Path

manifest = Path("datasets/manifests/cases_50.json")
output = Path("datasets/manifests/splits_50.json")

with open(manifest, encoding="utf-8") as f:
    cases = json.load(f)

groups = {}

for case in cases:
    patient = case.get("patient_id")
    if not patient:
        raise ValueError(f"Missing patient_id: {case['case_id']}")
    groups.setdefault(patient, []).append(case)

patients = list(groups.keys())
random.Random(42).shuffle(patients)

n = len(patients)
n_train = round(n * 0.70)
n_val = round(n * 0.15)

train_patients = set(patients[:n_train])
val_patients = set(patients[n_train:n_train + n_val])
test_patients = set(patients[n_train + n_val:])

splits = {
    "train": [c for c in cases if c["patient_id"] in train_patients],
    "val": [c for c in cases if c["patient_id"] in val_patients],
    "test": [c for c in cases if c["patient_id"] in test_patients]
}

for name, records in splits.items():
    print(f"{name}: {len(records)} cases, "
          f"{len({c['patient_id'] for c in records})} patients")

assert not (train_patients & val_patients)
assert not (train_patients & test_patients)
assert not (val_patients & test_patients)

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(splits, indent=2), encoding="utf-8")

print("\nPatient-level split completed successfully.")
print(f"Saved: {output}")