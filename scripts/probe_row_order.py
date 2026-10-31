
from datasets import load_dataset

ds = load_dataset(
    "MedOtter/LIDC-IDRI",
    split="train",
    streaming=True
)

ds = ds.select_columns(["case_id", "slice_index", "n_slices"])

rows = []

for i, row in enumerate(ds):
    rows.append(row)
    if i >= 399:
        break

print("Rows read:", len(rows))
print("First 5 rows:", rows[:5])

cases = []

for row in rows:
    if not cases or cases[-1][0] != row["case_id"]:
        cases.append([
            row["case_id"],
            row["n_slices"],
            []
        ])

    cases[-1][2].append(row["slice_index"])

for case_id, total, indices in cases:
    ordered = indices == sorted(indices)

    contiguous = (
        indices == list(range(indices[0], indices[0] + len(indices)))
    )

    print(
        f"{case_id}: n_slices={total}, "
        f"got={len(indices)}, "
        f"range={indices[0]}..{indices[-1]}, "
        f"ordered={ordered}, "
        f"contiguous={contiguous}"
    )
