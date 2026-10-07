"""Write a SYNTHETIC Eedi-shaped fixture for --dry-run tests. Not real data."""
import random
from pathlib import Path

import pandas as pd

out = Path(__file__).parent / "eedi"
out.mkdir(exist_ok=True)
rng = random.Random(0)
subjects = ["Fractions", "Negative Numbers", "BIDMAS", "Percentages", "Angles", "Algebra", "Decimals", "Ratio"]
mis = [{"MisconceptionId": i, "MisconceptionName": f"Synthetic misconception {i}"} for i in range(40)]
rows = []
for q in range(400):
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    correct = rng.choice("ABCD")
    vals = {o: rng.randint(10, 99) for o in "ABCD"}
    vals[correct] = a + b * c
    text = f"Calculate \\( {a} + {b} \\times {c} \\)" if q % 7 else "Which shape is shown below?"
    r = {"QuestionId": q, "ConstructId": q % 120, "ConstructName": f"Construct {q % 120}",
         "SubjectId": q % 8, "SubjectName": subjects[q % 8], "CorrectAnswer": correct, "QuestionText": text}
    for o in "ABCD":
        r[f"Answer{o}Text"] = f"\\( {vals[o]} \\)" if q % 5 else ("Only Tom" if o == correct else "Only Katie")
        r[f"Misconception{o}Id"] = None if o == correct or rng.random() < 0.3 else rng.randint(0, 39)
    rows.append(r)
pd.DataFrame(rows).to_csv(out / "train.csv", index=False)
pd.DataFrame(mis).to_csv(out / "misconception_mapping.csv", index=False)
print("wrote", out)
