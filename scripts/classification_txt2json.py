#convert classification report txt file to json file
#for cases where the json file was not saved during model training
import re, json

with open("./models/cnn_rgbs_5l/classification_report.txt", "r") as f:
    text = f.read()

# Match label names and 4 numeric columns
pattern = r"^(?P<label>[A-Za-z0-9_ ]+)\s+(?P<precision>\d\.\d+)\s+(?P<recall>\d\.\d+)\s+(?P<f1>\d\.\d+)\s+(?P<support>\d+)"
lines = re.finditer(pattern, text, flags=re.MULTILINE)

report = {}
for m in lines:
    label = m.group("label").strip()
    report[label] = {
        "precision": float(m.group("precision")),
        "recall": float(m.group("recall")),
        "f1-score": float(m.group("f1")),
        "support": int(m.group("support")),
    }


# accuracy line
acc_match = re.search(r"accuracy\s+([\d.]+)\s+(\d+)", text)
if acc_match:
    accuracy, support = acc_match.groups()
    report["accuracy"] = {"accuracy": float(accuracy), "support": float(support)}


with open("./models/cnn_rgbs_5l/classification_report.json", "w") as f:
    json.dump(report, f, indent=2)
