"""Regenerates the two notebooks in this folder: python notebooks/build_notebooks.py"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def nb(cells):
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        lines = [line + "\n" for line in src.split("\n")]
        lines[-1] = lines[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        out.append(cell)
    return {"cells": out, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "colab": {"provenance": []}},
            "nbformat": 4, "nbformat_minor": 5}


COMMON_PRIVACY = """> **Privacy first.** Google Colab runs on Google's cloud. Only upload data you are allowed to process there –
> **never confidential documents, patient, customer or employee data** unless your organisation approved it.
> The same notebook runs on your own computer with Jupyter (`pip install notebook`, then `jupyter notebook`), which
> keeps everything offline. Using Colab is subject to Google's terms; the libraries used here are open source
> (licences listed at the end)."""

# ----------------------------------------------------------------------------- tabular / text (scikit-learn)
tab = [
("markdown", f"""# Train a tabular or text model for Local AI Lab (scikit-learn → ONNX)

This notebook trains a model **only on the training rows** of a Model Arena dataset, exports it to ONNX with the
metadata Local AI Lab understands (labels, feature columns, dataset and split hashes), and gives you a file to upload
under **Model Arena → Models**. Because the split hash matches, the Arena knows the model never saw the validation or
test rows, and compares it fairly with the built-in algorithms.

{COMMON_PRIVACY}

**You need from Local AI Lab:** the CSV you uploaded, and *Model Arena → your dataset → Download split* (`split.csv`)."""),
("code", """# 1. Install (Colab already has pandas / scikit-learn; skl2onnx converts the pipeline to ONNX)
%pip install -q "skl2onnx>=1.17" "onnx>=1.16" "onnxruntime>=1.18\""""),
("markdown", "## 2. Upload the data\nIn Colab a file picker opens. On your own computer, set the two paths instead."),
("code", """import io, os, json
CSV_PATH = "data.csv"        # your dataset (same file you uploaded to Local AI Lab)
SPLIT_PATH = "split.csv"     # Model Arena → dataset → Download split
try:
    from google.colab import files  # only in Colab
    up = files.upload()
    names = list(up)
    CSV_PATH = next(n for n in names if "split" not in n.lower())
    SPLIT_PATH = next(n for n in names if "split" in n.lower())
except ImportError:
    pass
print(CSV_PATH, SPLIT_PATH)"""),
("markdown", "## 3. Settings – edit these"),
("code", """TARGET = "churn"            # the target column you chose in Local AI Lab
TASK = "classification"      # or "regression"
TEXT_COLUMN = None           # e.g. "ticket" for a text dataset (one long text column), else None
FEATURE_COLUMNS = None       # None = every column except the target / ignored ones below
IGNORE = ["customer_id"]     # ID-like columns (Local AI Lab ignores them too)
MODEL = "random_forest"      # tabular: random_forest | gradient_boosting | logistic_regression ; text: logistic_regression"""),
("code", """import csv
import numpy as np
import pandas as pd

with open(CSV_PATH, "rb") as f:
    raw = f.read()
text = raw.decode("utf-8-sig", errors="replace")
sep = csv.Sniffer().sniff(text[:65536], delimiters=",;\\t|").delimiter
df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False, skipinitialspace=True)
df.columns = [str(c).strip() for c in df.columns]

with open(SPLIT_PATH, encoding="utf-8") as f:
    header = f.readline()
meta = dict(kv.split("=", 1) for kv in header.lstrip("# ").split() if "=" in kv)
split = pd.read_csv(SPLIT_PATH, skiprows=1)  # first line: hashes
DATASET_SHA256, SPLIT_SHA256 = meta.get("dataset_sha256"), meta.get("split_sha256")
print("dataset", DATASET_SHA256[:12], "split", SPLIT_SHA256[:12])
print(split["split"].value_counts())

train_rows = split.loc[split["split"] == "train", "source_row"].astype(int).to_numpy()
train = df.iloc[train_rows].copy()
print(f"training on {len(train)} of {len(df)} rows – validation and test rows are NOT used")"""),
("code", """from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

y = train[TARGET]
if TASK == "classification":
    y = y.map(lambda v: str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v).strip())
else:
    y = pd.to_numeric(y, errors="coerce").astype(float)

if TEXT_COLUMN:
    features = [TEXT_COLUMN]
    X = train[TEXT_COLUMN].fillna("").astype(str).to_numpy().reshape(-1, 1)
    est = LogisticRegression(max_iter=2000) if TASK == "classification" else Ridge()
    pipe = Pipeline([("tfidf", ColumnTransformer([("t", TfidfVectorizer(max_features=50000, ngram_range=(1, 1)), 0)])),
                     ("model", est)])
else:
    features = FEATURE_COLUMNS or [c for c in df.columns if c != TARGET and c not in IGNORE]
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    X = train[features].copy()
    for c in num:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    for c in cat:  # missing text values become "" – exactly what Local AI Lab feeds the model
        X[c] = X[c].fillna("").astype(str)
    pre = ColumnTransformer([("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    if TASK == "classification":
        est = {"random_forest": RandomForestClassifier(200, random_state=0),
               "gradient_boosting": GradientBoostingClassifier(random_state=0),
               "logistic_regression": LogisticRegression(max_iter=2000)}[MODEL]
    else:
        est = {"random_forest": RandomForestRegressor(200, random_state=0),
               "gradient_boosting": GradientBoostingRegressor(random_state=0),
               "logistic_regression": Ridge()}[MODEL]
    pipe = Pipeline([("pre", pre), ("model", est)])

pipe.fit(X, y)
print("trained", type(est).__name__, "on", len(features), "feature column(s)")"""),
("markdown", "## 4. Export to ONNX with Local AI Lab metadata"),
("code", """import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

if TEXT_COLUMN:
    initial_types = [(TEXT_COLUMN, StringTensorType([None, 1]))]
else:
    initial_types = [(c, FloatTensorType([None, 1]) if c in num else StringTensorType([None, 1])) for c in features]
options = {id(pipe): {"zipmap": False}} if TASK == "classification" else {}
if TEXT_COLUMN:  # "C" locale works on every OS / container (en_US may be missing on minimal Linux images)
    options[TfidfVectorizer] = {"locale": "C"}
onx = to_onnx(pipe, initial_types=initial_types, options=options or None, target_opset=17)

labels = [str(c) for c in getattr(pipe, "classes_", [])]
meta = {"kind": "text" if TEXT_COLUMN else "tabular", "task": TASK,
        "feature_columns": json.dumps(features), "dataset_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256,
        "trained_on": "train", "framework": "scikit-learn + skl2onnx",
        "notebook": "local-ai-lab/notebooks/train_tabular_text_sklearn.ipynb"}
if labels:
    meta["labels"] = json.dumps(labels)
for k, v in meta.items():
    if v is not None:
        p = onx.metadata_props.add()
        p.key, p.value = f"local_ai_lab.{k}", str(v)
OUT = "model.onnx"
onnx.save(onx, OUT)
print("saved", OUT, f"{os.path.getsize(OUT) / 1024:.0f} KB", "labels:", labels)"""),
("code", """# 5. Sanity check with ONNX Runtime on a few training rows (the Arena does the real evaluation)
import onnxruntime as ort
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
sample = X[:5] if TEXT_COLUMN else X.iloc[:5]
if TEXT_COLUMN:
    feeds = {TEXT_COLUMN: np.asarray(sample, dtype=object).reshape(-1, 1)}
else:
    feeds = {c: (sample[c].to_numpy(np.float32) if c in num else sample[c].astype(str).to_numpy(object)).reshape(-1, 1)
             for c in features}
print(sess.run(None, feeds)[0][:5])"""),
("code", """# 6. Download the model
try:
    from google.colab import files
    files.download(OUT)
except ImportError:
    print("model written to", os.path.abspath(OUT))"""),
("markdown", """## 7. Use it in Local AI Lab
**Model Arena → Models → Upload model** → pick `model.onnx`, data type = tabular/text, task = the same.
Labels and feature columns are read from the file. "Training data" shows *trained only on this dataset's training
split* because the split hash matches. Then start a comparison on the same dataset.

**Licences:** scikit-learn (BSD-3-Clause), skl2onnx / ONNX (Apache-2.0), ONNX Runtime (MIT), pandas / NumPy (BSD).
Your data and the trained model are yours; make sure you have the right to use the data for training."""),
]

# ----------------------------------------------------------------------------- images (PyTorch)
img = [
("markdown", f"""# Train an image classifier for Local AI Lab (PyTorch → ONNX)

Trains on the **training images** of a Model Arena image dataset (a ZIP of class folders), exports ONNX with the
metadata Local AI Lab understands (labels in output order, preprocessing, dataset and split hashes), and gives you a
file to upload under **Model Arena → Models**.

{COMMON_PRIVACY}

**Weights and licences.** `PRETRAINED = False` trains a small CNN from scratch – no third-party weights involved.
`PRETRAINED = True` starts from torchvision's ImageNet weights (much better with few images). torchvision's code is
BSD-3-Clause, but *pre-trained weights may carry their own terms derived from their training data* (ImageNet's
terms restrict commercial use); check https://pytorch.org/vision/stable/models.html before using such a model
commercially. Use a GPU runtime in Colab (*Runtime → Change runtime type → T4 GPU*).

**You need from Local AI Lab:** the image ZIP you uploaded and *Model Arena → your dataset → Download split*."""),
("code", """# 1. Install (Colab already has torch / torchvision)
%pip install -q "onnx>=1.16" "onnxruntime>=1.18" pillow"""),
("code", """import io, os, json, zipfile
ZIP_PATH, SPLIT_PATH = "images.zip", "split.csv"
try:
    from google.colab import files
    up = files.upload()
    ZIP_PATH = next(n for n in up if n.lower().endswith(".zip"))
    SPLIT_PATH = next(n for n in up if "split" in n.lower())
except ImportError:
    pass

PRETRAINED = False   # True = torchvision MobileNetV3-Small ImageNet weights (see licence note above)
IMAGE_SIZE = 224 if PRETRAINED else 96
EPOCHS = 10
BATCH = 32"""),
("code", """import pandas as pd
with open(SPLIT_PATH, encoding="utf-8") as f:
    header = f.readline()
meta = dict(kv.split("=", 1) for kv in header.lstrip("# ").split() if "=" in kv)
split = pd.read_csv(SPLIT_PATH, skiprows=1)  # first line: hashes
DATASET_SHA256, SPLIT_SHA256 = meta.get("dataset_sha256"), meta.get("split_sha256")
member = dict(zip(split["source"], split["split"]))

SPLIT_DIRS = {"train", "training", "test", "testing", "val", "valid", "validation", "dev"}
items = []  # (zip name, label, split)
with zipfile.ZipFile(ZIP_PATH) as zf:
    for info in zf.infolist():
        name = info.filename.replace("\\\\", "/")
        if info.is_dir() or "__MACOSX" in name or name.split("/")[-1].startswith("."):
            continue
        parts = name.split("/")
        for k in range(len(parts)):  # the split file lists the path inside the ZIP (maybe without a wrapper folder)
            src = "/".join(parts[k:])
            if src in member:
                items.append((info.filename, parts[-2], member[src]))
                break
LABELS = sorted({lab for _, lab, _ in items})   # Local AI Lab orders classes alphabetically
train_items = [(n, LABELS.index(l)) for n, l, s in items if s == "train"]
print(len(items), "images,", len(train_items), "for training – validation/test images are NOT used. Classes:", LABELS)"""),
("code", """import random
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]   # = "imagenet" normalisation in Local AI Lab
train_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
eval_tf = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
                              transforms.Normalize(MEAN, STD)])

class ZipImages(Dataset):
    def __init__(self, entries, tf):
        self.entries, self.tf, self.zf = entries, tf, None
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, i):
        if self.zf is None:
            self.zf = zipfile.ZipFile(ZIP_PATH)
        name, y = self.entries[i]
        img = Image.open(io.BytesIO(self.zf.read(name))).convert("RGB")
        return self.tf(img), y

random.seed(0)
random.shuffle(train_items)
hold = max(1, len(train_items) // 10)   # early stopping uses a slice of the TRAINING images only
fit_items, hold_items = train_items[hold:], train_items[:hold]
fit_dl = DataLoader(ZipImages(fit_items, train_tf), batch_size=BATCH, shuffle=True, num_workers=2)
hold_dl = DataLoader(ZipImages(hold_items, eval_tf), batch_size=BATCH)

if PRETRAINED:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(LABELS))
else:
    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
    model = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, len(LABELS)))
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3 if not PRETRAINED else 3e-4)
loss_fn = nn.CrossEntropyLoss()"""),
("code", """best, best_state = -1.0, None
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in fit_dl:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in hold_dl:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item()
            total += len(yb)
    acc = correct / max(1, total)
    print(f"epoch {epoch + 1}: hold-out (from training images) accuracy {acc:.3f}")
    if acc > best:
        best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
model.load_state_dict(best_state)"""),
("markdown", "## Export to ONNX with Local AI Lab metadata"),
("code", """import onnx
model = model.cpu().eval()
dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
OUT = "image_model.onnx"
torch.onnx.export(model, dummy, OUT, input_names=["input"], output_names=["logits"],
                  dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
m = onnx.load(OUT)
meta = {"kind": "image", "task": "classification", "labels": json.dumps(LABELS),
        "preprocess": json.dumps({"size": IMAGE_SIZE, "layout": "nchw", "normalization": "imagenet", "color": "rgb",
                                  "resize_mode": "resize", "channels": 3}),
        "dataset_sha256": DATASET_SHA256, "split_sha256": SPLIT_SHA256, "trained_on": "train",
        "framework": f"pytorch {torch.__version__}",
        "weights": "torchvision MobileNetV3-Small ImageNet (check licence)" if PRETRAINED else "trained from scratch",
        "notebook": "local-ai-lab/notebooks/train_image_classifier_pytorch.ipynb"}
for k, v in meta.items():
    if v is not None:
        p = m.metadata_props.add()
        p.key, p.value = f"local_ai_lab.{k}", str(v)
onnx.save(m, OUT)
import onnxruntime as ort
print(ort.InferenceSession(OUT).run(None, {"input": dummy.numpy()})[0].shape, f"{os.path.getsize(OUT) / 1e6:.1f} MB")
try:
    from google.colab import files
    files.download(OUT)
except ImportError:
    print("model written to", os.path.abspath(OUT))"""),
("markdown", """## Use it in Local AI Lab
**Model Arena → Models → Upload model** → `image_model.onnx`, data type *image*, task *classification*. Labels and
preprocessing are filled in from the file; training data shows *trained only on this dataset's training split*.
Then compare it with the built-in classifiers (raw pixels, MobileNetV2 / EfficientNet-Lite features) on the same
dataset. Because this model never saw the validation/test images, the comparison is fair.

You can also upload it with role **feature extractor**: Local AI Lab then trains small classifiers on its features.

**Licences:** PyTorch / torchvision (BSD-3-Clause), ONNX (Apache-2.0), ONNX Runtime (MIT), Pillow (MIT-CMU). Pre-trained
weights: see the note at the top."""),
]

(OUT / "train_tabular_text_sklearn.ipynb").write_text(json.dumps(nb(tab), indent=1), encoding="utf-8")
(OUT / "train_image_classifier_pytorch.ipynb").write_text(json.dumps(nb(img), indent=1), encoding="utf-8")
print("ok")
