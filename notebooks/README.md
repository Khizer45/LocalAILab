# Notebooks – train your own model for the Model Arena

The Model Arena trains classic algorithms on your computer's CPU. Deep networks train much faster on a GPU – for
example the free GPU in Google Colab. These notebooks train a model **on exactly the training rows of a Model Arena
dataset**, export it to ONNX with metadata Local AI Lab understands, and give you a file to upload. Because the file
records the dataset and split fingerprints, the Arena can confirm that the model never saw the validation or test
rows, so its comparison stays honest.

| Notebook | Data | Model | Runs on |
|---|---|---|---|
| `train_tabular_text_sklearn.ipynb` | CSV: tabular columns or one text column | scikit-learn pipeline (random forest, gradient boosting, logistic / ridge regression; TF-IDF for text) → ONNX via skl2onnx | any CPU; Colab or local Jupyter |
| `train_image_classifier_pytorch.ipynb` | ZIP of class folders | small CNN from scratch, or MobileNetV3-Small with ImageNet weights (`PRETRAINED = True`) → ONNX | GPU recommended (Colab: *Runtime → Change runtime type → T4 GPU*) |

## Steps

1. In Local AI Lab: **Model Arena → Datasets** → your dataset → **Split**. This downloads `…_split.csv`
   (which row / image belongs to train, validation and test, plus the fingerprints).
2. Open a notebook:
   * **Colab:** <https://colab.research.google.com> → *File → Upload notebook* → pick the `.ipynb` file.
   * **Your computer (stays offline):** `pip install notebook`, then `jupyter notebook` in this folder.
3. Run the cells. The upload cell asks for your data file (the same CSV or ZIP you gave Local AI Lab) and the split
   file. Edit the *Settings* cell (target column, task, model).
4. Download the `.onnx` file at the end and upload it in **Model Arena → Models & extractors**. Labels, feature
   columns / preprocessing and the training-data provenance are filled in from the file.
5. Compare it with the built-in algorithms on the same dataset.

The notebooks train on the **training** rows only (the image notebook keeps 10 % of them for early stopping). They
never read validation or test labels. If you train on train + validation yourself, the metadata says so and the
Arena does not rank the model on validation.

## Privacy and licences – read before using Colab

* Google Colab runs on Google's cloud under Google's terms. **Do not upload confidential, personal, patient,
  customer or employee data** unless your organisation has approved it. The same notebooks run locally with
  Jupyter, which keeps everything on your computer.
* The libraries are open source: scikit-learn (BSD-3-Clause), skl2onnx, ONNX and ONNX Runtime (Apache-2.0 / MIT),
  PyTorch and torchvision (BSD-style).
* `PRETRAINED = True` downloads ImageNet-trained weights. The code is BSD-licensed, but the weights were trained on
  ImageNet, whose terms limit the dataset to non-commercial research and education. Check
  <https://pytorch.org/vision/stable/models.html> and get advice before using such a model commercially –
  `PRETRAINED = False` avoids the question.

## Regenerating the notebooks

The notebooks are generated from `build_notebooks.py` (so they can be reviewed as plain Python):

```bash
python notebooks/build_notebooks.py
```

`backend/tests/test_notebooks.py` executes the scikit-learn notebook end to end and checks that the Arena recognises
the exported model and its split; the PyTorch notebook is compile-checked (PyTorch is not installed in CI).
