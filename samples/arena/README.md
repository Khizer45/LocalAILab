# Model Arena sample files

Synthetic data (fixed random seed) for trying the Model Arena. Regenerate with:

```bat
backend\.venv\Scripts\python scripts\make_arena_samples.py --joblib
```

| File | Type | Upload as | What to try |
|---|---|---|---|
| `customer_churn.csv` | tabular classification, 700 rows | dataset, target `churn` | 8+ algorithms vs the uploaded joblib model; note that `customer_id` is ignored automatically and `monthly_charges` has a few missing values |
| `house_prices.csv` | tabular regression, 600 rows | dataset, target `price_lakh_pkr` | linear vs tree models with tuning; R², MAE in lakh PKR |
| `support_tickets.csv` | text classification, ~400 tickets, 5 classes, 8 % label noise | dataset, target `category` | TF-IDF models vs local LLMs zero-shot; label noise means ~90 % is the realistic ceiling |
| `shapes_images.zip` | image classification, 180 images (circle / square / triangle) | dataset | raw pixels vs MobileNetV2 / EfficientNet features (download them first) vs the ONNX model below |
| `shapes_classifier.onnx` | ONNX image classifier, input `[N, 3, 32, 32]`, output 3 probabilities (alphabetical class order) | model: Images, Classifier, normalisation **zero_one** | an “uploaded CNN” you did not train in the app |
| `churn_random_forest.joblib` | scikit-learn pipeline (created by the script, matches your scikit-learn version) | model: Tabular, Classification, *I trust this file* (admin) | an uploaded notebook model; trained on half of the rows, so the arena split's test rows may include rows it saw – a realistic leakage example |
