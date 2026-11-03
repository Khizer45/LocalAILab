"""Generate the sample datasets and models used to try the Model Arena.

All data is synthetic (fixed random seed), so it can be shared freely.

    python scripts/make_arena_samples.py            # CSVs, image ZIP, sample ONNX model
    python scripts/make_arena_samples.py --joblib   # also a scikit-learn .joblib model for the churn CSV

Outputs go to samples/arena/.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "samples" / "arena"
RNG = np.random.default_rng(7)


def churn(n: int = 700) -> pd.DataFrame:
    tenure = RNG.integers(1, 72, n)
    contract = RNG.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.2])
    internet = RNG.choice(["dsl", "fiber", "none"], n, p=[0.4, 0.45, 0.15])
    monthly = np.round(RNG.normal(55, 20, n).clip(15, 120) + (internet == "fiber") * 25, 2)
    calls = RNG.poisson(1.5, n)
    payment = RNG.choice(["card", "bank transfer", "easypaisa", "cash"], n)
    paperless = RNG.choice(["yes", "no"], n, p=[0.6, 0.4])
    logit = (-1.0 + 1.6 * (contract == "month-to-month") - 0.045 * tenure + 0.018 * (monthly - 55)
             + 0.45 * calls + 0.6 * (internet == "fiber") - 0.5 * (contract == "two-year") + RNG.normal(0, 0.8, n))
    churned = np.where(1 / (1 + np.exp(-logit)) > 0.5, "yes", "no")
    monthly = monthly.astype(object)
    monthly[RNG.choice(n, 12, replace=False)] = np.nan  # a few missing values, like real exports
    return pd.DataFrame({"customer_id": [f"C{10000 + i}" for i in range(n)], "tenure_months": tenure,
                         "contract": contract, "internet_service": internet, "monthly_charges": monthly,
                         "support_calls": calls, "payment_method": payment, "paperless_billing": paperless,
                         "churn": churned})


def houses(n: int = 600) -> pd.DataFrame:
    city = RNG.choice(["Islamabad", "Lahore", "Karachi", "Mirpur", "Peshawar"], n)
    area = RNG.integers(500, 4500, n)
    beds = np.clip((area / 700 + RNG.normal(0, 0.8, n)).round(), 1, 7).astype(int)
    baths = np.clip(beds - RNG.integers(0, 2, n), 1, 6)
    age = RNG.integers(0, 40, n)
    garage = RNG.choice(["yes", "no"], n, p=[0.5, 0.5])
    city_factor = pd.Series(city).map({"Islamabad": 1.35, "Lahore": 1.2, "Karachi": 1.15, "Mirpur": 0.9,
                                       "Peshawar": 0.85}).to_numpy()
    price = (area * 0.032 * city_factor + beds * 6 + baths * 4 - age * 0.9 + (garage == "yes") * 12
             + RNG.normal(0, 12, n)).clip(15)
    return pd.DataFrame({"area_sqft": area, "bedrooms": beds, "bathrooms": baths, "age_years": age, "city": city,
                         "garage": garage, "price_lakh_pkr": price.round(1)})


TICKETS = {
    "billing": ["I was charged twice for my {plan} plan this month", "Why is my invoice higher than last {period}?",
                "Please refund the late fee on my {period} bill", "My card payment for the {plan} plan failed",
                "Can I get a receipt for the payment I made last {period}?",
                "The bill shows a {plan} add-on I never ordered", "How do I change the card used for billing?",
                "I want to cancel auto-renewal before I am charged again"],
    "technical": ["The {device} app crashes when I open settings", "Internet keeps disconnecting every {n} minutes",
                  "I get error code {code} when I try to log in on my {device}",
                  "The router light is red and nothing loads", "Video calls freeze after {n} minutes on my {device}",
                  "Sync is not working between my {device} and laptop", "The update failed with error {code}",
                  "Pages load very slowly since yesterday"],
    "delivery": ["My order {order} has not arrived after {n} days", "The parcel was delivered to the wrong address",
                 "Can I change the delivery time for order {order}?", "The courier marked {order} delivered but I got nothing",
                 "The box for order {order} arrived damaged", "When will my {device} be shipped?",
                 "Tracking for {order} has not updated in {n} days", "I need express delivery for order {order}"],
    "account": ["I forgot my password and the reset email never arrives", "How do I change the email on my account?",
                "My account was locked after {n} attempts", "Please delete my account and all my data",
                "I cannot turn on two-factor authentication on my {device}",
                "Someone else may have logged into my account", "How can I add a second user to my account?",
                "I want to change my username"],
    "feedback": ["Great service, the agent solved my issue in {n} minutes", "The new {device} app design is confusing",
                 "Thank you for the quick refund", "Your support team was very rude to me yesterday",
                 "I love the new {plan} plan features", "Please add an Urdu language option",
                 "The checkout process takes too many steps", "Keep up the good work, very happy customer"],
}
FILL = {"plan": ["basic", "premium", "family", "business"], "period": ["month", "week", "quarter"],
        "device": ["Android", "iPhone", "Windows", "tablet"], "n": ["3", "5", "10", "15", "20"],
        "code": ["E102", "403", "500", "X17"], "order": ["#4471", "#5820", "#3109", "#7756"]}
PREFIX = ["", "", "Hi, ", "Hello team, ", "Urgent: ", "Please help. ", "Salam, "]
SUFFIX = ["", "", ".", ". Thanks.", ". Please check.", "!", ". What should I do?"]


MIXED = [("I was charged for a delivery that never arrived", "billing"),
         ("The app crashed while I was paying my bill", "technical"),
         ("I cannot log in to check my order status", "account"),
         ("Thanks for fixing my login problem so fast", "feedback"),
         ("My payment went through but the order still shows pending", "delivery"),
         ("The new update logged me out of my account", "technical"),
         ("Your refund process is too slow", "feedback"),
         ("I changed my email and now I don't get invoices", "account")]


def tickets(per_class: int = 80, noise: float = 0.08) -> pd.DataFrame:
    """Template tickets + a few mixed-intent messages + realistic label noise (8% of labels flipped)."""
    rows = []
    for label, templates in TICKETS.items():
        for _ in range(per_class):
            t = str(RNG.choice(templates))
            for key, values in FILL.items():
                while "{" + key + "}" in t:
                    t = t.replace("{" + key + "}", str(RNG.choice(values)), 1)
            suffix = str(RNG.choice(SUFFIX))
            if t.endswith("?") and suffix.startswith("."):
                suffix = suffix[1:]
            rows.append({"ticket": f"{RNG.choice(PREFIX)}{t}{suffix}".strip(), "category": label})
    for text, label in MIXED:
        for prefix in PREFIX[2:]:
            rows.append({"ticket": f"{prefix}{text}.".strip(), "category": label})
    df = pd.DataFrame(rows).drop_duplicates("ticket").reset_index(drop=True)
    labels = sorted(TICKETS)
    flip = RNG.random(len(df)) < noise
    df.loc[flip, "category"] = [str(RNG.choice([x for x in labels if x != y])) for y in df.loc[flip, "category"]]
    return df.sample(frac=1, random_state=7).reset_index(drop=True)


def shape_image(kind: str, size: int = 64):
    from PIL import Image, ImageDraw

    bg = tuple(int(v) for v in RNG.integers(150, 255, 3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    color = tuple(int(v) for v in RNG.integers(0, 140, 3))
    r = int(RNG.integers(12, 24))
    cx, cy = (int(v) for v in RNG.integers(r + 2, size - r - 2, 2))
    if kind == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif kind == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=color)
    arr = np.asarray(im).astype(np.int16) + RNG.integers(-18, 18, (size, size, 3))
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def shapes_zip(path: Path, per_class: int = 60) -> list[tuple[np.ndarray, str]]:
    data = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for kind in ("circle", "square", "triangle"):
            for i in range(per_class):
                im = shape_image(kind)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                info = zipfile.ZipInfo(f"shapes/{kind}/{kind}_{i:03d}.png", date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(info, buf.getvalue())  # fixed timestamps: the file is identical on every run
                data.append((np.asarray(im.resize((32, 32))), kind))
    return data


def shapes_onnx(path: Path, data: list[tuple[np.ndarray, str]]) -> None:
    """A tiny image classifier (logistic regression on 32×32 pixels) exported as a real ONNX graph.

    Input [N, 3, 32, 32] float, pixels scaled to 0–1 ("zero_one"), output [N, 3] probabilities in the order
    circle, square, triangle – the alphabetical order of the class folders.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from sklearn.linear_model import LogisticRegression

    X = np.stack([a.transpose(2, 0, 1).reshape(-1) / 255.0 for a, _ in data]).astype(np.float32)
    y = [lab for _, lab in data]
    idx = RNG.permutation(len(X))[: int(len(X) * 0.5)]  # trained on half the images only, so it is not perfect
    clf = LogisticRegression(C=0.05, max_iter=2000).fit(X[idx], np.array(y)[idx])
    W = clf.coef_.T.astype(np.float32)
    b = clf.intercept_.astype(np.float32)
    graph = helper.make_graph(
        [helper.make_node("Flatten", ["input"], ["flat"], axis=1),
         helper.make_node("Gemm", ["flat", "W", "b"], ["logits"]),
         helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1)],
        "shapes_logreg",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["batch", 3])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="local-ai-lab-sample")
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)


def churn_joblib(df: pd.DataFrame, path: Path) -> None:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    num = ["tenure_months", "monthly_charges", "support_calls"]
    cat = ["contract", "internet_service", "payment_method", "paperless_billing"]
    pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    pipe = Pipeline([("pre", pre), ("rf", RandomForestClassifier(n_estimators=150, random_state=1))])
    X = df[num + cat].copy()
    X["monthly_charges"] = pd.to_numeric(X["monthly_charges"])
    pipe.fit(X.iloc[: len(X) // 2], df["churn"].iloc[: len(X) // 2])
    joblib.dump(pipe, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joblib", action="store_true", help="also write churn_random_forest.joblib")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = churn()
    c.to_csv(OUT / "customer_churn.csv", index=False)
    houses().to_csv(OUT / "house_prices.csv", index=False)
    tickets().to_csv(OUT / "support_tickets.csv", index=False)
    data = shapes_zip(OUT / "shapes_images.zip")
    shapes_onnx(OUT / "shapes_classifier.onnx", data)
    if args.joblib:
        churn_joblib(c, OUT / "churn_random_forest.joblib")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:32s} {p.stat().st_size / 1024:8.1f} KB")


if __name__ == "__main__":
    main()
