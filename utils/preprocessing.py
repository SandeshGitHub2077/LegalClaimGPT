import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_features(cases: list) -> tuple:
    """
    Takes in a list of case dicts with fields like summary_text, injury_types, etc.
    Returns (X, y) for model input.
    """
    df = pd.DataFrame(cases)

    # Drop incomplete rows
    df = df[df["summary_text"].notnull() & df["settlement_amount"].notnull()]

    # Embed summaries (placeholder: you already embedded during training)
    summaries = df["summary_text"].tolist()

    # Encode injury_types (multi-label)
    mlb = MultiLabelBinarizer()
    injury_features = mlb.fit_transform(df["injury_types"].apply(lambda x: x if isinstance(x, list) else []))

    # Stack all numeric + injury features
    X = pd.DataFrame(injury_features, columns=mlb.classes_)
    X["medical_bills"] = df["medical_bills"]
    X["lost_wages"] = df["lost_wages"]
    X["plaintiff_age"] = df["plaintiff_age"]
    X["plaintiff_gender"] = df["plaintiff_gender"].map({"Male": 0, "Female": 1, None: -1})

    y = df["settlement_amount"]

    return X, y
