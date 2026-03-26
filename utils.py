
import pickle
from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.20, random_state=42):
    """Stratified train/test split — preserves class proportions."""
    return train_test_split(
        X, y,
        test_size    = test_size,
        random_state = random_state,
        stratify     = y,
    )


def save_model(model, filepath: str):
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {filepath}")


def load_model(filepath: str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def print_section(title: str):
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")
