"""Helper to obtain the `creditcard.csv` dataset used in the assignment (scripts/).

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    target = Path("creditcard.csv")
    if target.exists():
        print(f"Dataset already exists at {target.resolve()}")
        return

    # Try Kaggle API if credentials present
    kaggle_user = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")

    if kaggle_user and kaggle_key:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            print("Downloading creditcard.csv from Kaggle (mlg-ulb/creditcardfraud)...")
            api.dataset_download_file(
                "mlg-ulb/creditcardfraud",
                file_name="creditcard.csv",
                path=".",
                unzip=True,
            )
            if target.exists():
                print("Download complete.")
                return
        except Exception as exc:  # pragma: no cover - best-effort download
            print("Kaggle download failed:", exc)

    # Fallback: print instructions for manual download via kaggle CLI
    print("Could not download dataset automatically.")
    print("If you have the Kaggle CLI installed and configured, run:")
    print(
        "  kaggle datasets download -d mlg-ulb/creditcardfraud -f creditcard.csv -p . --unzip"
    )
    print(
        "Or visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud to download manually."
    )


if __name__ == "__main__":
    main()
