from pathlib import Path
from datasets import load_dataset
import pandas as pd




def get_data_dir() -> Path:
    """Return the project-level data directory."""
    return Path(__file__).resolve().parents[1] / "data"


def get_swebench_lite_path() -> Path:
    return get_data_dir() / "swebench_lite.csv"


def download_swebench_lite(path: Path):

    path.parent.mkdir(parents=True, exist_ok=True)

    # urllib.request.urlretrieve(SWE_BENCH_LITE_URL, path)
    # Load the dataset
    dataset = load_dataset("SWE-bench/SWE-bench_lite")

    # # Convert a specific split (e.g., "test") to a pandas DataFrame
    df = dataset["test"].to_pandas()
    df.to_csv(get_swebench_lite_path(), index = False)
    print(f"Downloaded SWE-Bench Lite to {path}")


def load_swebench_lite() -> pd.DataFrame:
    path = get_swebench_lite_path()
    if not path.exists():
        print("SWE-Bench Lite dataset not found.")
        print("Downloading SWE-Bench Lite...")
        download_swebench_lite(path)

    print("Loading SWE-Bench Lite dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} SWE-Bench Lite instances")

    return df


def get_instance_row(df: pd.DataFrame, instance_id: str) -> pd.Series:
    matches = df[df["instance_id"] == instance_id]

    if matches.empty:
        raise ValueError(
            f"Instance ID '{instance_id}' not found in SWE-Bench Lite."
        )

    if len(matches) > 1:
        print("Warning: multiple matches found, using first.")

    return matches.iloc[0]
