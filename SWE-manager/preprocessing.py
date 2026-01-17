import pandas as pd
import re
import ast 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')

def count_tests(col_value):
    """Count number of tests in a column (FAIL_TO_PASS or PASS_TO_PASS)."""
    if col_value is None:
        return 0
    if isinstance(col_value, list):
        return len(col_value)
    # sometimes it may be stored as a string representation of a list
    try:
        lst = ast.literal_eval(col_value)
        if isinstance(lst, list):
            return len(lst)
        return 0
    except Exception:
        return 0

def get_patch_size(patch_text):
    patch_size = 0
    if not isinstance(patch_text, str):
       patch_size
    # --- Files touched ---
    files = re.findall(r"diff --git a/(.*?) b/", patch_text)
    # --- Line-level changes ---
    added = sum(1 for l in patch_text.split("\n")
                if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in patch_text.split("\n")
                  if l.startswith("-") and not l.startswith("---"))
    patch_size =  added + removed
    return patch_size

def get_n_files_touched(patch_text):
    n_files_touched = 0
    if not isinstance(patch_text, str):
        return n_files_touched

    # --- Files touched ---
    files = re.findall(r"diff --git a/(.*?) b/", patch_text)
    n_files_touched = len(set(files))
    
    return n_files_touched

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    custom_stopwords = {"import", "python", "def", "return", "class", "file", "line", "would"}
    all_stopwords = stop_words.union(custom_stopwords)

    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in all_stopwords and len(word) > 2]
    return ' '.join(tokens)  # HDBSCAN works on embeddings of full text




def process_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    print("Processing instance...")

    ## 1calcualte n_files_touched
    raw_df["n_files_touched"] =  raw_df['patch'].apply(get_n_files_touched)

    ## 2compute patch_size
    raw_df["patch_size"] = raw_df['patch'].apply(get_patch_size)
    
    ## 3compute n_test_files_touched
    raw_df["n_test_files_touched"] = raw_df['test_patch'].apply(get_n_files_touched)

    ## 4compute test_patch_size
    raw_df["test_patch_size"] = raw_df['test_patch'].apply(get_patch_size)

    ## 5compute num_FAIL_TO_PASS
    raw_df["num_FAIL_TO_PASS"] = raw_df["FAIL_TO_PASS"].apply(count_tests)

    ## 6compute num_PASS_TO_PASS
    raw_df["num_PASS_TO_PASS"] = raw_df["PASS_TO_PASS"].apply(count_tests)

    # ## 7compute has_hint
    raw_df["hints_text"] = raw_df["hints_text"].fillna("")
    raw_df["has_hint"] = raw_df["hints_text"].apply(lambda x: len(x.strip()) > 0)
    raw_df["has_hint"] = raw_df["has_hint"].astype(int)

    # ## 8compute hint_length
    raw_df["hint_length"] = raw_df["hints_text"].apply(lambda x: len(x.split()))

    # ## 9compute problem_length_words
    # Fill missing problem statements with empty string
    raw_df["problem_statement"] = raw_df["problem_statement"].fillna("")
    raw_df["problem_length_words"] = raw_df["problem_statement"].apply(lambda x: len(x.split()))

    # ## 10TEXT_COL = "clean_text" combine, and cleaan problem statement+ hint text
    raw_df['clean_text'] = (raw_df['problem_statement'].fillna('') + ' ' + raw_df['hints_text'].fillna('')).apply(clean_text)

    print(f"Finished Processing {len(raw_df)} SWE-Bench Lite instances")
    return raw_df