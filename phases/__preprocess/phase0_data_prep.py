import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import re
import numpy as np
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from constants import COL_MAP, STAR_HEADERS, AM_HEADERS, DATASET_PATH, GRADE_DISTRIBUTION
from typing import Optional

SECTIONS = ['situation', 'task_action', 'result', 'pfr', 'learning']


# Download required NLTK data if not already present
for resource in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource not in ['punkt', 'averaged_perceptron_tagger'] else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def clean_section(text: str, section: str) -> str:
    """
    Remove sub-headings that appear within an already-extracted section.
    Handles students who split TASK/ACTION into separate sub-headings.
    Truncates DEAL framework content embedded in STAR sections.
    """
    if section == 'task_action':
        text = re.sub(r'(?m)^\s*ACTION\s*:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^\s*TASK\s*:\s*', '', text, flags=re.IGNORECASE)

    # Truncate any DEAL framework content embedded in STAR sections
    # Some students placed DEAL content inside their STAR Result column
    deal_boundary = re.compile(
        r'\s*(DEAL\s+FRAMEWORK\s*[:/]?|'           # DEAL FRAMEWORK: or DEAL FRAMEWORK
        r'D\s*[–\-]\s*DESCRIBE\s*[:/]|'            # D – DESCRIBE:
        r'(?<!\w)DESCRIBE\s*:|'                     # DESCRIBE: (not preceded by word char)
        r'(?<!\w)EXAMINE\s*:|'                      # EXAMINE:
        r'ARTICULATE\s+LEARNING\s*:|'              # ARTICULATE LEARNING:
        r'^\s*ACT\s*:\s*(?=[A-Z]))',
        re.IGNORECASE | re.MULTILINE
    )
    match = deal_boundary.search(text)
    if match:
        text = text[:match.start()].strip()

    return re.sub(r'\s+', ' ', text).strip()

def get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS treebank tag to WordNet POS for better lemmatisation."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # default

def normalise(text: str) -> str:
    """
    Normalise text:
    - Lowercase
    - Remove punctuation and special characters
    - Collapse multiple whitespace to single space
    - Strip leading/trailing whitespace
    Note: stop words are RETAINED intentionally — see 3.2.3
    """
    text = text.lower()
    # Remove punctuation and special characters but keep spacess
    text = re.sub(r'[^\w\s]', ' ', text)
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def lemmatise(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """
    Lemmatise text using POS-aware WordNet lemmatiser.
    Stop words are retained — only base form normalisation applied.
    """
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    lemmatised = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]
    return ' '.join(lemmatised)


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer) -> str:
    """Full pre-processing pipeline: normalise → lemmatise."""
    if not isinstance(text, str) or not text.strip():
        return ''
    text = normalise(text)
    text = lemmatise(text, lemmatizer)
    return text


# ─────────────────────────────────────────────
# STEP 1 — Load and inspect dataset
# ─────────────────────────────────────────────

def load_dataset(path: str) -> pd.DataFrame:
    """Load raw dataset, assign column names, drop header rows."""
    df = pd.read_excel(path, header=None)
    df = df.rename(columns=COL_MAP)

    # Drop any row where 'unit' looks like a header (e.g. 'Class')
    df = df[~df['unit'].isin(['Class', 'Unit'])]
    df = df.reset_index(drop=True)

    print(f"Loaded {len(df)} total rows")
    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print("\n -- Dataset inspection --")
    print(f"Total reflections: {len(df)}")
    print(f"\nUnits:")
    print(df['unit'].value_counts().to_string())

    print(f"\nGrade column (col 9) sample values:")
    print(df['grade'].value_counts().head(10).to_string())

    # Valid grades
    valid = df['grade'].apply(is_valid_grade)
    print(f"\nValid numeric grades: {valid.sum()}")
    print(f"Ungraded/invalid: {(~valid).sum()}")


# ─────────────────────────────────────────────
# STEP 2 — Filter and validate
# ─────────────────────────────────────────────

def is_valid_grade(val) -> bool:
    """Check if a grade value is a valid numeric grade between 0 and 10."""
    try:
        g = float(val)
        return 0 <= g <= 10
    except:
        return False


def has_all_sections(star_text: str, pfr_l_text: str) -> bool:
    """Check if a reflection has all 5 extractable sections."""
    if pd.isna(star_text) or pd.isna(pfr_l_text):
        return False
    star_text = str(star_text)
    pfr_l_text = str(pfr_l_text)

    has_situation = bool(STAR_HEADERS['situation'].search(star_text))
    has_task = bool(STAR_HEADERS['task_action'].search(star_text))
    has_result = bool(STAR_HEADERS['result'].search(star_text))
    has_pfr = bool(AM_HEADERS['pfr'].search(pfr_l_text))
    has_learning = bool(AM_HEADERS['learning'].search(pfr_l_text))

    return all([has_situation, has_task, has_result, has_pfr, has_learning])


def filter_usable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only reflections with a valid grade and all 5 extractable sections."""
    print("\n=== FILTERING USABLE REFLECTIONS ===")

    # Filter valid grades
    grade_mask = df['grade'].apply(is_valid_grade)
    print(f"Has valid grade: {grade_mask.sum()}")

    # Filter extractable sections
    section_mask = df.apply(
        lambda row: has_all_sections(row['star_text'], row['pfr_l_text']),
        axis=1
    )
    print(f"Has all 5 sections: {section_mask.sum()}")

    # Combined filter
    usable = df[grade_mask & section_mask].copy()
    usable['grade'] = usable['grade'].astype(float)
    usable = usable.reset_index(drop=True)

    print(f"\nUsable reflections retained: {len(usable)}")
    print(f"Dropped (no grade): {(~grade_mask).sum()}")
    print(f"Dropped (malformed sections): {(grade_mask & ~section_mask).sum()}")

    print(f"\nUsable by unit:")
    print(usable['unit'].value_counts().to_string())

    return usable

# ─────────────────────────────────────────────
# STEP 3 — Seperate to each section: S, TA, R, PR, L
# ─────────────────────────────────────────────

def extract_sections(star_text: str, pfr_l_text: str, 
                     star_headers: dict, am_headers: dict) -> Optional[dict]:
    """
    Extract all 5 sections from a reflection.
    Returns dict with keys: situation, task_action, result, pfr, learning
    Returns None if any section is empty or missing.
    """
    sections = {}

    # --- Extract STAR sections from col 11 ---
    star_text = str(star_text)

    # Find positions of each STAR heading
    star_positions = {}
    for key, pattern in star_headers.items():
        match = pattern.search(star_text)
        if match:
            star_positions[key] = (match.start(), match.end())

    # Extract text between headings
    star_keys = ['situation', 'task_action', 'result']
    for i, key in enumerate(star_keys):
        if key not in star_positions:
            return None  # heading not found

        start = star_positions[key][1]  # end of heading

        # Find the next heading's start position
        next_starts = []
        for other_key in star_keys:
            if other_key != key and other_key in star_positions:
                if star_positions[other_key][0] > star_positions[key][0]:
                    next_starts.append(star_positions[other_key][0])

        end = min(next_starts) if next_starts else len(star_text)
        section_text = star_text[start:end].strip()

        if not section_text:
            return None  # empty section

        sections[key] = section_text

    # --- Extract AM sections from col 12 ---
    pfr_l_text = str(pfr_l_text)

    am_positions = {}
    for key, pattern in am_headers.items():
        match = pattern.search(pfr_l_text)
        if match:
            am_positions[key] = (match.start(), match.end())

    am_keys = ['pfr', 'learning']
    for i, key in enumerate(am_keys):
        if key not in am_positions:
            return None

        start = am_positions[key][1]

        next_starts = []
        for other_key in am_keys:
            if other_key != key and other_key in am_positions:
                if am_positions[other_key][0] > am_positions[key][0]:
                    next_starts.append(am_positions[other_key][0])

        end = min(next_starts) if next_starts else len(pfr_l_text)
        section_text = pfr_l_text[start:end].strip()

        if not section_text:
            return None  # empty section

        sections[key] = section_text

    return sections


def extract_all_sections(df: pd.DataFrame, 
                         star_headers: dict, 
                         am_headers: dict) -> pd.DataFrame:
    """
    Apply section extraction to all rows.
    Drops rows where any section is empty or missing.
    """
    print("=== SECTION EXTRACTION ===")
    pre_drop_count = len(df)
    results = []
    dropped_empty = 0
    dropped_missing = 0

    for idx, row in df.iterrows():
        sections = extract_sections(
            row['star_text'], 
            row['pfr_l_text'],
            star_headers,
            am_headers
        )

        if sections is None:
            dropped_missing += 1
            continue

        # Clean sub-headings from sections
        for key in sections:
            sections[key] = clean_section(sections[key], key)

        # Section-level empty check
        empty_sections = [k for k, v in sections.items() if len(v.strip()) < 10]
        if empty_sections:
            dropped_empty += 1
            continue

        results.append({
            'unit': row['unit'],
            'submission_ref': row['submission_ref'],
            'student_id': row['student_id'],
            'grade': row['grade'],
            'topic': row['topic'],
            'situation': sections['situation'],
            'task_action': sections['task_action'],
            'result': sections['result'],
            'pfr': sections['pfr'],
            'learning': sections['learning'],
        })

    df_extracted = pd.DataFrame(results).reset_index(drop=True)


    # Drop list — manually identified malformed reflections
    # Tuple format: (student_id, submission_ref)
    DROP_ENTRIES = [
        # Two SITUATION headings detected — reflection appears to be two separate
        # reflections concatenated into one submission, making section extraction unreliable
        ('6520576F6F644A616B6534373133323538', 'Ref1_w7'),
        ('6D65656E343539353538353972616D6565', 'Ref2_w13'),

        # Student used 'DEAL LEARNING/REFLECTION:' as a merged heading combining
        # both PFR and Learning into one block. Learning section contains an activity
        # log rather than genuine reflective content, making it structurally inconsistent
        # with the rest of the corpus and unsuitable for Bloom annotation.
        ('726B34363631373639386D61726B2E6D61', 'Ref1_w7'),
        ('726B34363631373639386D61726B2E6D61', 'Ref2_w13'),
        # PFR section starts with DEAL 'Describe:' heading — student placed
        # DEAL framework content in PFR column instead of using correct AM headings
        # making the PFR section structurally inconsistent with the rest of the corpus
        ('696E4D6F68616D6D616434373633363439', 'Ref1_w7'),
    ]
    # 1. Manual drops first
    df_extracted = df_extracted[~df_extracted.apply(
        lambda row: (row['student_id'], row['submission_ref']) in DROP_ENTRIES,
        axis=1
    )].reset_index(drop=True)
    actual_dropped_manual = pre_drop_count - dropped_missing - dropped_empty - len(df_extracted)

    # 2. Dedup after
    before_dedup = len(df_extracted)
    df_extracted = df_extracted.drop_duplicates(
        subset=['student_id', 'result'],
        keep='first'
    ).reset_index(drop=True)
    dropped_duplicates = before_dedup - len(df_extracted)

    # 3. Print summary
    print(f"Raw extracted (before drops): {pre_drop_count}")
    print(f"Dropped (missing section): {dropped_missing}")
    print(f"Dropped (empty section < 10 chars): {dropped_empty}")
    print(f"Dropped (malformed — manual): {actual_dropped_manual}")
    print(f"Dropped (duplicates): {dropped_duplicates}")
    print(f"Final retained: {len(df_extracted)}")
    # Section-level stats
    print("Section length stats (word count):")
    for section in ['situation', 'task_action', 'result', 'pfr', 'learning']:
        lengths = df_extracted[section].apply(lambda x: len(x.split()))
        print(f"  {section:12s} — min: {lengths.min():4d}  "
              f"mean: {lengths.mean():6.1f}  "
              f"max: {lengths.max():4d}")

    return df_extracted

# ─────────────────────────────────────────────
# STEP 4 — Carry out preprocess methods for each section
# ─────────────────────────────────────────────

def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply pre-processing to all 5 sections for every reflection.
    Adds preprocessed_<section> columns alongside originals.
    Drops rows where any preprocessed section is empty.
    """
    print("=== PRE-PROCESSING ===")
    lemmatizer = WordNetLemmatizer()

    df = df.copy()
    dropped = 0

    for section in SECTIONS:
        col = f'preprocessed_{section}'
        print(f"  Processing {section}...")
        df[col] = df[section].apply(
            lambda x: preprocess_text(str(x), lemmatizer)
        )

    # Drop rows where any preprocessed section is empty
    empty_mask = df[[f'preprocessed_{s}' for s in SECTIONS]].apply(
        lambda col: col.str.strip() == ''
    ).any(axis=1)

    dropped = empty_mask.sum()
    df = df[~empty_mask].reset_index(drop=True)

    print(f"\nDropped (empty after preprocessing): {dropped}")
    print(f"Retained: {len(df)} reflections")

    # Quick sanity check — word counts per section
    print("\nPreprocessed section word counts:")
    for section in SECTIONS:
        col = f'preprocessed_{section}'
        lengths = df[col].apply(lambda x: len(x.split()))
        print(f"  {section:12s} — min: {lengths.min():4d}  "
              f"mean: {lengths.mean():6.1f}  "
              f"max: {lengths.max():4d}")

    return df

def grade_distribution(df: pd.DataFrame):

    unique_keywords = df["topic"].unique().tolist()
    print(len(unique_keywords))
    unit_counts = df["unit"].value_counts()

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import pandas as pd
    import numpy as np

    # ── Replace this with your actual df ──────────────────────────
    # unit_counts = df["unit"].value_counts()
    unit_counts = pd.Series({
        "24S1_ENGG1000": 405,
        "24S1_ENGG3050": 270,
        "22S2_ENGG1050": 247,
        "23S1_ENGG1000": 172,
        "24S1_ENGG2050": 153,
        "24S1_ENGG1050": 105,
        "23S1_ENGG1050":  80,
        "23S2_ENGG1000":  74,
        "23S2_ENGG1050":  48,
        "22S1_ENGG1050":   6,
    })
    # ──────────────────────────────────────────────────────────────

    # Colour mapping by unit family
    COLOUR_MAP = {
        "ENGG1000": "#2563EB",   # blue
        "ENGG1050": "#16A34A",   # green
        "ENGG2050": "#D97706",   # amber
        "ENGG3050": "#DC2626",   # red
    }

    def get_colour(label):
        for key, colour in COLOUR_MAP.items():
            if key in label:
                return colour
        return "#6B7280"

    labels  = unit_counts.index.tolist()
    values  = unit_counts.values.tolist()
    colours = [get_colour(l) for l in labels]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bars = ax.barh(labels, values, color=colours, height=0.6,
                edgecolor="white", linewidth=1.2)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left",
                fontsize=10, fontweight="600", color="#1E293B")

    # Grid
    ax.xaxis.grid(True, color="#E2E8F0", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD5E1")

    # Axes
    ax.set_xlabel("Number of Reflections", fontsize=11,
                color="#475569", labelpad=10)
    ax.set_xlim(0, max(values) * 1.15)
    ax.tick_params(axis="y", labelsize=10, colors="#1E293B")
    ax.tick_params(axis="x", labelsize=10, colors="#64748B")
    ax.invert_yaxis()

    # Title
    ax.set_title("Dataset Composition by Unit and Semester",
                fontsize=13, fontweight="700", color="#0F172A",
                pad=16, loc="left")

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=k) for k, c in COLOUR_MAP.items()
    ]
    ax.legend(handles=legend_patches, title="Unit Family",
            title_fontsize=9, fontsize=9,
            loc="lower right", framealpha=0.85,
            edgecolor="#E2E8F0")

    plt.tight_layout()
    plt.savefig("unit_distribution.png", dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved → unit_distribution.png")

    hd = 0

    curr_grades = {
        "fail": 0,
        "pass": 0,
        "credit": 0,
        "distinction": 0,
        "high_distinction": 0,
    }

    for grade in df.grade:
        for band, ranges in GRADE_DISTRIBUTION.items():
            x1, x2 = ranges;
            if float(x1) <= int(grade) <= float(x2):
                curr_grades[band] +=1
                if band == 'high_distinction':
                    hd += grade

            
    print("CURRENT GRADE DISTRIBUTION:\n")
    print(curr_grades)
    print(f"fail:{curr_grades['fail'] / 1560}\n")
    print(f"pass:{curr_grades['pass'] / 1560}\n")
    print(f"credit:{curr_grades['credit'] / 1560}\n")
    print(f"distinction:{curr_grades['distinction'] / 1560}\n")
    print(f"high_distinction:{curr_grades['high_distinction'] / 1560}\n")
    print(f"hd average: {hd/curr_grades['high_distinction']}")
    

if __name__ == '__main__':
    df_raw = load_dataset(DATASET_PATH)
    df_usable = filter_usable(df_raw)
    df_extracted = extract_all_sections(df_usable, STAR_HEADERS, AM_HEADERS)

    df_extracted.to_csv('data/extracted_sections.csv', index=False)
    print(f"Saved extracted sections to data/extracted_sections.csv")

    df_preprocessed = preprocess_all(df_extracted)
    df_preprocessed.to_csv('data/preprocessed_sections.csv', index=False)
    print(f"Saved to data/preprocessed_sections.csv")

