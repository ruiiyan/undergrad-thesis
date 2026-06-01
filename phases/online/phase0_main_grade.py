"""
phase4_online.py
----------------
Online Phase — Assess a target reflection using the two-signal pipeline.

Loads all offline artifacts once, then processes a target reflection through:
  1. Preprocessing
  2. Embedding
  3. Cluster assignment
  4. Signal 1 — weighted grade from top-k neighbours
  5. Signal 2 — Bloom taxonomy annotation
  6. Confidence gate — 3x3 matrix combining signals
  7. Final grade + feedback output

Usage:
    from phases.online.phase4_online import load_resources, assess_reflection

    resources = load_resources()
    result = assess_reflection(
        situation='...',
        task_action='...',
        result='...',
        pfr='...',
        learning='...',
        topic='Team communication',
        resources=resources,
        k=5
    )
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic
from dotenv import load_dotenv
from utils.bloom.annotate_bloom import annotate_bloom, load_system_prompt
from nltk.stem import WordNetLemmatizer
import nltk

from phases.__preprocess.phase0_data_prep import normalise, lemmatise
from constants import SECTIONS, THEMATIC_SECTIONS, EXAMPLE_REFLECTIONS, CONFIDENCE_GATE
from constants import ONLINE_PARAMETERS, S2_LEARNING_WEIGHT, S2_PFR_WEIGHT, S2_RESULT_WEIGHT, S1_WEIGHT, S2_WEIGHT, BLOOM_TO_GRADE, EXPECTED_BLOOM

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CENTROIDS_PATH      = 'data/clusters/cluster_centroids.npy'
KEYWORDS_PATH       = 'data/clusters/cluster_keywords.csv'
ANNOTATED_PATH      = 'data/annotated_reflections.csv'
EMBEDDINGS_PATH     = 'data/embeddings/all/all-MiniLM-L6-v2/all.npy'
UMAP_REDUCER_PATH   = 'data/clusters/umap_reducer.pkl'

SBERT_MODEL         = 'all-MiniLM-L6-v2'
BLOOM_MODEL         = 'claude-sonnet-4-6'

# Empirical defaults — tune after evaluation
DEFAULT_K                   = 10
CLUSTER_CONFIDENCE_THRESHOLD = 0.3

# Confidence gate band thresholds
S1_LOW_MAX  = 6.5
S1_HIGH_MIN = 7.5
S2_LOW_MAX  = 2      # Bloom level
S2_HIGH_MIN = 5      # Bloom level


S2_NORM_CONFIG = 'D'  # A=linear, B=linear capped, C=50/50 blend, D=pure rubric

S2_NORM_PARAMS = {
    'A': {'min_bloom': 1, 'max_bloom': 6, 'grade_cap': 10, 'rubric_blend': 0.0},
    'B': {'min_bloom': 1, 'max_bloom': 5, 'grade_cap': 10, 'rubric_blend': 0.0},
    'C': {'min_bloom': 1, 'max_bloom': 5, 'grade_cap': 10, 'rubric_blend': 0.5},
    'D': {'min_bloom': 1, 'max_bloom': 5, 'grade_cap': 10, 'rubric_blend': 1.0},
}

# Confidence gate S2 band thresholds per config
S2_BAND_THRESHOLDS = {
    'A': {'low_max': 5.7,  'high_min': 8.5},   # Pass boundary, HD boundary
    'B': {'low_max': 5.7,  'high_min': 8.5},
    'C': {'low_max': 5.7,  'high_min': 8.5},
    'D': {'low_max': 5.7,  'high_min': 8.5},   # same for all since rubric-grounded
}
# ─────────────────────────────────────────────
# RESOURCE LOADING
# ─────────────────────────────────────────────

def load_resources() -> dict:
    """
    Load all offline artifacts once.
    Pass the returned dict to assess_reflection() for efficiency.
    """
    load_dotenv()
    print('Loading offline resources...')

    # NLTK
    for resource in ['wordnet', 'omw-1.4', 'punkt', 'punkt_tab',
                     'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
        try:
            nltk.data.find(f'corpora/{resource}' if resource not in
                           ['punkt', 'averaged_perceptron_tagger']
                           else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

    lemmatizer = WordNetLemmatizer()

    # SBERT model
    print('  Loading SBERT model...')
    sbert = SentenceTransformer(SBERT_MODEL)

    # UMAP reducer
    print('  Loading UMAP reducer...')
    with open(UMAP_REDUCER_PATH, 'rb') as f:
        reducer = pickle.load(f)

    # Cluster centroids
    print('  Loading cluster centroids...')
    centroids_dict = np.load(CENTROIDS_PATH, allow_pickle=True).item()
    cluster_ids = sorted(centroids_dict.keys())
    centroid_matrix = np.stack([centroids_dict[c] for c in cluster_ids])

    # Cluster keywords
    kw_df = pd.read_csv(KEYWORDS_PATH)
    kw_lookup = dict(zip(kw_df['cluster'], kw_df['keywords']))

    # Annotated reference reflections
    print('  Loading annotated reference reflections...')
    df_ref = pd.read_csv(ANNOTATED_PATH)

    # Reference embeddings
    print('  Loading reference embeddings...')
    ref_embeddings = np.load(EMBEDDINGS_PATH)

    # Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

    print('Resources loaded.\n')

    return {
        'lemmatizer':     lemmatizer,
        'sbert':          sbert,
        'reducer':        reducer,
        'cluster_ids':    cluster_ids,
        'centroid_matrix': centroid_matrix,
        'kw_lookup':      kw_lookup,
        'df_ref':         df_ref,
        'ref_embeddings': ref_embeddings,
        'client':         client,
    }

def normalise_bloom_to_grade(bloom_weighted: float,
                              config: str = S2_NORM_CONFIG) -> float:
    import math
    params       = S2_NORM_PARAMS[config]
    rubric_blend = params['rubric_blend']
    bloom_range  = params['max_bloom'] - params['min_bloom']

    # Linear
    linear = (bloom_weighted - params['min_bloom']) / bloom_range * 10
    linear = max(0.0, min(params['grade_cap'], linear))

    # Rubric interpolation
    lower = max(1, min(5, math.floor(bloom_weighted)))
    upper = max(1, min(5, math.ceil(bloom_weighted)))
    fraction = bloom_weighted - lower
    rubric = (BLOOM_TO_GRADE[lower] if lower == upper else
              BLOOM_TO_GRADE[lower] + fraction * (BLOOM_TO_GRADE[upper] - BLOOM_TO_GRADE[lower]))

    return round((1 - rubric_blend) * linear + rubric_blend * rubric, 2)

def compute_bloom_penalty(bloom_weighted: float, s1_band: str) -> float:
    """
    Penalises S1 when actual Bloom deviates from expected range for S1 band.
    
    Returns multiplier in [0, 1]:
    - 1.0 = Bloom within expected range, no penalty
    - approaches 0 = Bloom far below expected, heavy penalty
    """
    low, high = EXPECTED_BLOOM[s1_band]
    
    if bloom_weighted >= low:
        # Within or above expected range — no penalty
        return 1.0
    
    # Below expected range — penalise proportionally
    # How far below the lower bound are we?
    deficit = low - bloom_weighted  # e.g. expected 3.5, got 1.5 → deficit = 2.0
    bloom_range = 5.0 - 1.0        # full Bloom range
    
    # Penalty scales with deficit as proportion of full range
    penalty = 1.0 - (deficit / bloom_range)
    return max(0.1, penalty)  # floor at 0.1 — never fully zero

# ─────────────────────────────────────────────
# STEP 1 — PREPROCESS
# ─────────────────────────────────────────────

def preprocess_target(sections: dict, lemmatizer: WordNetLemmatizer) -> dict:
    """Normalise and lemmatise all 5 sections."""
    preprocessed = {}
    for section in SECTIONS:
        text = sections.get(section, '')
        text = normalise(str(text))
        text = lemmatise(text, lemmatizer)
        preprocessed[section] = text
    return preprocessed


# ─────────────────────────────────────────────
# STEP 2 — EMBED
# ─────────────────────────────────────────────

def embed_target(preprocessed: dict, sbert: SentenceTransformer) -> np.ndarray:
    """
    Embed all 5 preprocessed sections and mean aggregate.
    Consistent with offline embedding pipeline.
    """
    section_embeddings = []
    for section in SECTIONS:
        text = preprocessed.get(section, '')
        emb = sbert.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True
        )[0]
        section_embeddings.append(emb)
    return np.mean(section_embeddings, axis=0)


# ─────────────────────────────────────────────
# STEP 3 — ASSIGN CLUSTER
# ─────────────────────────────────────────────

def assign_cluster(embedding: np.ndarray,
                   reducer,
                   cluster_ids: list,
                   centroid_matrix: np.ndarray,
                   kw_lookup: dict) -> dict:
    """
    Transform embedding via UMAP and assign to nearest cluster centroid.
    Returns cluster assignment and confidence.
    """
    # UMAP transform — use saved reducer, do NOT refit
    reduced = reducer.transform(embedding.reshape(1, -1))

    # Cosine similarity to all centroids
    sims = cosine_similarity(embedding.reshape(1, -1), centroid_matrix)[0]
    best_idx = np.argmax(sims)
    best_cluster = cluster_ids[best_idx]
    best_sim = sims[best_idx]

    # Confidence based on similarity threshold
    if best_sim < CLUSTER_CONFIDENCE_THRESHOLD:
        cluster_confidence = 'low'
        flag_low_confidence = True
    else:
        cluster_confidence = 'high'
        flag_low_confidence = False

    return {
        'cluster':            best_cluster,
        'cluster_similarity': float(best_sim),
        'cluster_confidence': cluster_confidence,
        'flag_low_confidence': flag_low_confidence,
        'keywords':           kw_lookup.get(best_cluster, 'No keywords available'),
    }


# ─────────────────────────────────────────────
# STEP 4 — SIGNAL 1
# ─────────────────────────────────────────────

def compute_signal1(embedding: np.ndarray,
                    cluster: int,
                    df_ref: pd.DataFrame,
                    ref_embeddings: np.ndarray,
                    k: int = DEFAULT_K) -> dict:
    """
    Weighted average grade from top-k most similar neighbours in assigned cluster.
    Weight proportional to cosine similarity.
    """
    # Filter reference reflections to assigned cluster
    cluster_mask = df_ref['cluster'] == cluster
    cluster_indices = df_ref[cluster_mask].index.tolist()

    if len(cluster_indices) == 0:
        return {
            'signal1_grade': None,
            'signal1_n_neighbours': 0,
            'signal1_neighbours': [],
            'signal1_flag': True,
        }

    # Compute cosine similarity to all cluster members
    cluster_embeddings = ref_embeddings[cluster_indices]
    sims = cosine_similarity(embedding.reshape(1, -1), cluster_embeddings)[0]

    # Top-k by similarity
    top_k = min(k, len(cluster_indices))
    top_k_idx = np.argsort(sims)[::-1][:top_k]

    # Weighted average grade
    top_sims = sims[top_k_idx]
    top_grade_indices = [cluster_indices[i] for i in top_k_idx]
    top_grades = df_ref.iloc[top_grade_indices]['grade'].values

    # Double check weights and grades are aligned
    assert len(top_sims) == len(top_grades) == top_k
    weights = top_sims / top_sims.sum()

    signal1_grade = float(np.dot(weights, top_grades))

    neighbours = [
        {
            'student_id': df_ref.iloc[top_grade_indices[j]]['student_id'],
            'grade':      float(top_grades[j]),
            'similarity': float(top_sims[j]),
            'weight':     float(weights[j]),
        }
        for j in range(top_k)
    ]

    return {
        'signal1_grade':        signal1_grade,
        'signal1_n_neighbours': top_k,
        'signal1_neighbours':   neighbours,
        'signal1_flag':         False,
        'signal1_avg_similarity': float(np.mean(top_sims)),
        'signal1_cluster_size': len(sims)
    }


# ─────────────────────────────────────────────
# STEP 5 — SIGNAL 2
# ─────────────────────────────────────────────
def compute_signal2(raw_sections: dict,
                    topic: str,
                    keywords: str,
                    client,
                    config: str = S2_NORM_CONFIG) -> dict:

    annotation = annotate_bloom(
        client=client,
        topic=topic,
        keywords=keywords,
        result=raw_sections.get('result', ''),
        pfr=raw_sections.get('pfr', ''),
        learning=raw_sections.get('learning', ''),
        model=BLOOM_MODEL
    )

    if not annotation['success']:
        return {
            'signal2_bloom_weighted': None,
            'signal2_normalised':     None,
            'signal2_result_level':   None,
            'signal2_pfr_level':      None,
            'signal2_learning_level': None,
            'signal2_justifications': {},
            'signal2_flag':           True,
            'signal2_error':          annotation.get('error'),
        }

    r_level = annotation['result']['level']
    p_level = annotation['pfr']['level']
    l_level = annotation['learning']['level']

    bloom_weighted = (
        S2_RESULT_WEIGHT   * r_level +
        S2_PFR_WEIGHT      * p_level +
        S2_LEARNING_WEIGHT * l_level
    )

    # Normalise to grade scale based on config
    params     = S2_NORM_PARAMS[config]
    bloom_range = params['max_bloom'] - params['min_bloom']
    signal2_normalised = normalise_bloom_to_grade(bloom_weighted, config)
    # signal2_normalised = (bloom_weighted - params['min_bloom']) / bloom_range * 10
    # signal2_normalised = float(max(0.0, min(params['grade_cap'], signal2_normalised)))

    return {
        'signal2_bloom_weighted': round(float(bloom_weighted), 2),
        'signal2_normalised':     round(signal2_normalised, 2),
        'signal2_result_level':   r_level,
        'signal2_pfr_level':      p_level,
        'signal2_learning_level': l_level,
        'signal2_justifications': {
            'result':   annotation['result']['justification'],
            'pfr':      annotation['pfr']['justification'],
            'learning': annotation['learning']['justification'],
        },
        'signal2_flag': False,
    }

# ─────────────────────────────────────────────
# STEP 6 — CONFIDENCE GATE
# ─────────────────────────────────────────────

def get_band(value: float, signal: str, config: str = S2_NORM_CONFIG) -> str:
    if signal == 's1':
        low_max, high_min = S1_LOW_MAX, S1_HIGH_MIN
    else:
        thresholds = S2_BAND_THRESHOLDS[config]
        low_max  = thresholds['low_max']
        high_min = thresholds['high_min']

    if value <= low_max:
        return 'low'
    elif value >= high_min:
        return 'high'
    return 'mid'

def get_band_s1(grade: float, avg_similarity: float) -> str:
    """
    S1 band adjusted continuously by average neighbour similarity.
    High similarity preserves or upgrades band confidence.
    Low similarity downgrades band.
    """
    # Adjust grade by similarity confidence
    # similarity 1.0 → no change
    # similarity 0.5 → grade scaled by 0.75
    # similarity 0.0 → grade scaled by 0.5

    sim_scalar = 0.5 + 0.5 * avg_similarity
    adjusted_grade = grade * sim_scalar

    if adjusted_grade <= S1_LOW_MAX:
        return 'low'
    elif adjusted_grade >= S1_HIGH_MIN:
        return 'high'
    return 'mid'


def confidence_gate(disparity: float,
                    avg_similarity: float,
                    cluster_size: int,
                    flag_low_confidence: bool,
                    disparity_threshold: float = 3.0,
                    similarity_threshold: float = 0.6,
                    cluster_size_threshold: int = 15) -> dict:

    """
    Confidence based purely on uncertainty signals.
    Not on S1/S2 band combinations.
    """
    flags = []

    if disparity > disparity_threshold:
        flags.append('high_disparity')
    if avg_similarity < similarity_threshold:
        flags.append('low_similarity')
    if cluster_size < cluster_size_threshold:
        flags.append('small_cluster')
    # Flag 4 — cluster assignment uncertain
    if flag_low_confidence:
        flags.append('low_cluster_confidence')

    # Confidence based on number of flags
    n_flags = len(flags)
    if n_flags == 0:
        confidence = 'high'
        flag_for_review = False
    elif n_flags == 1:
        confidence = 'medium'
        flag_for_review = False
    else:
        confidence = 'low'
        flag_for_review = True

    return {
        'confidence':      confidence,
        'flag_for_review': flag_for_review,
        'flags':           flags,
    }
# ─────────────────────────────────────────────
# STEP 7 — FINAL GRADE
# ─────────────────────────────────────────────

def compute_final_grade(signal1_grade: float,
                        signal2_normalised: float,
                        bloom_penalty: float,
                        disparity_penalty: float,
                        disparity: float) -> dict:

    adjusted_s1 = signal1_grade * bloom_penalty #removed disparity penalty
    final_grade = S1_WEIGHT * adjusted_s1 + S2_WEIGHT * signal2_normalised
    final_grade = float(max(0.0, min(10.0, final_grade)))

    return {
        'final_grade':       round(final_grade, 2),
        'bloom_penalty':     round(bloom_penalty, 3),
        'disparity':         round(disparity, 2),
        'disparity_penalty': round(disparity_penalty, 3),
        'adjusted_s1':       round(adjusted_s1, 2),
        'flag_disparity':    disparity > 2.5,
    }
# ─────────────────────────────────────────────
# MAIN ASSESSMENT FUNCTION
# ─────────────────────────────────────────────

def assess_reflection(situation: str,
                      task_action: str,
                      result: str,
                      pfr: str,
                      learning: str,
                      topic: str = 'Unknown',
                      resources: dict = None,
                      k: int = DEFAULT_K,
                      alpha: float = 0.15,
                      min_multiplier: float = 0.5,
                      s1_weight: float = S1_WEIGHT,
                      disparity_threshold: float = 3.0,
                      similarity_threshold: float = 0.6,
                      cluster_size_threshold: float = 15,
                      **kwargs) -> dict:
    """
    Full online assessment pipeline for a single target reflection.

    Args:
        situation, task_action, result, pfr, learning: raw section text
        topic:     reflection topic (optional)
        resources: preloaded offline artifacts from load_resources()
        k:         number of neighbours for Signal 1

    Returns:
        dict with all signals, confidence, final grade, and feedback
    """
    if resources is None:
        resources = load_resources()

    raw_sections = {
        'situation':   situation,
        'task_action': task_action,
        'result':      result,
        'pfr':         pfr,
        'learning':    learning,
    }

    # Step 1 — Preprocess
    preprocessed = preprocess_target(raw_sections, resources['lemmatizer'])

    # Step 2 — Embed
    embedding = embed_target(preprocessed, resources['sbert'])

    # Step 3 — Assign cluster
    cluster_result = assign_cluster(
        embedding,
        resources['reducer'],
        resources['cluster_ids'],
        resources['centroid_matrix'],
        resources['kw_lookup'],
    )

    # Step 4 — Signal 1
    s1_result = compute_signal1(
        embedding,
        cluster_result['cluster'],
        resources['df_ref'],
        resources['ref_embeddings'],
        k=k,
    )

    # Step 5 — Signal 2
    s2_result = compute_signal2(
        raw_sections,
        topic,
        cluster_result['keywords'],
        resources['client'],
    )

    # Handle failures
    if s1_result['signal1_flag'] or s2_result['signal2_flag']:
        return {
            **cluster_result,
            **s1_result,
            **s2_result,
            'final_grade':    None,
            'confidence':     'low',
            'flag_for_review': True,
            'feedback':       cluster_result['keywords'],
        }

        # Get s1_band first — only needs raw S1 and similarity
    s1_band = get_band_s1(s1_result['signal1_grade'], 
                        float(s1_result['signal1_avg_similarity']))

    # Compute bloom penalty using s1_band
    bloom_penalty = compute_bloom_penalty(
        s2_result['signal2_bloom_weighted'], 
        s1_band
    )

    # Compute disparity
    disparity = abs(s1_result['signal1_grade'] - s2_result['signal2_normalised'])
    disparity_penalty = 1 / (1 + alpha * disparity)  # use alpha param

    grade_result = compute_final_grade(
        s1_result['signal1_grade'],
        s2_result['signal2_normalised'],
        bloom_penalty,
        disparity_penalty,
        disparity,
    )

    # Step 6 — Confidence gate
    gate = confidence_gate(
        disparity=disparity,
        avg_similarity=float(s1_result['signal1_avg_similarity']),
        cluster_size=s1_result['signal1_cluster_size'],
        flag_low_confidence=cluster_result['flag_low_confidence'],
        disparity_threshold=disparity_threshold,        
        similarity_threshold=similarity_threshold,      
        cluster_size_threshold=cluster_size_threshold,  
    )
    final_grade = grade_result['final_grade']

    flag_for_review = gate['flag_for_review']

    return {
        # Cluster
        'assigned_cluster':    cluster_result['cluster'],
        'cluster_keywords':    cluster_result['keywords'],
        'cluster_similarity':  cluster_result['cluster_similarity'],
        'cluster_confidence':  cluster_result['cluster_confidence'],

        # Signal 1
        'signal1_grade':       s1_result['signal1_grade'],
        'signal1_n_neighbours': s1_result['signal1_n_neighbours'],
        'signal1_neighbours':  s1_result['signal1_neighbours'],

        # Signal 2
        'signal2_bloom_weighted': s2_result['signal2_bloom_weighted'],
        'signal2_result_level':   s2_result['signal2_result_level'],
        'signal2_pfr_level':      s2_result['signal2_pfr_level'],
        'signal2_learning_level': s2_result['signal2_learning_level'],
        'signal2_justifications': s2_result['signal2_justifications'],

        # Gate
        # 's1_band':        gate['s1_band'],
        # 's2_band':        gate['s2_band'],
        'confidence':     gate['confidence'],
        'flag_for_review': flag_for_review,

        # Final
        'feedback':    cluster_result['keywords'],
        'final_grade':        final_grade,
        # 'signal2_normalised': signal2_normalised,
        # 'norm_config':        norm_config,
    }

if __name__ == '__main__':
    resources = load_resources()

    TOLERANCE = 1.5

    print("\n" + "=" * 70)
    print("RUNNING REFLECTION ASSESSMENT VALIDATION")
    print("=" * 70)

    correct = 0
    need_review = 0

    for ex in EXAMPLE_REFLECTIONS:

        result = assess_reflection(
            **ONLINE_PARAMETERS,
            situation=ex["situation"],
            task_action=ex["task_action"],
            result=ex["result"],
            pfr=ex["pfr"],
            learning=ex["learning"],
            topic=ex["topic"],
            resources=resources
        )

        review_flag = result['flag_for_review'] == True

        if review_flag:
            need_review += 1

        predicted = result["final_grade"]
        expected = ex["grade"]

        difference = abs(predicted - expected)

        confirmed = difference <= TOLERANCE

        if confirmed:
            correct += 1

        print("\n" + "-" * 70)
        print(f"Example ID:        {ex['id']}")
        print(f"Topic:             {ex['topic']}")
        print(f"Expected grade:    {expected}/10")
        print(f"Predicted grade:   {predicted}/10")
        print(f"Difference:        {difference:.2f}")
        print(f"Confirmed:         {'YES' if confirmed else 'NO'}")

        print("\nModel Details:")
        print(f"  Cluster:         {result['assigned_cluster']}")
        print(f"  Similarity:      {result['cluster_similarity']:.3f}")
        print(f"  S1 Grade:        {result['signal1_grade']:.2f}")
        print(f"  S2 Bloom:        {result['signal2_bloom_weighted']:.2f}")
        print(f"  Confidence:      {result['confidence']}")
        print(f"  Review Flag:     {result['flag_for_review']}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    accuracy = correct / len(EXAMPLE_REFLECTIONS) * 100

    print(f"Confirmed within tolerance: {correct}/{len(EXAMPLE_REFLECTIONS)}")
    print(f"Needs review: {need_review}/{len(EXAMPLE_REFLECTIONS)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Tolerance used: ±{TOLERANCE}")