# undergrad-thesis

An NLP pipeline that automatically grades open-ended student engineering reflections (STAR + PFR/Learning format) by combining two signals: a weighted K-nearest-neighbour grade from a cluster of semantically similar past reflections (Signal 1), and a Bloom's taxonomy annotation produced by an LLM (Signal 2). The two signals are fused through a confidence gate to produce a final grade and review flag.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file at the project root with your Anthropic API key (required for Phase 3):

```
ANTHROPIC_API_KEY=your-key-here
```

## How to run

Run all phases in order from the project root:

```bash
python run.py
```

Or run individual phases manually:

```bash
python phases/__preprocess/phase0_data_prep.py   # data prep
python phases/offline/phase1_embed.py             # embedding
python phases/offline/phase2_cluster.py           # clustering
python phases/online/phase0_main_grade.py         # grading
```

## Pipeline overview

| Phase | Script | What it does |
|-------|--------|-------------|
| 0 | `phases/__preprocess/phase0_data_prep.py` | Loads raw Excel dataset, filters valid reflections, extracts and preprocesses the 5 STAR sections |
| 1 | `phases/offline/phase1_embed.py` | Embeds all preprocessed sections using SBERT (`all-MiniLM-L6-v2`) |
| 2 | `phases/offline/phase2_cluster.py` | Runs UMAP + HDBSCAN to cluster the reference corpus; saves centroids and UMAP reducer for online use |
| 3 | `phases/online/phase0_main_grade.py` | Assesses example reflections through the full two-signal pipeline and prints predicted vs expected grades |

## Output locations

| Output | Location |
|--------|----------|
| Extracted sections | `data/extracted_sections.csv` |
| Preprocessed sections | `data/preprocessed_sections.csv` |
| Section embeddings | `data/embeddings/all/all-MiniLM-L6-v2/` |
| Clustered reflections | `data/clusters/clustered_reflections.csv` |
| Cluster centroids | `data/clusters/cluster_centroids.npy` |
| Cluster keywords | `data/clusters/cluster_keywords.csv` |
| UMAP reducer | `data/clusters/umap_reducer.pkl` |
| LOOCV evaluation results | `data/evaluation/loocv_results.csv` |
