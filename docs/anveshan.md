# Hidden Relationship Discovery -- Prototype v1

A baseline, fully-explainable implementation of the six-stage pipeline:
evidence integration -> entity resolution -> temporal heterogeneous graph
construction -> hidden-relationship discovery -> evidence-path explanation ->
audit/integrity logging. Built against `synthetic_investigation_dataset/`
(see the dataset's own metadata for its structure).

## Setup

```
pip install -r requirements.txt
```

Edit `DATASET_ROOT` at the top of `data_loader.py` if your dataset lives
somewhere other than the default path.

## Run everything

```
python3 run_pipeline.py
```

This runs all six stages and writes:
- `pair_features.csv` -- engineered cross-source features for every person pair
- `scored_pairs.csv` -- every pair with its model score, split, and label
- `explanations.json` -- full evidence-path explanations for top findings
- `audit_log.jsonl` -- hash-chained log of every inference + evidence fingerprints
- `MODEL_REPORT.md` -- evaluation metrics, feature importances, worked examples
- `CASE_010_evidence_graph.png` -- example multi-layer evidence visualization

## Module map

| File | Responsibility |
|---|---|
| `data_loader.py` | Loads all entities/evidence/documents; loads ground truth ONLY under an explicitly separate method, never used by feature/model code. |
| `entity_resolution.py` | Resolves aliases/nicknames/initials/handles/vehicle-plate fragments/org mentions in free text back to candidate person_ids, with a confidence score per candidate (deliberately ambiguous -- multiple candidates per alias is the expected, realistic output). Per-method confidence weights were tightened based on measured accuracy -- see `ENTITY_RESOLUTION_EVAL.md`. |
| `graph_builder.py` | Builds the temporal heterogeneous multigraph and four person-level evidence layers: `comm_graph`, `financial_graph`, `colocation_graph`, `vehicle_graph` (plus `org_graph`). Every edge carries the exact evidence record IDs behind it. |
| `features.py` | For every candidate person pair: direct-edge indicators, shortest-path distance per layer, common neighbors, shared org/home location, text co-mention score, incident co-presence, and a cross-source corroboration count. |
| `embeddings.py` | Optional v2 feature source: node2vec embeddings on the combined person-to-person evidence graph, added as cosine-similarity/dot/L2 pair features. See `EMBEDDING_COMPARISON.md` for why these are NOT used in the shipped model. |
| `model.py` | Case-safe train/validation/test split construction (a person in a held-out case's network never contributes a training label) + XGBoost classifier + ranking/AUC evaluation broken out by hidden vs explicit and by split. |
| `compare_v1_v2.py` | Trains/evaluates the classifier on v1 (hand-engineered) vs v2 (+ embeddings) features under identical splits/seeds; writes `EMBEDDING_COMPARISON.md`. |
| `explain.py` | For any pair, walks the same graphs the model scored to produce the actual supporting path(s) per layer -- the explanation IS the evidence, not a post-hoc approximation. |
| `audit.py` | SHA-256 fingerprints of the evidence files at run time + an append-only, hash-chained log of every inference (which evidence record IDs supported it). `verify_chain()` detects any tampering after the fact. |
| `query_demo.py` | Investigator-facing CLI for the demo: `top` (ranked candidate list), `explain <a> <b>` (full evidence-path readout), `person <id>` (dossier). Run this instead of opening CSVs live. |
| `entity_resolution_eval.py` | Eval-only: scores the resolver against the dataset generator's internal alias ground truth (never touched by the model). |
| `run_pipeline.py` | Orchestrates the full v1 pipeline end-to-end and writes the final report. |
| `neo4j_export.py`, `neo4j_loader_driver.py` | Export/load the same evidence into a real Neo4j graph database -- see `README_NEO4J.md`. |

## Current status / known limitations (read before presenting)

- **Approach**: feature-engineered graph + gradient-boosted classifier, not a
  GNN. Chosen because the dataset has only 26 hidden-relationship positive
  examples total -- too few to train a deep model that generalizes rather
  than memorizes. A GNN (with this same explainability layer bolted on top)
  is a natural v2.
- **What works well**: the model reliably distinguishes TRUE from FALSE/
  coincidental relationships (all 10 designed "false decoy" pairs score in
  the bottom half of the score distribution), and correctly finds several
  held-out (never-seen-in-training) hidden relationships via multi-hop
  communication/financial/co-location/vehicle reasoning -- see
  `MODEL_REPORT.md` for the worked examples with real evidence-record IDs.
- **What's weaker**: hidden relationships that depend on the hardest
  "indirect-via-intermediary-only" pattern (no co-location scheduled at all,
  only a 2-hop communication chain) score inconsistently -- some rank in the
  top 1% of all pairs, others don't clear a useful threshold. This is an
  honest, reportable limitation, not a bug: it's exactly the "hard" tier the
  dataset was designed to include.
- **Entity resolution** is heuristic (regex + fuzzy string match + a few
  corroborating signals), not a trained NER/resolution model. Measured
  directly against internal ground truth (`ENTITY_RESOLUTION_EVAL.md`):
  77% recall, 25% precision@1 overall, with initials/nickname matching
  (~50% precision@1) far more reliable than single-letter matching
  (~10%, no better than chance) -- weights were retuned accordingly.
- **Graph embeddings were tried and rejected for the shipped model**: node2vec
  embeddings added as extra features made every held-out metric flat or
  worse (TEST ROC-AUC 0.984 -> 0.969, average precision 0.683 -> 0.670) --
  see `EMBEDDING_COMPARISON.md`. This is a measured result, not a guess,
  and it confirms the original reasoning below: 26 hidden positives isn't
  enough signal for a learned graph representation to beat the
  hand-engineered path-length/common-neighbor features here. Worth
  revisiting if the dataset is scaled up substantially.
- **Not yet built**: a persistent web UI (the CLI in `query_demo.py` covers
  the demo use case) and incremental/streaming graph updates.
