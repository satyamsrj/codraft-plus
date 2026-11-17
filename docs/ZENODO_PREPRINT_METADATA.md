# Zenodo Preprint Deposit Metadata Package

This document contains the exact metadata and deposit fields for uploading the CoDraft+ research paper to [Zenodo](https://zenodo.org/deposit/new).

---

### Basic Metadata

- **Resource Type:** `Publication` $\rightarrow$ `Preprint`
- **Publication Date:** `2026-11-17` (Today's date)
- **Title:** `CoDraft+: Adaptive Multi-Model Collaboration through a Shared Canonical Workspace for Resource-Constrained Edge Devices`
- **Authors:**
  - Family name: `Saroj`
  - Given names: `Satyam`
  - Affiliation: `Ajay Kumar Garg Engineering College, Ghaziabad, India`
  - ORCID: *(Add if available)*
- **License:** `Open Access` $\rightarrow$ `Apache License 2.0` (matching repository license)
- **Primary Language:** `English (eng)`

---

### Description / Abstract (Copy & Paste)

```markdown
Consider two resource-constrained edge devices tasked with generating a rigorous technical report: deploying a monolithic 70-billion-parameter language model triggers immediate out-of-memory (OOM) faults, while conventional multi-agent debate floods limited context windows with repetitive conversational transcripts, squandering up to 60% of inference compute on conversational pleasantries. In this paper, we introduce CoDraft+, an adaptive multi-model collaborative framework engineered for consumer-grade laptops and edge hardware with strictly less than 8 GB of resident RAM. Departing from dialogue-based debate and uncoordinated ensemble voting, CoDraft+ operationalizes human-inspired collaborative writing: heterogeneous compact models (e.g., Qwen-2.5-3B and Gemma-3-1B, or sub-50M parameter pairs) iteratively construct, refine, and cross-verify one single evolving canonical document within an embedded, sectioned workspace. The framework integrates: (1) a structured shared workspace backed by SQLite in Write-Ahead Logging (WAL) mode with ACID versioning, (2) a 4-tier resilient delta-patching engine eliminating syntax parse crashes on small-model outputs, (3) a grounded Evidence Ledger providing atomic claim extraction and attribution tracking, and (4) an Adaptive Collaboration Controller that routes simple queries through an instant single-pass fast-path while dynamically halting complex collaborative iterations upon semantic stagnation or full claim resolution. Comprehensive empirical evaluation across 11 standard benchmarks demonstrates that CoDraft+ achieves superadditive collaboration gains: pairing a 14M and 31M model yields an +79.7% relative improvement in ROUGE-L and +29.4% in Token F1 over a monolithic 1.58x larger model (70M), while maintaining a strictly lower peak resident memory footprint (1234.1 MB vs. 1250.1 MB).
```

---

### Keywords & Indexing Terms

```text
Collaborative Multi-LLM, Edge Computing, Shared Canonical Workspace, Delta Patching, Evidence Ledger, Memory-Constrained AI, Adaptive Controller, Consumer Hardware, Small Language Models
```

---

### Related Identifiers (Link to GitHub & Artifacts)

- **Identifier 1:**
  - Relation: `isSupplementTo`
  - Identifier: `https://github.com/satyamsrj/codraft-plus`
  - Resource type: `Software`
- **Identifier 2:**
  - Relation: `isDocumentedBy`
  - Identifier: `https://github.com/satyamsrj/codraft-plus/blob/main/paper_assets/experimental_details_and_evaluation_record.md`
  - Resource type: `Technical Note`

---

### Files to Upload to Zenodo

1. `CoDraft_Plus_Research_Paper.pdf` (Compiled from `paper/main.tex` or exported from `paper/CODRAFT_PLUS_RESEARCH_PAPER.md`)
2. `references.bib`
3. Optional supplementary zip: `paper_assets/figures/`
