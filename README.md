# TRUTHLens

**Ask 5 AI models at once. Trust only what they all agree on.**

TRUTHLens is an AI trust-intelligence platform that analyzes answers from multiple large language models and determines which claims are actually reliable. Instead of relying on a single AI response, TRUTHLens aggregates, analyzes, and verifies outputs from multiple models to produce a safer synthesized answer.

---

# Inspiration

AI models often present answers with high confidence regardless of whether the information is correct. This creates a trust problem: a single model cannot reliably judge the truthfulness of its own output.

TRUTHLens addresses this by querying multiple independent models simultaneously and comparing their responses. If multiple models independently agree and the claims are supported by evidence, the confidence in those claims increases. If they disagree or cannot be verified, the system flags them.

The goal is to move from **single-model authority** to **multi-model verification**.

---

# What It Does

TRUTHLens analyzes AI responses through a multi-stage trust pipeline.

## Multi-Model Consensus Analysis

Five large language models respond to the same prompt simultaneously:

- GPT-4  
- Gemini  
- Claude  
- Llama 3  
- Kimi  

Each model response is broken into **atomic claims**. These claims are then:

1. Embedded into vector space  
2. Clustered based on semantic similarity  
3. Compared across models  

Claims are scored on three axes:

- **Agreement** – how many models independently produce the same claim  
- **Evidence Verification** – whether external evidence supports the claim  
- **Model Independence** – whether agreement appears coincidental or derivative

---

## Trust Scoring

Each claim receives a composite trust score and is labeled as one of the following:

- **Verified** – supported by evidence and multi-model agreement  
- **Unverified** – plausible but lacking evidence  
- **Rejected** – contradicted by evidence  
- **Subjective** – opinion-based or interpretive  
- **Neutral** – informational but not verifiable

---

## Safe Answer Synthesis

Instead of selecting a single model's output, TRUTHLens synthesizes a final answer from the **highest-confidence claim clusters** produced by the trust scoring stage.

This produces a response that is grounded in consensus and verification rather than individual model authority.

---

# Visualization Modes

TRUTHLens includes three visualization systems to help users understand the trust pipeline.

## Claim Constellation (3D)

Claims are projected into 3D space using **UMAP dimensionality reduction**.  
Clusters of similar claims appear as constellations, allowing users to visually see consensus groups.


## Trust Matrix

A 2D scatter plot mapping:

- **X-axis:** Model Consensus  
- **Y-axis:** Verification Confidence  

Claims fall into **Safe Zones** or **Danger Zones**, visually indicating reliability.

---

# How We Built It

## Frontend

- Designed in **Figma**
- Built with **React**
- Real-time visualization of claim clusters and pipeline stages
- Uses **UMAP projection** to map high-dimensional embeddings into interactive 3D visualizations

## Backend

- **FastAPI orchestration server**
- Coordinates model calls and analysis pipeline
- Uses **WebSockets** to stream analysis stages back to the frontend in real time

## Machine Learning Pipeline

Runs on **Modal serverless GPUs**.

Pipeline stages include:

### Claim Extraction
GPT-4o-mini extracts structured atomic claims from model outputs.

### Embedding
Claims are embedded using **bge-large-en**.

### Semantic Clustering
Claims with similar meaning are grouped into clusters.

---

## Verification

Truth validation uses **DeBERTa-v3**, a Natural Language Inference (NLI) model.

It evaluates whether:

- evidence **supports** the claim
- evidence **contradicts** the claim
- evidence is **neutral**

This provides a mathematical basis for claim scoring.

---

# Challenges We Ran Into

### Multi-Model Output Normalization

Each LLM outputs responses in different formats. We built a deterministic extraction layer to convert responses into structured atomic claims before analysis.

### 3D Visualization Performance

Mapping high-dimensional embeddings into interactive 3D clusters required careful optimization to prevent UI lag while maintaining visual accuracy.

### Visualization Layout

Translating the mathematical UMAP space into a readable UI layout required custom scaling and positioning logic for claim nodes.

---

# Accomplishments We're Proud Of

- Built a full pipeline from **React → FastAPI → Concurrent LLM APIs → Modal GPU ML pipeline**
- Developed a **mathematical trust heuristic** combining consensus and textual entailment
- Implemented real-time streaming analysis of claims
- Created interactive visualizations that make AI reliability interpretable

---

# What We Learned

**Visualization is as important as the model.**  
The value of a trust system depends on users understanding how conclusions were reached.

**Consensus alone is not enough.**  
Multiple models agreeing does not guarantee correctness. Independence and verification are essential.

---

# What's Next for TRUTHLens

Future development will focus on expanding the trust-analysis ecosystem.

Planned features include:

- Additional visualization systems
- **Knowledge Decks** for structured claim summaries
- **Evidence Mapping** linking claims to verifiable sources
- Improved independence detection between models
- Larger model ensembles

---

# TRUTHLens

**Ask multiple models.  
Verify their claims.  
Trust the consensus.**

