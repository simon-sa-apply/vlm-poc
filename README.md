# VLM Image Similarity Search Agent

A Proof of Concept (PoC) reactive agent that transforms a static pool of images into an intelligent, searchable visual knowledge base using Gemini 1.5 Flash 002 and Multimodal Embeddings.

## Prerequisites
- GCP project with Vertex AI API enabled
- Active billing account linked to the project
- `gcloud` CLI installed and authenticated (`gcloud auth application-default login`)
- Python 3.11+

## Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your `GCP_PROJECT_ID`.
4. Authenticate with Google Cloud:
   ```bash
   gcloud auth application-default login
   ```

## Usage

### Indexing
Run the full indexing pipeline on a directory of images:
```bash
python agent.py index --image-dir ./pool
```

### Querying
Search for similar images using a reference image:
```bash
python agent.py query --image ./ref.jpg --n-results 3
```
