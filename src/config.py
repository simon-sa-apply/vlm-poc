import os
from dotenv import load_dotenv
import vertexai

load_dotenv()

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID")
if not GCP_PROJECT_ID:
    raise KeyError("GCP_PROJECT_ID environment variable is required.")

GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

CHROMA_PERSIST_PATH = "./chroma_store"
CHROMA_COLLECTION_NAME = "image_pool"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_N_RESULTS = 3
THROTTLE_DELAY = 0.5
CHECKPOINT_FILE = ".checkpoint"

def init_vertex_ai():
    vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
