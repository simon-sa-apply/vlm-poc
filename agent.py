import argparse
import logging
import sys
from pathlib import Path

from src.config import init_vertex_ai, DEFAULT_N_RESULTS
from src.indexer import run_indexing_pipeline
from src.embeddings import embed_image
from src.vector_store import get_collection, query_similar

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[TqdmLoggingHandler()]
)
logger = logging.getLogger(__name__)

def handle_index(args):
    init_vertex_ai()
    run_indexing_pipeline(args.image_dir)

def handle_query(args):
    init_vertex_ai()
    
    image_path = Path(args.image)
    if not image_path.exists() or not image_path.is_file():
        logger.error(f"Reference image does not exist: {args.image}")
        sys.exit(1)
        
    try:
        collection = get_collection()
        if collection.count() == 0:
            logger.error("ChromaDB store is empty. Please run 'index' first.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load ChromaDB store: {e}")
        sys.exit(1)
        
    try:
        query_embedding = embed_image(str(image_path))
    except Exception as e:
        logger.error(f"Failed to embed reference image: {e}")
        sys.exit(1)
        
    results = query_similar(collection, query_embedding, args.n_results)
    
    print("\n" + "="*40)
    print("SIMILARITY SEARCH RESULTS")
    print("="*40)
    for res in results:
        similarity = 1.0 - res["distance"]
        print(f"Rank {res['rank']}: {res['path']}")
        print(f"  Category: {res['category']}")
        print(f"  Similarity: {similarity:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="VLM Image Similarity Search Agent")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")
    
    index_parser = subparsers.add_parser("index", help="Run the full indexing pipeline")
    index_parser.add_argument("--image-dir", required=True, help="Directory containing images to index")
    
    query_parser = subparsers.add_parser("query", help="Search for similar images")
    query_parser.add_argument("--image", required=True, help="Path to the reference image")
    query_parser.add_argument("--n-results", type=int, default=DEFAULT_N_RESULTS, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)

if __name__ == "__main__":
    main()
