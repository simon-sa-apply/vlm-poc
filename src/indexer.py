import os
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from src.config import SUPPORTED_EXTENSIONS, THROTTLE_DELAY, CHECKPOINT_FILE
from src.vlm import categorize_image
from src.embeddings import embed_image
from src.vector_store import get_collection, generate_record_id, upsert_image

logger = logging.getLogger(__name__)

def get_mime_type(ext: str) -> str:
    ext = ext.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    elif ext == ".webp":
        return "image/webp"
    return "application/octet-stream"

def scan_directory(image_dir: str) -> list[Path]:
    dir_path = Path(image_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Directory does not exist: {image_dir}")
    
    supported_files = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                supported_files.append(file_path)
            else:
                logger.warning(f"Skipping unsupported file format: {file_path}")
    
    supported_files.sort()
    logger.info(f"Discovered {len(supported_files)} supported images in {image_dir}")
    if not supported_files:
        raise ValueError(f"No supported images found in {image_dir}")
    return supported_files

def load_checkpoint() -> set[str]:
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def append_checkpoint(filename: str):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(f"{filename}\n")

def process_single_image(file_path: Path, collection) -> bool:
    filename = file_path.name
    path_str = str(file_path)
    mime_type = get_mime_type(file_path.suffix)
    
    try:
        with open(file_path, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        logger.error(f"Failed to read image {path_str}: {e}")
        return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_vlm = executor.submit(categorize_image, image_bytes, mime_type, path_str)
        future_emb = executor.submit(embed_image, path_str)
        
        try:
            category_result = future_vlm.result()
            category = category_result["category"] if category_result else "unknown"
        except Exception as e:
            logger.warning(f"VLM failed for {path_str}, defaulting to 'unknown'. Error: {e}")
            category = "unknown"
            
        try:
            embedding = future_emb.result()
        except Exception as e:
            logger.error(f"Embedding failed for {path_str}. Skipping image. Error: {e}")
            return False

    record_id = generate_record_id(filename)
    try:
        upsert_image(collection, record_id, embedding, category, path_str, filename)
        append_checkpoint(filename)
        return True
    except Exception as e:
        logger.error(f"Failed to upsert {path_str} to ChromaDB: {e}")
        return False

def run_indexing_pipeline(image_dir: str):
    start_time = time.time()
    files = scan_directory(image_dir)
    processed_files = load_checkpoint()
    
    collection = get_collection()
    
    success_count = 0
    fail_count = 0
    
    to_process = [f for f in files if f.name not in processed_files]
    skip_count = len(files) - len(to_process)
    
    if skip_count > 0:
        logger.info(f"Skipping {skip_count} already processed images.")
        
    with tqdm(total=len(to_process), desc="Indexing Images", unit="img") as pbar:
        for file_path in to_process:
            success = process_single_image(file_path, collection)
            if success:
                success_count += 1
            else:
                fail_count += 1
            
            pbar.update(1)
            time.sleep(THROTTLE_DELAY)
            
    elapsed_time = time.time() - start_time
    avg_latency = (elapsed_time * 1000) / (success_count + fail_count) if (success_count + fail_count) > 0 else 0
    
    print("\n" + "="*40)
    print("INDEXING COMPLETION SUMMARY")
    print("="*40)
    print(f"Total images processed (success): {success_count}")
    print(f"Total images failed:              {fail_count}")
    print(f"Total images skipped (checkpoint):{skip_count}")
    print(f"Average latency per image:        {avg_latency:.2f} ms")
    print(f"Total elapsed time:               {elapsed_time:.2f} s")
    print("="*40)
    
    if fail_count > 0:
        logger.warning(f"Pipeline completed with {fail_count} failures.")
