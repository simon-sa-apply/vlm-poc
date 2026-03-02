import logging
import time
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt
from vertexai.vision_models import Image, MultimodalEmbeddingModel

logger = logging.getLogger(__name__)

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(f"Retrying Embedding call (attempt {retry_state.attempt_number}) due to error: {retry_state.outcome.exception()}")
)
def embed_image(image_path: str) -> list[float]:
    model_emb = MultimodalEmbeddingModel.from_pretrained("multimodalembedding@001")
    start_time = time.time()
    try:
        image_obj = Image.load_from_file(image_path)
        embeddings = model_emb.get_embeddings(image=image_obj, dimension=1408)
        latency = (time.time() - start_time) * 1000
        logger.info(f"Embedding call success for {image_path} | Latency: {latency:.2f}ms")
        return embeddings.image_embedding
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"Embedding call failed for {image_path} | Latency: {latency:.2f}ms | Error: {e}")
        raise
