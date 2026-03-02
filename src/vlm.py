import json
import logging
import time
from tenacity import retry, wait_exponential, retry_if_exception_type, stop_after_attempt
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig

logger = logging.getLogger(__name__)

CATEGORIZATION_PROMPT = "Return the single primary visual category of this image."

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(f"Retrying VLM call (attempt {retry_state.attempt_number}) due to error: {retry_state.outcome.exception()}")
)
def categorize_image(image_bytes: bytes, mime_type: str = "image/jpeg", image_path: str = "unknown") -> dict | None:
    model = GenerativeModel("gemini-1.5-flash-002")
    image_part = Part.from_data(data=image_bytes, mime_type=mime_type)
    config = GenerationConfig(
        response_mime_type="application/json",
        response_schema={"type": "object", "properties": {"category": {"type": "string"}}},
        temperature=0.0,
    )
    
    start_time = time.time()
    try:
        response = model.generate_content(
            [image_part, CATEGORIZATION_PROMPT],
            generation_config=config,
        )
        latency = (time.time() - start_time) * 1000
        
        result = json.loads(response.text)
        if not isinstance(result, dict) or not result.get("category") or not isinstance(result["category"], str):
            logger.warning(f"Malformed VLM response for {image_path}: {response.text}")
            return None
            
        logger.info(f"VLM call success for {image_path} | Latency: {latency:.2f}ms")
        return result
    except Exception as e:
        latency = (time.time() - start_time) * 1000
        logger.error(f"VLM call failed for {image_path} | Latency: {latency:.2f}ms | Error: {e}")
        raise
