import os
import logging
import time

import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
MAX_WAIT = 300  # 5 minutes max
logger = logging.getLogger(__name__)


def wait_for_ollama() -> None:
    """Block until the Ollama server is reachable and the configured model is available."""
    logger.info("Waiting for Ollama at %s.", OLLAMA_HOST)
    start = time.time()

    while time.time() - start < MAX_WAIT:
        try:
            # Poll the Ollama tags endpoint until the service is up and the model is visible.
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                # Keep waiting until the exact model we plan to use has been pulled.
                if any(MODEL in m for m in models):
                    logger.info("Ollama is ready. Model %s is available.", MODEL)
                    return
                logger.info("Ollama is reachable, but model %s is not available yet.", MODEL)
        except requests.RequestException:
            logger.warning("Ollama is not ready yet. Retrying in 5 seconds.")
        time.sleep(5)

    raise RuntimeError(f"Ollama not ready after {MAX_WAIT}s. Exiting.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    wait_for_ollama()
