import os
import time

from src.logging_config import get_logger

logger = get_logger(__name__)


def cleanup_stale_clips(max_age_seconds: int = 3600):
  """Removes generated search clips older than the specified age

  (default: 1 hour) from the data directory to prevent disk bloat.
  """
  data_dir = "./data"
  if not os.path.exists(data_dir):
    return

  now = time.time()
  cleaned_count = 0

  for filename in os.listdir(data_dir):
    # Target only generated search clips, not source videos
    if "_clip_" in filename and filename.endswith(".mp4"):
      file_path = os.path.join(data_dir, filename)
      if os.path.isfile(file_path):
        if now - os.path.getmtime(file_path) > max_age_seconds:
          try:
            os.remove(file_path)
            cleaned_count += 1
          except Exception as e:
            logger.error("Failed to delete stale clip %s: %s", filename, str(e))

  if cleaned_count > 0:
    logger.info("Disk Lifecycle: Cleaned up %d stale search clips.", cleaned_count)