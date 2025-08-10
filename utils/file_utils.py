import logging
import os

logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file, save_dir="uploads") -> str:
    logger.info(f"Saving uploaded file: {uploaded_file.name}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        logger.info(f"Created directory: {save_dir}")

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File saved at {file_path}")
    return file_path
