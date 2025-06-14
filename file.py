import fitz
import io
import os
import base64
import tempfile
import requests
import logging
from urllib.parse import urlparse
from PIL import Image
import cv2
import numpy as np

logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_to_tempfile(url: str) -> str:
    logging.info(f"Downloading file from URL: {url}")
    resp = requests.get(url)
    resp.raise_for_status()
    parsed = urlparse(url)
    suffix = os.path.splitext(parsed.path)[1] or ''
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(resp.content)
    tmp.close()
    logging.info(f"File downloaded and saved to: {tmp.name}")
    return tmp.name

def get_text_percentage(file_name: str) -> float:
    logging.info(f"Calculating text percentage for PDF: {file_name}")
    total_page_area = 0.0
    total_text_area = 0.0
    doc = fitz.open(file_name)
    for page in doc:
        total_page_area += abs(page.rect)
        for b in page.get_text("blocks"):
            r = fitz.Rect(b[:4])
            total_text_area += abs(r)
    doc.close()
    perc = total_text_area / total_page_area if total_page_area else 0
    logging.info(f"Text percentage: {perc:.4f}")
    return perc

def image_enhancement(image_bytes: bytes) -> bytes:
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 4
    )
    kernel = np.ones((1, 1), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(opened, kernel, iterations=1)
    final = cv2.bitwise_not(dilated)

    _, processed_bytes = cv2.imencode('.jpg', final)
    return processed_bytes.tobytes()

def extract_images_base64(file_path: str):
    is_temp = False
    if file_path.lower().startswith(("http://", "https://")):
        local_path = download_to_tempfile(file_path)
        is_temp = True
    else:
        local_path = file_path

    _, ext = os.path.splitext(local_path)
    ext = ext.lower()
    base64_images = []

    try:
        if ext in {'.jpg', '.jpeg', '.png'}:
            logging.info("Processing image file...")
            with open(local_path, 'rb') as img_f:
                img_bytes = img_f.read()
                Image.open(io.BytesIO(img_bytes)).verify()
                processed = image_enhancement(img_bytes)
                b64 = base64.b64encode(processed).decode('utf-8')
                base64_images.append(b64)

        elif ext == '.pdf':
            text_perc = get_text_percentage(local_path)
            pdf_document = fitz.open(local_path)

            if text_perc < 0.5:
                # Image-based PDF (scanned): render to image    
                logging.info("Processing image-based PDF...")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("jpeg")
                    Image.open(io.BytesIO(img_bytes)).verify()
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    base64_images.append(b64)
            else:
                # Text-based PDF (searchable): render to image
                logging.info("Processing text-based PDF...")
                for page_num in range(pdf_document.page_count):
                    page = pdf_document.load_page(page_num)
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("jpeg")
                    Image.open(io.BytesIO(img_bytes)).verify()
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    base64_images.append(b64)

            pdf_document.close()
            logging.info(f"Extracted and processed {len(base64_images)} images from PDF.")

        else:
            raise ValueError(f"Unsupported file type: '{ext}'.")

    finally:
        if is_temp:
            try:
                os.unlink(local_path)
                logging.info(f"Temporary file deleted: {local_path}")
            except OSError:
                logging.warning(f"Failed to delete temp file: {local_path}")

    return base64_images
