import os
import tempfile
import urllib.parse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
import httpx
import logging
from llm_ops import extractor_llm

logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

class ExtractRequest(BaseModel):
    file_url: HttpUrl
    rebate_program_number: str


@app.post("/extract-validate")
async def extract_endpoint(req: ExtractRequest):
    logging.info(f"Received extract request for program: {req.rebate_program_number}")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(str(req.file_url))
            resp.raise_for_status()
            content = resp.content
    except Exception as e:
        logging.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=400, detail=f"Error downloading file: {e}")

    parsed = urllib.parse.urlparse(str(req.file_url))
    suffix = os.path.splitext(parsed.path)[1] or ''
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    try:
        tmp_file.write(content)
        tmp_file.close()
        try:
            extracted = extractor_llm([tmp_file.name])
        except Exception as ee:
            logging.error(f"Extractor error: {ee}")
            raise HTTPException(status_code=500, detail=f"Extraction error: {ee}")
        logging.info(f"Extraction completed for Invoice: {req.rebate_program_number}")
    finally:
        try:
            os.unlink(tmp_file.name)
            logging.info(f"Temp file deleted: {tmp_file.name}")
        except OSError:
            logging.warning(f"Temp file deletion failed: {tmp_file.name}")

    return {
        "Invoice Extraction": extracted
    }


@app.post("/health-check")
async def health_check():
    """
    Health check endpoint to verify the service is running.
    """
    return {"status": "OK", "message": "Service is running"}