from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json
import re
from typing import List
import os
import logging
from file_process import extract_images_as_base64

load_dotenv()
logging.basicConfig(filename='logs/app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("OPENAI_KEY")
llm = ChatOpenAI(model="gpt-4o", api_key=api_key)


json_schema = {
  "Invoice": {
    "InvoiceData": {
      "InvoiceNumber": "",
      "InvoiceDate": ""
    },
    "Product": [
      {
        "ProductDescription": "",
        "Quantity": ""
      },
      {
        "ProductDescription": "",
        "Quantity": ""
      }
    ],
    "CustomerInformation": {
      "FirstName": "",
      "LastName": "",
      "Address": ""
    },
    "StoreInformation": {
      "StoreName": "",
      "Address": ""
    }
  }
}

def extractor_llm(image_paths: List[str]) -> dict:
    logging.info(f"Starting extractor_llm with paths: {image_paths}")
    if len(image_paths) != 1:
        raise ValueError("Exactly one PDF path should be provided.")
    pdf_path = image_paths[0]
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    base64_images = extract_images_as_base64(pdf_path)
    if not base64_images:
        raise ValueError("No images found in the provided PDF.")

    prompt = (
     """You are expert in reading images and extracting necessary details. You are given an image. Extract the necessary information from it 
        and populate the following JSON schema exactly, using the same keys.  Extract and populate correct details from image.\n
        Invoice Information:\n
        InvoiceNumber: It is the order number or invoice number mentioned on image. If not given then fill "NA"\n
        InvoiceDate: It is the invoice data on the invoice image.\n
        ProductDescription: This is description about Product id of tires purchased. Give exact details of product as given in invoice. Not about any kind service taken.\n
        Quantity: If quantity is directly not given for respective product then calculate it by dividing total amount of product with product price.\n
        
        Customer Information:\n
        FirstName: First Name of the customer/Advisor who puschaed product.\n
        LastName: Last Name of the customer/Advisor who puschaed product.\n
        Address: Fill with Address of customer only if available below customer name otherwise fill "NA".\n

        Store Information:  *IF INVOICE IS FROM TIRE RACK THEN STORE INFORMATION IS GIVEN ON THE TOP LEFT*\n
        StoreName: Store Name from where product is purchased. If not available on invoice fill with "NA". Do not fill with shipping adress name.\n
        Adress: Fill with Address of the store if available on the invoice otherwise fill "NA".  It will be avialalble below store Information of invoice otherwise fill with "NA". \n
        
        If in any invoice date is written like printed and written then get complete date. Like some time it is like Date: 20 ... and in ... someone enters 25 so it means 2025.\n
        Always remember on invoice first Store information is given and then after it customer information is given. Also, billing informatin and shipping information as related to customer only.\n
        
        *STRICTLY - Do not fill any field by yourself. Fill fields only if available on invoice.\n
        *NOTE - If there are more than one productdescription available then add details in the json for second, third etc. If only one then fill only one and return respective only.\n
        Respond **only** with valid JSON (no extra text).\n\n

        Your are given an example filled schema. Fill details like given example:\n
          {
            "Invoice": {
              "InvoiceData": {
                "InvoiceNumber": "abc123",
                "InvoiceDate": "05-06-2025"
              },
              "Product": [
                {
                  "ProductDescription": "A245/456 HK DYNPRO HTRH12*# 111222 113T",
                  "Quantity": "4"
                }
              ],
              "CustomerInformation": {
                "FirstName": "John",
                "LastName": "Adams",
                "Address": "5919 Main St, Springfield, IL 62704-6215"
              },
              "StoreInformation": {
                "StoreName": "Tire Rack",
                "Address": "710 Kendall St, South Bend, IN 46601-8222"
              }
            }
          }\n
         Respond **only** with valid JSON (no extra text).\n\n
         Schema:\n"""
         f"{json.dumps(json_schema, indent=2)}")

    content_list = [{"type": "text", "text": prompt}]
    ext = os.path.splitext(pdf_path)[1].lower().lstrip('.')
    mime = f"image/{'jpeg' if ext in ('jpg','jpeg') else ext}"

    for b64 in base64_images:
        content_list.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    try:
        message = HumanMessage(content=content_list)
        response = llm.invoke([message])
        print("*********LLM response received.*******************")
        print(response)
        raw = response.content.strip()
        logging.info(f"LLM raw response: {raw}")
    except Exception as e:
        logging.error(f"Error during LLM invoke: {e}")
        raise

    m = re.search(r'(\{[\s\S]*\})', raw)
    if not m:
        raise ValueError("No JSON object found in model response:\n" + raw)

    try:
        parsed = json.loads(m.group(1))
        logging.info("Parsed JSON successfully.")
        return parsed
    except Exception as je:
        logging.error(f"JSON parsing error: {je}")
        raise