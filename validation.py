from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import json
import re
from typing import Dict
import os
import logging
from guidelines import guidelines

load_dotenv()
guidelines = guidelines

api_key = os.getenv("OPENAI_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)


def validator_llm(extracted_data: Dict, guidelines_rules: str = guidelines) -> Dict[str, str]:
    # 1) Load guidelines text
    if not os.path.exists(guidelines_rules):
        raise FileNotFoundError(f"Guidelines file not found: {guidelines_rules}")
    with open(guidelines_rules, "r", encoding="utf-8") as gf:
        guidelines_text = gf.read().strip()
    if not guidelines_text:
        raise ValueError("Guidelines file is empty")

    # 2) Build prompt
    extracted_json_str = json.dumps(extracted_data, indent=2)
    prompt = (
        "You are an expert validator for invoice extraction. You are given:\n"
        "1) Eligible products, date ranges, and expected quantity rules in plain text.\n"
        "2) The extracted invoice data as JSON.\n\n"
        "Read the rules carefully and check:\n"
        "1. Product description: If it matches any eligible product, return “True” or “False”. It can be possible eligible product name may lie in middle, or in end in the product description. If it is matching with eligile product then return matched\n"
        "2. Purchase date: If it falls within the given promotion date range, return “True” else “False”.\n"
        "3. Quantity: Expected value is 4. If exactly 4, return “True” else “False”.\n\n"
        "Respond ONLY with a JSON object EXACTLY in this format (no extra keys/text):\n"
        "{\n"
        '  "Product description": "True or "False",\n'
        '  "purchase_date": "True" or "False",\n'
        '  "quantity": "True" or "False"\n'
        "}\n\n"
        "Rules:\n"
        f"{guidelines_text}\n\n"
        "Extracted Data:\n"
        f"{extracted_json_str}\n"
    )

    # 3) Send to LLM
    message = HumanMessage(content=[{"type": "text", "text": prompt}])
    response = llm.invoke([message])
    raw = response.content.strip()

    # 4) Extract JSON
    m = re.search(r"(\{[\s\S]*\})", raw)
    if not m:
        raise ValueError("No JSON object found in validator LLM response:\n" + raw)
    result_json_str = m.group(1)
    try:
        result = json.loads(result_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON from validator LLM response: {e}\nResponse was:\n{result_json_str}")

    # 5) Preserve the matched-with-percentage strings
    expected_keys = ["Product description", "purchase_date", "quantity"]
    normalized = {}
    for key in expected_keys:
        if key not in result:
            raise ValueError(f"Key '{key}' missing in validator response JSON: {result}")
        val = str(result[key]).strip()
        # Optionally: validate val matches pattern like "(matched|not matched) with \d+%".
        normalized[key] = val
    return normalized