import os
import json
from extractor import extract_text_blocks
from heading_detector import detect_headings

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "inputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

def process_pdf(pdf_path):
    blocks = extract_text_blocks(pdf_path)
    result = detect_headings(blocks)
    return result

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".pdf"):
            continue
        pdf_path = os.path.join(INPUT_DIR, filename)
        result = process_pdf(pdf_path)
        output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Processed: {filename} → {output_path}")

if __name__ == "__main__":
    main()