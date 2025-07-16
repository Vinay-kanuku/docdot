# extractor.py
import fitz  # PyMuPDF

def extract_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
     
    all_blocks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
 
        for block in blocks:
            if block["type"] != 0:
                continue  # skip images
            for line in block["lines"]:
                line_text = ""
                font_size = None
                bbox = None
                for span in line["spans"]:
                    if font_size is None:
                        font_size = span["size"]
                        bbox = span["bbox"]
                    line_text += span["text"]
                text = line_text.strip()
                if text:
                    all_blocks.append({
                        "text": text,
                        "page": page_num + 1,
                        "font_size": font_size,
                        "bbox": bbox
                    })
    return all_blocks


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extractor.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    text_blocks = extract_text_blocks(pdf_path)
    
    for block in text_blocks:
        print(f"Page {block['page']}: {block['text']} (Font size: {block['font_size']}, BBox: {block['bbox']})")

