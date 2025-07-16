
import re
from collections import Counter

def detect_headings(blocks):
    # Find the title: largest font on page 1
    page1_blocks = [b for b in blocks if b["page"] == 1]
    title_block = max(page1_blocks, key=lambda b: b["font_size"])
    title = title_block["text"]

    # Collect unique font sizes document-wide
    sizes = [b["font_size"] for b in blocks]
    size_counts = Counter(sizes)
    sorted_sizes = sorted(size_counts.keys(), reverse=True)

 
    # Map sizes to levels dynamically
    size_to_level = {}
    if len(sorted_sizes) >= 4:
        size_to_level[sorted_sizes[1]] = "H1"
        size_to_level[sorted_sizes[2]] = "H2"
        size_to_level[sorted_sizes[3]] = "H3"
    elif len(sorted_sizes) >= 3:
        size_to_level[sorted_sizes[1]] = "H1"
        size_to_level[sorted_sizes[2]] = "H2"

    outline = []
    for b in blocks:
        text = b["text"]
        page = b["page"]
        size = b["font_size"]

        level = None
        # Numbering pattern check
        if re.match(r"^\d+\.\s", text):
            level = "H1"
        elif re.match(r"^\d+\.\d+\s", text):
            level = "H2"
        elif re.match(r"^\d+\.\d+\.\d+\s", text):
            level = "H3"
        elif size in size_to_level:
            level = size_to_level[size]

        if level and len(text) < 200:
            outline.append({
                "level": level,
                "text": text,
                "page": page
            })

    return {
        "title": title,
        "outline": outline
    }



if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python heading_detector.py <pdf_path>")
        sys.exit(1)
    
    from extractor import extract_text_blocks
    pdf_path = sys.argv[1]
    text_blocks = extract_text_blocks(pdf_path)
    
    headings = detect_headings(text_blocks)
    
    # print(f"Title: {headings['title']}")
    # for item in headings["outline"]:
    #     print(f"{item['level']}: {item['text']} (Page {item['page']})")





