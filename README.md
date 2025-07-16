# docdot

A simple Python tool to extract headings and document structure from PDF files.

## Usage

1. Place your PDF file in the `inputs/` directory.
2. Run:
   ```bash
   python heading_detector.py inputs/yourfile.pdf
   ```
3. The output will show the detected title and outline (headings) from the document.

## Files
- `extractor.py`: Extracts text blocks from PDF.
- `heading_detector.py`: Detects headings and title from extracted blocks.
- `inputs/`: Place your PDF files here.
- `outputs/`: Output files (JSON, etc.) are saved here.

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
# docdot
