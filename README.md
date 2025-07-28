# DocDot - PDF Document Structure Analyzer

DocDot is a tool for extracting and analyzing the structure of PDF documents. It detects document titles and headings hierarchically, enabling better document understanding and navigation.

## Features

- **Title Detection**: Automatically identifies document titles based on font sizes and positioning
- **Heading Detection**: Identifies hierarchical heading structure (H1, H2, H3, etc..)
- **Section Recognition**: Detects numbered sections and subsections
- **Structural Analysis**: Recognizes common document structures like "Table of Contents", "Introduction", etc.
- **Docker Support**: Containerized processing for consistent execution across environments
- **JSON Output**: Structured output format for easy integration with other systems

## How It Works

1. **Text Extraction**: The tool extracts text blocks from PDFs using PyMuPDF, preserving information about font sizes, positioning, and other metadata.

2. **Document Analysis**: The system analyzes font distribution, page structure, and text patterns to understand the document's structure.

3. **Heading Detection**: Multiple strategies are used to identify headings:
   - Font size-based detection
   - Explicit section numbering patterns (e.g., "1. Introduction", "2.1 Methods")
   - Recognition of common structural headings

4. **Output Generation**: Results are saved as JSON files with the document title and hierarchical outline.

## Requirements

- Python 3.10 or higher
- PyMuPDF 1.23.14
- Docker (optional, for containerized execution)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Vinay-kanuku/docdot.git
   cd docdot
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Local Execution

Process PDF files directly:

```bash
python process_pdfs.py
```

By default, the script looks for PDF files in an `inputs` directory and outputs JSON results to an `outputs` directory.

## Application Build and Run using Docker

### Build the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

### Run the Docker Container

```bash
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/pdfs:ro -v $(pwd)/sample_dataset/outputs:/app/outputs --network none pdf-processor
```

## Project Structure

- `extractor.py`: Basic text block extraction from PDFs
- `heading_detector.py`: Advanced heading detection and document structure analysis
- `process_pdfs.py`: Main processing script that handles PDF processing workflow
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration for Docker deployment

## Output Format

The tool generates a JSON file for each processed PDF with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 3
    },
    {
      "level": "H2",
      "text": "Background",
      "page": 4
    }
  ]
}
```