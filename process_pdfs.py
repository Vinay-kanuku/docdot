# main.py - Updated to use improved heading detection
import os
import json
import time
from pathlib import Path

# Import the improved system
try:
    from heading_detector import ImprovedHeadingDetector, extract_enhanced_text_blocks
    USE_IMPROVED = True
except ImportError:
    # Fallback to original system
    from extractor import extract_text_blocks
    from heading_detector import detect_headings
    USE_IMPROVED = False

def process_pdf(pdf_path: str) -> dict:
    """Process a single PDF with improved heading detection"""
    try:
        start_time = time.time()
        
        if USE_IMPROVED:
            # Use improved system
            blocks = extract_enhanced_text_blocks(pdf_path)
            detector = ImprovedHeadingDetector()
            result = detector.detect_headings(blocks)
        else:
            # Use original system
            blocks = extract_text_blocks(pdf_path)
            result = detect_headings(blocks)
        
        processing_time = time.time() - start_time
        print(f"Processed {os.path.basename(pdf_path)} in {processing_time:.2f}s")
        
        return result
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return {"title": f"Error processing {os.path.basename(pdf_path)}", "outline": []}

def main():
    """Main processing function"""
    # Handle both Docker and local environments
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check for Docker environment first
    if os.path.exists("/app/input"):
        input_dir = "/app/input"
        output_dir = "/app/output"
    else:
        # Local environment
        input_dir = os.path.join(script_dir, "inputs")
        output_dir = os.path.join(script_dir, "outputs")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    if os.path.exists(input_dir):
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print(f"Using {'improved' if USE_IMPROVED else 'original'} heading detection")
    
    for filename in pdf_files:
        pdf_path = os.path.join(input_dir, filename)
        
        # Process the PDF
        result = process_pdf(pdf_path)
        
        # Generate output filename
        output_filename = f"{Path(filename).stem}.json"
        output_path = os.path.join(output_dir, output_filename)
        
        # Write JSON output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved: {output_filename}")
        
        # Debug output for comparison
        print(f"Title: {result['title']}")
        print(f"Found {len(result['outline'])} headings:")
        for heading in result['outline'][:5]:  # Show first 5
            print(f"  {heading['level']}: {heading['text'][:50]}... (Page {heading['page']})")
        if len(result['outline']) > 5:
            print(f"  ... and {len(result['outline']) - 5} more")
        print()
    
    print(f"All files processed. Output saved to {output_dir}")

if __name__ == "__main__":
    main()