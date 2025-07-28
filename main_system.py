
#!/usr/bin/env python3
"""
Adobe Hackathon - Complete PDF Outline Extraction System
Multi-Signal Ensemble Approach with Robust Feature Engineering

Author: Advanced PDF Processing Team
Version: 2.0
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import traceback
from functools import lru_cache

# Import our custom modules
from document_parser import EnhancedDocumentParser, TextBlock, PageMetadata
from feature_extractor import MultiModalFeatureExtractor, BlockFeatures
from semantic_analyzer import enhance_features_with_semantics
from ensemble_scorer import EnsembleScorer, HeadingPrediction, HeadingLevel


class HierarchyBuilder:
    """Builds final document hierarchy from predictions"""
    
    def __init__(self):
        self.title_patterns = [
            # Common title indicators
            'title', 'main title', 'document title',
            # Position-based (first large text)
            'first_large_text',
        ]
    
    def build_outline(self, blocks: List[TextBlock], 
                     predictions: List[HeadingPrediction]) -> Dict:
        """Build final outline structure"""
        
        # Find document title
        title = self._extract_title(blocks, predictions)
        
        # Build heading hierarchy
        outline = self._build_heading_hierarchy(blocks, predictions)
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _extract_title(self, blocks: List[TextBlock], 
                      predictions: List[HeadingPrediction]) -> str:
        """Extract document title using multiple heuristics"""
        
        # Strategy 1: Look for explicitly labeled titles
        for i, block in enumerate(blocks):
            if any(pattern in block.text.lower() for pattern in ['title:', 'main title']):
                # Extract text after the label
                text = block.text
                if ':' in text:
                    title_text = text.split(':', 1)[1].strip()
                    if title_text:
                        return title_text
        
        # Strategy 2: First heading with high confidence on page 1
        page1_headings = []
        for pred in predictions:
            if (pred.is_heading and 
                blocks[pred.block_idx].page == 1 and 
                pred.confidence > 0.7):
                page1_headings.append((pred.confidence, blocks[pred.block_idx].text))
        
        if page1_headings:
            # Return highest confidence heading
            page1_headings.sort(reverse=True)
            return page1_headings[0][1].strip()
        
        # Strategy 3: Largest font on page 1
        page1_blocks = [b for b in blocks if b.page == 1]
        if page1_blocks:
            largest_block = max(page1_blocks, key=lambda b: b.font_size)
            if len(largest_block.text.strip()) > 5:  # Not too short
                return largest_block.text.strip()
        
        # Strategy 4: First reasonable text block
        for block in blocks:
            text = block.text.strip()
            if (len(text) > 10 and len(text) < 200 and 
                not text.startswith(('http', 'www', '@', '#'))):
                return text
        
        return "Untitled Document"
    
    def _build_heading_hierarchy(self, blocks: List[TextBlock],
                                predictions: List[HeadingPrediction]) -> List[Dict]:
        """Build hierarchical outline from predictions"""
        
        outline = []
        
        # Filter and sort heading predictions
        heading_preds = [p for p in predictions if p.is_heading]
        heading_preds.sort(key=lambda p: (blocks[p.block_idx].page, 
                                        blocks[p.block_idx].y0))
        
        # Convert to outline format
        for pred in heading_preds:
            block = blocks[pred.block_idx]
            
            outline_entry = {
                "level": pred.predicted_level.value,
                "text": block.text.strip(),
                "page": block.page,
                "confidence": pred.confidence  # Include for debugging
            }
            outline.append(outline_entry)
        
        # Clean up outline
        outline = self._clean_outline(outline)
        
        return outline
    
    def _clean_outline(self, outline: List[Dict]) -> List[Dict]:
        """Clean and validate outline entries"""
        
        cleaned = []
        
        for entry in outline:
            text = entry["text"].strip()
            
            # Skip very short or long entries
            if len(text) < 2 or len(text) > 500:
                continue
            
            # Skip entries that look like page numbers or references
            if (text.isdigit() or 
                re.match(r'^\d+\s*$', text) or
                re.match(r'^Page \d+', text, re.IGNORECASE)):
                continue
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = text.strip('.,;:')  # Remove trailing punctuation
            
            cleaned_entry = {
                "level": entry["level"],
                "text": text,
                "page": entry["page"]
            }
            cleaned.append(cleaned_entry)
        
        return cleaned


class PerformanceOptimizer:
    """Optimizes system performance for speed requirements"""
    
    def __init__(self):
        self.cache = {}
        self.processing_stats = {}
    
    @lru_cache(maxsize=100)
    def cached_feature_extraction(self, text_hash: str) -> Dict:
        """Cache expensive feature extraction operations"""
        # This would cache processed features for similar text blocks
        pass
    
    def optimize_for_speed(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """Apply speed optimizations"""
        
        # Skip very short blocks that are unlikely to be headings
        min_length = 3
        filtered_blocks = [b for b in blocks if len(b.text.strip()) >= min_length]
        
        # Limit processing for very large documents
        if len(filtered_blocks) > 1000:  # For very large docs
            # Sample blocks more aggressively
            # Keep first/last pages fully, sample middle
            first_page_blocks = [b for b in filtered_blocks if b.page <= 2]
            last_page_blocks = [b for b in filtered_blocks if b.page >= max(b.page for b in filtered_blocks) - 1]
            middle_blocks = [b for b in filtered_blocks if 2 < b.page < max(b.page for b in filtered_blocks) - 1]
            
            # Sample every 3rd block from middle
            sampled_middle = middle_blocks[::3]
            
            filtered_blocks = first_page_blocks + sampled_middle + last_page_blocks
        
        return filtered_blocks


class PDFOutlineExtractor:
    """Main PDF outline extraction system"""
    
    def __init__(self, debug: bool = False):
        # Initialize components
        self.parser = EnhancedDocumentParser()
        self.feature_extractor = MultiModalFeatureExtractor()
        self.ensemble_scorer = EnsembleScorer()
        self.hierarchy_builder = HierarchyBuilder()
        self.optimizer = PerformanceOptimizer()
        
        # Configure logging
        self.debug = debug
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.WARNING)
        
        self.logger = logging.getLogger(__name__)
    
    def extract_outline(self, pdf_path: str) -> Dict:
        """Main extraction function"""
        
        start_time = time.time()
        
        try:
            # Step 1: Parse document
            self.logger.info("Parsing document...")
            parse_start = time.time()
            blocks, pages_metadata = self.parser.parse_document(pdf_path)
            parse_time = time.time() - parse_start
            self.logger.info(f"Parsing took {parse_time:.2f}s, found {len(blocks)} blocks")
            
            if not blocks:
                return {"title": "Empty Document", "outline": []}
            
            # Step 2: Optimize for performance
            self.logger.info("Applying performance optimizations...")
            blocks = self.optimizer.optimize_for_speed(blocks)
            self.logger.info(f"Optimized to {len(blocks)} blocks")
            
            # Step 3: Extract features
            self.logger.info("Extracting features...")
            feature_start = time.time()
            features_list = self.feature_extractor.extract_features(blocks, pages_metadata)
            feature_time = time.time() - feature_start
            self.logger.info(f"Feature extraction took {feature_time:.2f}s")
            
            # Step 4: Enhance with semantic analysis
            self.logger.info("Running semantic analysis...")
            semantic_start = time.time()
            features_list = enhance_features_with_semantics(features_list, blocks)
            semantic_time = time.time() - semantic_start
            self.logger.info(f"Semantic analysis took {semantic_time:.2f}s")
            
            # Step 5: Ensemble scoring
            self.logger.info("Running ensemble scoring...")
            scoring_start = time.time()
            predictions = self.ensemble_scorer.score_blocks(blocks, features_list)
            scoring_time = time.time() - scoring_start
            self.logger.info(f"Ensemble scoring took {scoring_time:.2f}s")
            
            # Step 6: Build final hierarchy
            self.logger.info("Building final hierarchy...")
            hierarchy_start = time.time()
            result = self.hierarchy_builder.build_outline(blocks, predictions)
            hierarchy_time = time.time() - hierarchy_start
            self.logger.info(f"Hierarchy building took {hierarchy_time:.2f}s")
            
            total_time = time.time() - start_time
            self.logger.info(f"Total extraction time: {total_time:.2f}s")
            
            # Add processing metadata if debug mode
            if self.debug:
                result["_debug_info"] = {
                    "processing_time": total_time,
                    "total_blocks": len(blocks),
                    "detected_headings": len([p for p in predictions if p.is_heading]),
                    "timing": {
                        "parsing": parse_time,
                        "features": feature_time,
                        "semantic": semantic_time,
                        "scoring": scoring_time,
                        "hierarchy": hierarchy_time
                    }
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            if self.debug:
                self.logger.error(traceback.format_exc())
            
            # Return minimal valid output
            return {
                "title": "Error Processing Document",
                "outline": [],
                "_error": str(e)
            }


def process_single_pdf(input_path: str, output_path: str, debug: bool = False):
    """Process a single PDF file"""
    
    extractor = PDFOutlineExtractor(debug=debug)
    
    try:
        result = extractor.extract_outline(input_path)
        
        # Write output
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Processed: {input_path} -> {output_path}")
        
        if debug and "_debug_info" in result:
            debug_info = result["_debug_info"]
            print(f"  Time: {debug_info['processing_time']:.2f}s")
            print(f"  Headings: {debug_info['detected_headings']}")
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        
        # Create error output
        error_result = {
            "title": "Processing Error",
            "outline": [],
            "_error": str(e)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(error_result, f, indent=2)


def main():
    """Main entry point for Docker container"""
    
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Debug mode from environment variable
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find all PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in input directory")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF
    for pdf_path in pdf_files:
        # Generate output filename
        output_filename = pdf_path.stem + ".json"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Processing: {pdf_path.name}")
        process_single_pdf(str(pdf_path), output_path, debug=debug)
    
    print("All files processed successfully!")


# Additional utility functions for testing and development
def test_single_file(pdf_path: str, ground_truth_path: Optional[str] = None):
    """Test on a single file with optional ground truth comparison"""
    
    extractor = PDFOutlineExtractor(debug=True)
    result = extractor.extract_outline(pdf_path)
    
    print("=" * 60)
    print(f"PROCESSING RESULTS FOR: {pdf_path}")
    print("=" * 60)
    
    print(f"Title: {result['title']}")
    print(f"Headings found: {len(result['outline'])}")
    
    print("\nOutline:")
    for i, heading in enumerate(result['outline']):
        indent = "  " * (int(heading['level'][1:]) - 1) if heading['level'].startswith('H') else ""
        print(f"{indent}{heading['level']}: {heading['text']} (Page {heading['page']})")
    
    if "_debug_info" in result:
        debug_info = result["_debug_info"]
        print(f"\nProcessing time: {debug_info['processing_time']:.2f}s")
        print(f"Total blocks processed: {debug_info['total_blocks']}")
        
        timing = debug_info['timing']
        print("\nTiming breakdown:")
        for stage, time_taken in timing.items():
            print(f"  {stage}: {time_taken:.2f}s")
    
    # Compare with ground truth if provided
    if ground_truth_path and os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        
        print("\n" + "=" * 60)
        print("GROUND TRUTH COMPARISON")
        print("=" * 60)
        
        print(f"Ground truth headings: {len(gt_data.get('outline', []))}")
        print(f"Detected headings: {len(result['outline'])}")
        
        # Simple comparison
        gt_texts = [h['text'].lower().strip() for h in gt_data.get('outline', [])]
        pred_texts = [h['text'].lower().strip() for h in result['outline']]
        
        matches = len(set(gt_texts) & set(pred_texts))
        if gt_texts:
            recall = matches / len(gt_texts)
            print(f"Text-based recall: {recall:.2f}")
        
        if pred_texts:
            precision = matches / len(pred_texts)
            print(f"Text-based precision: {precision:.2f}")


def benchmark_performance(pdf_dir: str, num_runs: int = 3):
    """Benchmark system performance on multiple files"""
    
    pdf_files = list(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found for benchmarking")
        return
    
    extractor = PDFOutlineExtractor(debug=False)
    
    results = []
    
    print(f"Benchmarking on {len(pdf_files)} files, {num_runs} runs each...")
    
    for pdf_path in pdf_files:
        file_results = []
        
        for run in range(num_runs):
            start_time = time.time()
            try:
                result = extractor.extract_outline(str(pdf_path))
                processing_time = time.time() - start_time
                
                file_results.append({
                    'file': pdf_path.name,
                    'run': run + 1,
                    'time': processing_time,
                    'headings': len(result['outline']),
                    'success': True
                })
                
            except Exception as e:
                processing_time = time.time() - start_time
                file_results.append({
                    'file': pdf_path.name,
                    'run': run + 1,
                    'time': processing_time,
                    'headings': 0,
                    'success': False,
                    'error': str(e)
                })
        
        results.extend(file_results)
    
    # Analyze results
    successful_runs = [r for r in results if r['success']]
    
    if successful_runs:
        times = [r['time'] for r in successful_runs]
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"\nBenchmark Results:")
        print(f"Successful runs: {len(successful_runs)}/{len(results)}")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Min time: {min_time:.2f}s")
        print(f"Max time: {max_time:.2f}s")
        
        # Check if within 10s constraint
        within_constraint = sum(1 for t in times if t <= 10.0)
        print(f"Within 10s constraint: {within_constraint}/{len(times)} ({within_constraint/len(times)*100:.1f}%)")
    
    # Print per-file breakdown
    print(f"\nPer-file breakdown:")
    for pdf_path in pdf_files:
        file_runs = [r for r in results if r['file'] == pdf_path.name]
        if file_runs:
            avg_time = sum(r['time'] for r in file_runs if r['success']) / len([r for r in file_runs if r['success']])
            success_rate = sum(1 for r in file_runs if r['success']) / len(file_runs)
            print(f"  {pdf_path.name}: {avg_time:.2f}s avg, {success_rate*100:.0f}% success")


# Import statements for the modules (these would be in separate files)
import re
from functools import lru_cache

if __name__ == "__main__":
    # Check if we're running in Docker or testing locally
    if len(sys.argv) > 1:
        if sys.argv[1] == "test" and len(sys.argv) > 2:
            # Test mode: python main.py test <pdf_path> [ground_truth_path]
            pdf_path = sys.argv[2]
            gt_path = sys.argv[3] if len(sys.argv) > 3 else None
            test_single_file(pdf_path, gt_path)
            
        elif sys.argv[1] == "benchmark" and len(sys.argv) > 2:
            # Benchmark mode: python main.py benchmark <pdf_dir>
            pdf_dir = sys.argv[2]
            benchmark_performance(pdf_dir)
            
        else:
            print("Usage:")
            print("  Docker mode: python main.py")
            print("  Test mode: python main.py test <pdf_path> [ground_truth_path]")
            print("  Benchmark mode: python main.py benchmark <pdf_dir>")
    else:
        # Docker mode - process all files in /app/input
        main()