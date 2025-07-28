# improved_heading_detector.py
import re
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import statistics

@dataclass
class TextBlock:
    text: str
    page: int
    font_size: float
    bbox: Tuple[float, float, float, float]
    font_name: str = ""
    is_bold: bool = False
    line_height: float = 0.0
    x_offset: float = 0.0

class ImprovedHeadingDetector:
    def __init__(self):
        # Refined heading patterns with context awareness
        self.section_patterns = {
            'main_section': r'^(\d+)\.\s+(.+)$',          # "1. Introduction"
            'subsection': r'^(\d+)\.(\d+)\s+(.+)$',       # "2.1 Intended Audience"
            'subsubsection': r'^(\d+)\.(\d+)\.(\d+)\s+(.+)$',  # "2.1.1 Details"
        }
        
        # Patterns that should NOT be considered headings
        self.exclude_patterns = [
            r'^\d+\.\s+[A-Z][a-z]+.*\s+(who|that|which|and|or|in|on|at|to|from)',  # List items in sentences
            r'^\d+\.\s+.{100,}',  # Very long text (likely paragraph)
            r'^Page\s+\d+',       # Page numbers
            r'^©\s+',             # Copyright
            r'^Version\s+',       # Version info
        ]

        # Common structural headings
        self.structural_headings = [
            'table of contents', 'revision history', 'acknowledgements', 
            'references', 'appendix', 'introduction', 'conclusion',
            'abstract', 'summary', 'overview', 'background'
        ]

    def extract_document_structure(self, blocks: List[TextBlock]) -> Dict:
        """Analyze document structure to understand layout patterns"""
        # Group blocks by page for better analysis
        pages = defaultdict(list)
        for block in blocks:
            pages[block.page].append(block)
        
        # Find font patterns
        font_analysis = self._analyze_fonts(blocks)
        
        # Find page structure patterns
        structure_analysis = self._analyze_page_structure(pages)
        
        return {
            'font_analysis': font_analysis,
            'structure_analysis': structure_analysis,
            'total_pages': len(pages)
        }

    def _analyze_fonts(self, blocks: List[TextBlock]) -> Dict:
        """Enhanced font analysis with better clustering"""
        font_sizes = [b.font_size for b in blocks if b.text.strip()]
        
        if not font_sizes:
            return {'body_size': 12, 'size_levels': {}}
        
        # Use statistical analysis for better size detection
        size_counter = Counter(font_sizes)
        
        # Body text is the most frequent size
        body_size = max(size_counter.items(), key=lambda x: x[1])[0]
        
        # Create size hierarchy based on frequency and size
        unique_sizes = sorted(set(font_sizes), reverse=True)
        
        # Group sizes into levels more intelligently
        size_levels = {}
        for size in unique_sizes:
            ratio = size / body_size
            frequency = size_counter[size]
            
            # Assign level based on size ratio and frequency
            if ratio >= 1.4 and frequency <= len(blocks) * 0.1:  # Large, infrequent
                if size not in size_levels:
                    size_levels[size] = 'title'
            elif ratio >= 1.2 and frequency <= len(blocks) * 0.15:  # Medium-large, infrequent
                if size not in size_levels:
                    size_levels[size] = 'h1'
            elif ratio >= 1.1 and frequency <= len(blocks) * 0.2:   # Slightly larger, infrequent
                if size not in size_levels:
                    size_levels[size] = 'h2'
        
        return {
            'body_size': body_size,
            'size_levels': size_levels,
            'size_distribution': size_counter,
            'unique_sizes': unique_sizes
        }

    def _analyze_page_structure(self, pages: Dict) -> Dict:
        """Analyze page layout patterns"""
        first_page_blocks = pages.get(1, [])
        
        # Find potential title candidates on first page
        title_candidates = []
        for block in first_page_blocks:
            if (len(block.text.strip()) > 5 and 
                len(block.text.strip()) < 100 and
                not any(re.match(pattern, block.text) for pattern in self.exclude_patterns)):
                title_candidates.append(block)
        
        return {
            'title_candidates': title_candidates,
            'first_page_blocks': len(first_page_blocks)
        }

    def detect_title(self, blocks: List[TextBlock], doc_structure: Dict) -> str:
        """Improved title detection"""
        font_analysis = doc_structure['font_analysis']
        structure_analysis = doc_structure['structure_analysis']
        
        # Strategy 1: Look for title-level font on first page
        title_candidates = []
        
        for block in structure_analysis['title_candidates']:
            score = 0
            
            # Font size score
            if block.font_size in font_analysis['size_levels']:
                if font_analysis['size_levels'][block.font_size] == 'title':
                    score += 50
                elif font_analysis['size_levels'][block.font_size] == 'h1':
                    score += 30
            
            # Position score (higher on page = more likely title)
            if block.bbox and block.bbox[1] < 200:  # Top of page
                score += 20
            
            # Length score (reasonable title length)
            text_len = len(block.text.strip())
            if 10 <= text_len <= 80:
                score += 15
            elif text_len > 100:
                score -= 20
            
            # Content score
            text_lower = block.text.lower().strip()
            if any(word in text_lower for word in ['overview', 'introduction', 'guide', 'manual']):
                score += 10
            
            title_candidates.append((block, score))
        
        if title_candidates:
            # Get highest scoring candidate
            best_candidate = max(title_candidates, key=lambda x: x[1])
            if best_candidate[1] > 20:  # Minimum threshold
                return best_candidate[0].text.strip()
        
        # Fallback: Use largest font on first page
        page1_blocks = [b for b in blocks if b.page == 1 and b.text.strip()]
        if page1_blocks:
            title_block = max(page1_blocks, key=lambda b: b.font_size)
            return title_block.text.strip()
        
        return "Document Title"

    def classify_heading(self, block: TextBlock, doc_structure: Dict) -> Optional[Dict]:
        """Improved heading classification"""
        text = block.text.strip()
        
        # Skip if matches exclude patterns
        if any(re.match(pattern, text) for pattern in self.exclude_patterns):
            return None
        
        # Skip very long text (likely paragraphs)
        if len(text) > 150:
            return None
        
        font_analysis = doc_structure['font_analysis']
        
        # Check for explicit section numbering patterns
        for pattern_name, pattern in self.section_patterns.items():
            match = re.match(pattern, text)
            if match:
                # Additional validation for numbered sections
                if pattern_name == 'main_section':
                    section_num = int(match.group(1))
                    section_text = match.group(2).strip()
                    
                    # Validate it's not a list item
                    if (len(section_text) < 100 and 
                        not section_text.lower().startswith(('professionals who', 'people who', 'those who'))):
                        return {
                            'level': 'H1',
                            'text': text,
                            'page': block.page,
                            'confidence': 0.9,
                            'method': 'numbered_section'
                        }
                
                elif pattern_name == 'subsection':
                    return {
                        'level': 'H2',
                        'text': text,
                        'page': block.page,
                        'confidence': 0.85,
                        'method': 'numbered_subsection'
                    }
                
                elif pattern_name == 'subsubsection':
                    return {
                        'level': 'H3',
                        'text': text,
                        'page': block.page,
                        'confidence': 0.8,
                        'method': 'numbered_subsubsection'
                    }
        
        # Check for structural headings (Table of Contents, etc.)
        text_lower = text.lower()
        if text_lower in self.structural_headings:
            return {
                'level': 'H1',
                'text': text,
                'page': block.page,
                'confidence': 0.8,
                'method': 'structural_heading'
            }
        
        # Font-based classification (with more conservative thresholds)
        if block.font_size in font_analysis['size_levels']:
            font_level = font_analysis['size_levels'][block.font_size]
            confidence = 0.6  # Lower confidence for font-only detection
            
            # Boost confidence for reasonable heading characteristics
            if (block.is_bold or 
                'bold' in block.font_name.lower() or
                len(text) < 80):
                confidence += 0.1
            
            if font_level == 'h1':
                return {
                    'level': 'H1',
                    'text': text,
                    'page': block.page,
                    'confidence': confidence,
                    'method': 'font_size'
                }
            elif font_level == 'h2':
                return {
                    'level': 'H2',
                    'text': text,
                    'page': block.page,
                    'confidence': confidence,
                    'method': 'font_size'
                }
        
        return None

    def detect_headings(self, blocks: List[TextBlock]) -> Dict:
        """Main heading detection with improved logic"""
        if not blocks:
            return {"title": "", "outline": []}
        
        # Analyze document structure
        doc_structure = self.extract_document_structure(blocks)
        
        # Detect title
        title = self.detect_title(blocks, doc_structure)
        
        # Find all potential headings
        heading_candidates = []
        for block in blocks:
            if block.text.strip():
                heading_info = self.classify_heading(block, doc_structure)
                if heading_info:
                    heading_candidates.append(heading_info)
        
        # Filter and sort headings
        outline = self._filter_and_sort_headings(heading_candidates)
        
        return {
            "title": title,
            "outline": outline
        }

    def _filter_and_sort_headings(self, candidates: List[Dict]) -> List[Dict]:
        """Filter duplicates and false positives, then sort"""
        if not candidates:
            return []
        
        # Remove duplicates based on text similarity
        filtered = []
        seen_texts = set()
        
        # Sort by confidence first, then by page and position
        candidates.sort(key=lambda x: (-x['confidence'], x['page']))
        
        for candidate in candidates:
            text_normalized = re.sub(r'\s+', ' ', candidate['text'].lower().strip())
            
            # Skip if too similar to existing heading
            if not any(self._texts_similar(text_normalized, seen) for seen in seen_texts):
                seen_texts.add(text_normalized)
                
                # Remove confidence and method from output
                filtered_candidate = {
                    'level': candidate['level'],
                    'text': candidate['text'],
                    'page': candidate['page']
                }
                filtered.append(filtered_candidate)
        
        # Final sort by page, then by level hierarchy
        level_order = {'H1': 1, 'H2': 2, 'H3': 3}
        filtered.sort(key=lambda x: (x['page'], level_order.get(x['level'], 4)))
        
        return filtered

    def _texts_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two texts are similar enough to be considered duplicates"""
        if not text1 or not text2:
            return False
        
        # Simple similarity check
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return text1 == text2
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold

# Enhanced extractor with better metadata
def extract_enhanced_text_blocks(pdf_path: str) -> List[TextBlock]:
    """Enhanced text extraction with comprehensive metadata"""
    import fitz
    
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] != 0:  # Skip images
                continue
                
            for line in block["lines"]:
                line_text = ""
                font_size = None
                font_name = ""
                is_bold = False
                bbox = None
                
                for span in line["spans"]:
                    if font_size is None:
                        font_size = span.get("size", 12)
                        font_name = span.get("font", "")
                        # Enhanced bold detection
                        font_lower = font_name.lower()
                        is_bold = (span.get("flags", 0) & 2**4 != 0 or  # Bold flag
                                 any(bold_indicator in font_lower 
                                     for bold_indicator in ["bold", "black", "heavy", "semibold"]))
                        bbox = span.get("bbox", (0, 0, 0, 0))
                    line_text += span.get("text", "")
                
                text = line_text.strip()
                if text and font_size:
                    all_blocks.append(TextBlock(
                        text=text,
                        page=page_num + 1,
                        font_size=font_size,
                        bbox=bbox,
                        font_name=font_name,
                        is_bold=is_bold,
                        x_offset=bbox[0] if bbox else 0
                    ))
    
    doc.close()
    return all_blocks

# Updated detect_headings function for backward compatibility
def detect_headings(blocks):
    """Backward compatibility wrapper"""
    # Convert old format to new format
    text_blocks = []
    for block in blocks:
        text_blocks.append(TextBlock(
            text=block["text"],
            page=block["page"],
            font_size=block["font_size"],
            bbox=block.get("bbox", (0, 0, 0, 0)),
            font_name="",
            is_bold=False,
            x_offset=block.get("bbox", (0, 0, 0, 0))[0] if block.get("bbox") else 0
        ))
    
    detector = ImprovedHeadingDetector()
    return detector.detect_headings(text_blocks)