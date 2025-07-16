import re
import json
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import unicodedata
import statistics
from functools import lru_cache

class HeadingLevel(Enum):
    TITLE = "TITLE"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"

@dataclass
class TextBlock:
    text: str
    page: int
    font_size: float
    bbox: Tuple[float, float, float, float]
    is_bold: bool = False
    is_italic: bool = False
    font_name: str = ""
    
    @property
    def x(self) -> float:
        return self.bbox[0]
    
    @property
    def y(self) -> float:
        return self.bbox[1]
    
    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

@dataclass
class HeadingCandidate:
    block: TextBlock
    confidence: float
    level: HeadingLevel
    features: Dict[str, float]
    
class DocumentAnalyzer:
    """Analyzes document structure and layout patterns"""
    
    def __init__(self, blocks: List[TextBlock]):
        self.blocks = blocks
        self.page_stats = self._compute_page_stats()
        self.font_stats = self._compute_font_stats()
        self.layout_patterns = self._analyze_layout_patterns()
    
    def _compute_page_stats(self) -> Dict[int, Dict]:
        stats = {}
        for page_num in set(b.page for b in self.blocks):
            page_blocks = [b for b in self.blocks if b.page == page_num]
            stats[page_num] = {
                'block_count': len(page_blocks),
                'avg_font_size': np.mean([b.font_size for b in page_blocks]),
                'font_sizes': [b.font_size for b in page_blocks],
                'left_margin': min(b.x for b in page_blocks),
                'right_margin': max(b.x + b.width for b in page_blocks),
                'top_margin': min(b.y for b in page_blocks),
                'bottom_margin': max(b.y + b.height for b in page_blocks)
            }
        return stats
    
    def _compute_font_stats(self) -> Dict:
        all_sizes = [b.font_size for b in self.blocks]
        size_counter = Counter(all_sizes)
        
        return {
            'sizes': sorted(set(all_sizes), reverse=True),
            'size_counts': size_counter,
            'most_common_size': size_counter.most_common(1)[0][0],
            'body_text_size': self._estimate_body_text_size(size_counter),
            'size_percentiles': {
                'p95': np.percentile(all_sizes, 95),
                'p90': np.percentile(all_sizes, 90),
                'p75': np.percentile(all_sizes, 75),
                'p50': np.percentile(all_sizes, 50)
            }
        }
    
    def _estimate_body_text_size(self, size_counter: Counter) -> float:
        """Estimate body text size using heuristics"""
        # Most common size is likely body text
        most_common = size_counter.most_common(3)
        
        # Filter out very small sizes (footnotes, captions)
        candidates = [size for size, count in most_common if size > 8]
        
        if candidates:
            return candidates[0]
        return most_common[0][0] if most_common else 12.0
    
    def _analyze_layout_patterns(self) -> Dict:
        """Analyze document layout patterns"""
        left_alignments = defaultdict(int)
        font_size_positions = defaultdict(list)
        
        for block in self.blocks:
            # Track left alignment patterns
            left_alignments[round(block.x, 1)] += 1
            
            # Track font size vs position
            font_size_positions[block.font_size].append(block.x)
        
        # Find common left margins
        common_margins = sorted(left_alignments.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'common_left_margins': [margin for margin, count in common_margins],
            'font_size_positions': dict(font_size_positions),
            'has_consistent_margins': len(common_margins) > 0 and common_margins[0][1] > len(self.blocks) * 0.3
        }

class TextPatternAnalyzer:
    """Analyzes text patterns that indicate headings"""
    
    # Comprehensive heading patterns
    HEADING_PATTERNS = [
        # Numbered patterns
        (r'^\d+\.?\s+', 'numbered_section', 0.9),
        (r'^\d+\.\d+\.?\s+', 'numbered_subsection', 0.85),
        (r'^\d+\.\d+\.\d+\.?\s+', 'numbered_subsubsection', 0.8),
        
        # Roman numerals
        (r'^[IVX]+\.?\s+', 'roman_numeral', 0.8),
        (r'^[ivx]+\.?\s+', 'roman_numeral_lower', 0.7),
        
        # Alphabetic patterns
        (r'^[A-Z]\.?\s+', 'alpha_upper', 0.7),
        (r'^[a-z]\.?\s+', 'alpha_lower', 0.6),
        
        # Special patterns
        (r'^Chapter\s+\d+', 'chapter', 0.95),
        (r'^Section\s+\d+', 'section', 0.9),
        (r'^Part\s+\d+', 'part', 0.95),
        (r'^Appendix\s+[A-Z]', 'appendix', 0.9),
        
        # Question patterns
        (r'^\d+\.\s*[A-Z].*\?', 'question', 0.8),
        
        # All caps (short)
        (r'^[A-Z\s]{2,30}$', 'all_caps_short', 0.7),
        
        # Bracketed numbers
        (r'^\[\d+\]', 'bracketed_number', 0.6),
        (r'^\(\d+\)', 'parenthesized_number', 0.6),
    ]
    
    # Semantic heading indicators
    SEMANTIC_INDICATORS = {
        'introduction': 0.8,
        'conclusion': 0.8,
        'abstract': 0.9,
        'summary': 0.8,
        'methodology': 0.8,
        'results': 0.8,
        'discussion': 0.8,
        'references': 0.9,
        'bibliography': 0.9,
        'acknowledgments': 0.8,
        'appendix': 0.9,
        'overview': 0.7,
        'background': 0.7,
        'literature': 0.7,
        'review': 0.7,
        'analysis': 0.7,
        'implementation': 0.7,
        'evaluation': 0.7,
        'future work': 0.7,
        'limitations': 0.7,
    }
    
    def __init__(self):
        self.compiled_patterns = [(re.compile(pattern, re.IGNORECASE), name, score) 
                                  for pattern, name, score in self.HEADING_PATTERNS]
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze text for heading patterns"""
        features = {}
        clean_text = text.strip()
        
        # Pattern matching
        for pattern, name, base_score in self.compiled_patterns:
            if pattern.match(clean_text):
                features[f'pattern_{name}'] = base_score
        
        # Semantic analysis
        lower_text = clean_text.lower()
        for indicator, score in self.SEMANTIC_INDICATORS.items():
            if indicator in lower_text:
                features[f'semantic_{indicator}'] = score
        
        # Text characteristics
        features.update({
            'is_short': 1.0 if len(clean_text) < 100 else 0.0,
            'is_very_short': 1.0 if len(clean_text) < 50 else 0.0,
            'is_title_case': 1.0 if clean_text.istitle() else 0.0,
            'is_upper_case': 1.0 if clean_text.isupper() else 0.0,
            'ends_with_colon': 1.0 if clean_text.endswith(':') else 0.0,
            'word_count': len(clean_text.split()),
            'char_count': len(clean_text),
            'has_numbers': 1.0 if re.search(r'\d', clean_text) else 0.0,
            'starts_with_number': 1.0 if re.match(r'^\d', clean_text) else 0.0,
            'punctuation_ratio': len(re.findall(r'[^\w\s]', clean_text)) / max(len(clean_text), 1)
        })
        
        return features

class PositionalAnalyzer:
    """Analyzes positional features of text blocks"""
    
    def __init__(self, doc_analyzer: DocumentAnalyzer):
        self.doc_analyzer = doc_analyzer
    
    def analyze_position(self, block: TextBlock) -> Dict[str, float]:
        """Analyze positional features"""
        features = {}
        page_stats = self.doc_analyzer.page_stats[block.page]
        
        # Page-relative position
        features['page_position_y'] = block.y / page_stats['bottom_margin']
        features['is_top_of_page'] = 1.0 if block.y < page_stats['top_margin'] + 50 else 0.0
        
        # Margin analysis
        left_margin = block.x
        features['left_margin'] = left_margin
        features['is_left_aligned'] = 1.0 if left_margin in self.doc_analyzer.layout_patterns['common_left_margins'][:2] else 0.0
        features['is_indented'] = 1.0 if left_margin > min(self.doc_analyzer.layout_patterns['common_left_margins']) + 10 else 0.0
        
        # Isolation (whitespace around)
        features['isolation_score'] = self._calculate_isolation_score(block)
        
        return features
    
    def _calculate_isolation_score(self, block: TextBlock) -> float:
        """Calculate how isolated a block is from surrounding text"""
        same_page_blocks = [b for b in self.doc_analyzer.blocks if b.page == block.page]
        
        # Find nearest blocks above and below
        above_blocks = [b for b in same_page_blocks if b.y < block.y]
        below_blocks = [b for b in same_page_blocks if b.y > block.y]
        
        space_above = min([block.y - b.y - b.height for b in above_blocks], default=100)
        space_below = min([b.y - block.y - block.height for b in below_blocks], default=100)
        
        # Normalize and combine
        isolation = (space_above + space_below) / 100.0
        return min(isolation, 1.0)

class FontAnalyzer:
    """Analyzes font-related features"""
    
    def __init__(self, doc_analyzer: DocumentAnalyzer):
        self.doc_analyzer = doc_analyzer
    
    def analyze_font(self, block: TextBlock) -> Dict[str, float]:
        """Analyze font features"""
        features = {}
        font_stats = self.doc_analyzer.font_stats
        
        # Font size analysis
        features['font_size'] = block.font_size
        features['font_size_ratio'] = block.font_size / font_stats['body_text_size']
        features['is_larger_than_body'] = 1.0 if block.font_size > font_stats['body_text_size'] else 0.0
        features['is_largest_on_page'] = 1.0 if block.font_size == max(
            b.font_size for b in self.doc_analyzer.blocks if b.page == block.page
        ) else 0.0
        
        # Percentile ranking
        all_sizes = [b.font_size for b in self.doc_analyzer.blocks]
        percentile = (sum(1 for s in all_sizes if s < block.font_size) / len(all_sizes)) * 100
        features['size_percentile'] = percentile / 100.0
        
        # Font style
        features['is_bold'] = 1.0 if block.is_bold else 0.0
        features['is_italic'] = 1.0 if block.is_italic else 0.0
        
        # Font name analysis
        if block.font_name:
            features['font_suggests_heading'] = 1.0 if any(
                indicator in block.font_name.lower() 
                for indicator in ['bold', 'heavy', 'black', 'title', 'heading']
            ) else 0.0
        
        return features

class MultilingualAnalyzer:
    """Handles multilingual text analysis"""
    
    LANGUAGE_SPECIFIC_PATTERNS = {
        'japanese': {
            'patterns': [
                (r'^第\d+章', 'chapter', 0.95),
                (r'^第\d+節', 'section', 0.9),
                (r'^\d+\.', 'numbered', 0.8),
            ],
            'indicators': ['概要', '序論', '結論', '参考文献', '付録']
        },
        'chinese': {
            'patterns': [
                (r'^第\d+章', 'chapter', 0.95),
                (r'^第\d+节', 'section', 0.9),
                (r'^\d+\.', 'numbered', 0.8),
            ],
            'indicators': ['概述', '介绍', '结论', '参考文献', '附录']
        },
        'arabic': {
            'patterns': [
                (r'^الفصل\s+\d+', 'chapter', 0.95),
                (r'^\d+\.', 'numbered', 0.8),
            ],
            'indicators': ['مقدمة', 'خلاصة', 'المراجع', 'الملحق']
        }
    }
    
    def detect_language(self, text: str) -> str:
        """Detect text language using Unicode ranges"""
        # Japanese (Hiragana, Katakana, Kanji)
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        
        # Arabic
        if re.search(r'[\u0600-\u06FF]', text):
            return 'arabic'
        
        # Chinese
        if re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        
        # Korean
        if re.search(r'[\uAC00-\uD7AF]', text):
            return 'korean'
        
        return 'latin'
    
    def analyze_multilingual(self, text: str) -> Dict[str, float]:
        """Analyze text for multilingual heading patterns"""
        features = {}
        language = self.detect_language(text)
        features['language'] = language
        
        if language in self.LANGUAGE_SPECIFIC_PATTERNS:
            patterns = self.LANGUAGE_SPECIFIC_PATTERNS[language]
            
            # Pattern matching
            for pattern, name, score in patterns['patterns']:
                if re.search(pattern, text):
                    features[f'multilingual_{name}'] = score
            
            # Indicator matching
            for indicator in patterns['indicators']:
                if indicator in text:
                    features[f'multilingual_indicator'] = 0.8
        
        return features

class HierarchyValidator:
    """Validates and corrects heading hierarchy"""
    
    def __init__(self):
        self.level_order = [HeadingLevel.TITLE, HeadingLevel.H1, HeadingLevel.H2, HeadingLevel.H3]
    
    def validate_hierarchy(self, candidates: List[HeadingCandidate]) -> List[HeadingCandidate]:
        """Validate and correct heading hierarchy"""
        if not candidates:
            return candidates
        
        # Sort by page and position
        sorted_candidates = sorted(candidates, key=lambda c: (c.block.page, c.block.y))
        
        # Correct hierarchy violations
        corrected = []
        last_level_index = -1
        
        for candidate in sorted_candidates:
            current_level_index = self.level_order.index(candidate.level)
            
            # Don't allow jumps of more than 1 level
            if current_level_index > last_level_index + 1:
                # Adjust to next logical level
                candidate.level = self.level_order[min(last_level_index + 1, len(self.level_order) - 1)]
                candidate.confidence *= 0.9  # Slightly reduce confidence for corrections
            
            corrected.append(candidate)
            last_level_index = self.level_order.index(candidate.level)
        
        return corrected

class RobustHeadingDetector:
    """Main heading detection system with multi-signal analysis"""
    
    def __init__(self):
        self.text_analyzer = TextPatternAnalyzer()
        self.multilingual_analyzer = MultilingualAnalyzer()
        self.hierarchy_validator = HierarchyValidator()
        
        # Feature weights (can be tuned)
        self.feature_weights = {
            'font_size_ratio': 0.3,
            'pattern_numbered_section': 0.25,
            'pattern_numbered_subsection': 0.2,
            'pattern_numbered_subsubsection': 0.15,
            'is_larger_than_body': 0.2,
            'is_bold': 0.15,
            'is_short': 0.1,
            'isolation_score': 0.1,
            'is_left_aligned': 0.1,
            'size_percentile': 0.15,
            'semantic_introduction': 0.2,
            'semantic_conclusion': 0.2,
            'semantic_abstract': 0.25,
            'multilingual_chapter': 0.3,
            'multilingual_section': 0.25,
        }
    
    def detect_headings(self, blocks: List[Dict]) -> Dict:
        """Main heading detection function"""
        # Convert to TextBlock objects
        text_blocks = self._convert_blocks(blocks)
        
        # Initialize analyzers
        doc_analyzer = DocumentAnalyzer(text_blocks)
        positional_analyzer = PositionalAnalyzer(doc_analyzer)
        font_analyzer = FontAnalyzer(doc_analyzer)
        
        # Find title
        title = self._find_title(text_blocks, doc_analyzer)
        
        # Generate heading candidates
        candidates = []
        for block in text_blocks:
            if self._is_potential_heading(block, doc_analyzer):
                candidate = self._analyze_block(
                    block, doc_analyzer, positional_analyzer, font_analyzer
                )
                if candidate.confidence > 0.3:  # Threshold for consideration
                    candidates.append(candidate)
        
        # Validate hierarchy
        validated_candidates = self.hierarchy_validator.validate_hierarchy(candidates)
        
        # Convert to output format
        outline = []
        for candidate in validated_candidates:
            if candidate.level != HeadingLevel.TITLE:  # Exclude title from outline
                outline.append({
                    "level": candidate.level.value,
                    "text": candidate.block.text.strip(),
                    "page": candidate.block.page
                })
        
        return {
            "title": title,
            "outline": outline
        }
    
    def _convert_blocks(self, blocks: List[Dict]) -> List[TextBlock]:
        """Convert raw blocks to TextBlock objects"""
        text_blocks = []
        for block in blocks:
            text_block = TextBlock(
                text=block["text"],
                page=block["page"],
                font_size=block["font_size"],
                bbox=block.get("bbox", (0, 0, 0, 0)),
                is_bold=block.get("is_bold", False),
                is_italic=block.get("is_italic", False),
                font_name=block.get("font_name", "")
            )
            text_blocks.append(text_block)
        return text_blocks
    
    def _find_title(self, blocks: List[TextBlock], doc_analyzer: DocumentAnalyzer) -> str:
        """Find document title with enhanced heuristics"""
        # Try page 1 first
        page1_blocks = [b for b in blocks if b.page == 1 and b.text.strip()]
        
        if page1_blocks:
            # Multiple criteria for title selection
            title_candidates = []
            
            for block in page1_blocks:
                score = 0
                
                # Font size (primary factor)
                if block.font_size == max(b.font_size for b in page1_blocks):
                    score += 3
                
                # Position (titles often near top)
                if block.y < np.percentile([b.y for b in page1_blocks], 25):
                    score += 2
                
                # Length (titles are usually not too long)
                if 10 < len(block.text) < 200:
                    score += 1
                
                # Centered text
                page_width = max(b.x + b.width for b in page1_blocks) - min(b.x for b in page1_blocks)
                if abs(block.x - page_width/2) < page_width * 0.1:
                    score += 1
                
                title_candidates.append((score, block))
            
            # Return highest scoring candidate
            if title_candidates:
                return max(title_candidates, key=lambda x: x[0])[1].text.strip()
        
        # Fallback to largest text anywhere
        if blocks:
            return max(blocks, key=lambda b: b.font_size).text.strip()
        
        return "Untitled Document"
    
    def _is_potential_heading(self, block: TextBlock, doc_analyzer: DocumentAnalyzer) -> bool:
        """Quick filter for potential headings"""
        text = block.text.strip()
        
        # Basic filters
        if not text or len(text) > 300:
            return False
        
        # Font size filter
        if block.font_size < doc_analyzer.font_stats['body_text_size'] * 0.9:
            return False
        
        # Pattern-based quick acceptance
        if any(pattern.match(text) for pattern, _, _ in self.text_analyzer.compiled_patterns):
            return True
        
        # Semantic quick acceptance
        if any(indicator in text.lower() for indicator in self.text_analyzer.SEMANTIC_INDICATORS):
            return True
        
        # Font-based acceptance
        if block.font_size > doc_analyzer.font_stats['body_text_size'] * 1.1:
            return True
        
        return False
    
    def _analyze_block(self, block: TextBlock, doc_analyzer: DocumentAnalyzer, 
                      positional_analyzer: PositionalAnalyzer, 
                      font_analyzer: FontAnalyzer) -> HeadingCandidate:
        """Comprehensive block analysis"""
        
        # Gather features from all analyzers
        features = {}
        features.update(self.text_analyzer.analyze_text(block.text))
        features.update(positional_analyzer.analyze_position(block))
        features.update(font_analyzer.analyze_font(block))
        features.update(self.multilingual_analyzer.analyze_multilingual(block.text))
        
        # Calculate confidence score
        confidence = self._calculate_confidence(features)
        
        # Determine heading level
        level = self._determine_level(features, confidence)
        
        return HeadingCandidate(
            block=block,
            confidence=confidence,
            level=level,
            features=features
        )
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence score using weighted features"""
        score = 0.0
        total_weight = 0.0
        
        for feature, value in features.items():
            if feature in self.feature_weights:
                weight = self.feature_weights[feature]
                score += weight * value
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        # Boost for strong patterns
        if any(k.startswith('pattern_') and v > 0.8 for k, v in features.items()):
            score *= 1.2
        
        # Boost for semantic indicators
        if any(k.startswith('semantic_') and v > 0.7 for k, v in features.items()):
            score *= 1.1
        
        return min(score, 1.0)
    
    def _determine_level(self, features: Dict[str, float], confidence: float) -> HeadingLevel:
        """Determine heading level based on features"""
        
        # Pattern-based level determination
        if features.get('pattern_numbered_section', 0) > 0.5:
            return HeadingLevel.H1
        elif features.get('pattern_numbered_subsection', 0) > 0.5:
            return HeadingLevel.H2
        elif features.get('pattern_numbered_subsubsection', 0) > 0.5:
            return HeadingLevel.H3
        
        # Multilingual pattern-based
        if features.get('multilingual_chapter', 0) > 0.5:
            return HeadingLevel.H1
        elif features.get('multilingual_section', 0) > 0.5:
            return HeadingLevel.H2
        
        # Font size-based with confidence weighting
        font_ratio = features.get('font_size_ratio', 1.0)
        if font_ratio > 1.5 and confidence > 0.7:
            return HeadingLevel.H1
        elif font_ratio > 1.2 and confidence > 0.6:
            return HeadingLevel.H2
        elif font_ratio > 1.0 and confidence > 0.5:
            return HeadingLevel.H3
        
        # Default based on confidence
        if confidence > 0.8:
            return HeadingLevel.H1
        elif confidence > 0.6:
            return HeadingLevel.H2
        else:
            return HeadingLevel.H3

# Enhanced extractor with better font information
def extract_enhanced_text_blocks(pdf_path):
    """Enhanced text extraction with better font information"""
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
                bbox = None
                is_bold = False
                is_italic = False
                font_name = ""
                
                for span in line["spans"]:
                    if font_size is None:
                        font_size = span["size"]
                        bbox = span["bbox"]
                        font_name = span.get("font", "")
                        
                        # Extract font style information
                        if font_name:
                            is_bold = any(keyword in font_name.lower() 
                                        for keyword in ['bold', 'heavy', 'black'])
                            is_italic = any(keyword in font_name.lower() 
                                          for keyword in ['italic', 'oblique'])
                    
                    line_text += span["text"]
                
                text = line_text.strip()
                if text:
                    all_blocks.append({
                        "text": text,
                        "page": page_num + 1,
                        "font_size": font_size,
                        "bbox": bbox,
                        "is_bold": is_bold,
                        "is_italic": is_italic,
                        "font_name": font_name
                    })
    
    doc.close()
    return all_blocks

# Main execution
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python robust_heading_detector.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Extract enhanced text blocks
    text_blocks = extract_enhanced_text_blocks(pdf_path)
    
    # Detect headings
    detector = RobustHeadingDetector()
    result = detector.detect_headings(text_blocks)
    
    # Output results
    print(json.dumps(result, indent=2, ensure_ascii=False))