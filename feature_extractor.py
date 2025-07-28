import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
import unicodedata
from functools import lru_cache
import math

from document_parser import TextBlock, PageMetadata


@dataclass
class BlockFeatures:
    """Comprehensive feature set for a text block"""
    # Core text features
    text_length: float
    word_count: float  
    sentence_count: float
    avg_word_length: float
    
    # Font & Style features
    font_size: float
    font_size_ratio: float  # relative to page average
    font_size_percentile: float  # percentile in document
    is_bold: float
    is_italic: float
    is_underlined: float
    
    # Position features
    x_position: float  # normalized 0-1
    y_position: float  # normalized 0-1
    distance_from_left: float
    distance_from_top: float
    is_centered: float
    
    # Layout features
    isolation_score: float  # how isolated the block is
    column_position: float  # which column (0-1)
    indentation_level: float
    
    # Spatial relationship features
    blocks_above_count: float
    blocks_below_count: float
    avg_font_size_above: float
    avg_font_size_below: float
    
    # Pattern recognition features
    starts_with_number: float
    has_numbering_pattern: float
    numbering_level: float  # 1.1.1 = level 3
    has_bullet_pattern: float
    
    # Semantic features (will be filled by semantic analyzer)
    semantic_importance: float = 0.0
    topic_coherence: float = 0.0
    keyword_density: float = 0.0
    
    # Language features
    language_confidence: float = 0.0
    is_multilingual: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.text_length, self.word_count, self.sentence_count, self.avg_word_length,
            self.font_size, self.font_size_ratio, self.font_size_percentile,
            self.is_bold, self.is_italic, self.is_underlined,
            self.x_position, self.y_position, self.distance_from_left, self.distance_from_top,
            self.is_centered, self.isolation_score, self.column_position, self.indentation_level,
            self.blocks_above_count, self.blocks_below_count, 
            self.avg_font_size_above, self.avg_font_size_below,
            self.starts_with_number, self.has_numbering_pattern, self.numbering_level,
            self.has_bullet_pattern, self.semantic_importance, self.topic_coherence,
            self.keyword_density, self.language_confidence, self.is_multilingual
        ])


class PatternDetector:
    """Advanced pattern detection for heading identification"""
    
    # Comprehensive heading patterns with weights
    HEADING_PATTERNS = [
        # Hierarchical numbering
        (r'^\s*(\d+)\.?\s+', 'level_1_number', 0.95, 1),
        (r'^\s*(\d+)\.(\d+)\.?\s+', 'level_2_number', 0.90, 2), 
        (r'^\s*(\d+)\.(\d+)\.(\d+)\.?\s+', 'level_3_number', 0.85, 3),
        (r'^\s*(\d+)\.(\d+)\.(\d+)\.(\d+)\.?\s+', 'level_4_number', 0.80, 4),
        
        # Roman numerals
        (r'^\s*([IVX]+)\.?\s+', 'roman_upper', 0.85, 1),
        (r'^\s*([ivx]+)\.?\s+', 'roman_lower', 0.80, 2),
        
        # Alphabetic
        (r'^\s*([A-Z])\.?\s+', 'alpha_upper', 0.75, 1),
        (r'^\s*([a-z])\.?\s+', 'alpha_lower', 0.70, 2),
        
        # Special document structures
        (r'^\s*(Chapter|CHAPTER)\s+\d+', 'chapter', 0.98, 1),
        (r'^\s*(Section|SECTION)\s+\d+', 'section', 0.95, 2),
        (r'^\s*(Part|PART)\s+\d+', 'part', 0.98, 1),
        (r'^\s*(Appendix|APPENDIX)\s+[A-Z]?', 'appendix', 0.93, 1),
        
        # Question patterns
        (r'^\s*(\d+)\.?\s+.*\?', 'question', 0.80, 3),
        (r'^\s*Q\d+[.:]\s*', 'question_format', 0.85, 3),
        
        # Bullet patterns
        (r'^\s*[•·▪▫■□▲△]\s+', 'bullet', 0.60, 3),
        (r'^\s*[-−–—]\s+', 'dash_bullet', 0.55, 3),
        (r'^\s*[*]\s+', 'asterisk_bullet', 0.55, 3),
        
        # Bracketed patterns
        (r'^\s*\[(\d+)\]\s*', 'bracketed_number', 0.70, 3),
        (r'^\s*\((\d+)\)\s*', 'parenthesized_number', 0.65, 3),
    ]
    
    # Semantic heading indicators
    SEMANTIC_INDICATORS = {
        # High confidence
        'abstract': 0.95, 'introduction': 0.90, 'conclusion': 0.90, 'summary': 0.88,
        'methodology': 0.85, 'results': 0.85, 'discussion': 0.85, 'references': 0.95,
        'bibliography': 0.95, 'acknowledgments': 0.85, 'acknowledgements': 0.85,
        
        # Medium confidence  
        'background': 0.75, 'literature review': 0.80, 'related work': 0.80,
        'experimental setup': 0.75, 'evaluation': 0.75, 'analysis': 0.70,
        'implementation': 0.70, 'future work': 0.75, 'limitations': 0.70,
        
        # Lower confidence
        'overview': 0.65, 'outline': 0.65, 'objectives': 0.65, 'goals': 0.60,
    }
    
    # Multilingual patterns
    MULTILINGUAL_PATTERNS = {
        'japanese': [
            (r'第(\d+)章', 'chapter', 0.95, 1),
            (r'第(\d+)節', 'section', 0.90, 2),
            (r'(\d+)\.', 'numbered', 0.85, None),
        ],
        'chinese': [
            (r'第(\d+)章', 'chapter', 0.95, 1),
            (r'第(\d+)节', 'section', 0.90, 2),
            (r'(\d+)\.', 'numbered', 0.85, None),
        ],
        'arabic': [
            (r'الفصل\s+(\d+)', 'chapter', 0.95, 1),
            (r'(\d+)\.', 'numbered', 0.85, None),
        ]
    }
    
    def __init__(self):
        self.compiled_patterns = []
        for pattern, name, confidence, level in self.HEADING_PATTERNS:
            self.compiled_patterns.append((
                re.compile(pattern, re.IGNORECASE), name, confidence, level
            ))
        
        # Compile multilingual patterns
        self.multilingual_compiled = {}
        for lang, patterns in self.MULTILINGUAL_PATTERNS.items():
            self.multilingual_compiled[lang] = []
            for pattern, name, confidence, level in patterns:
                self.multilingual_compiled[lang].append((
                    re.compile(pattern, re.IGNORECASE), name, confidence, level
                ))
    
    def detect_patterns(self, text: str) -> Dict[str, float]:
        """Detect heading patterns in text"""
        features = {}
        clean_text = text.strip()
        
        if not clean_text:
            return features
        
        # Standard pattern detection
        max_confidence = 0.0
        detected_level = 0
        
        for pattern, name, confidence, level in self.compiled_patterns:
            match = pattern.match(clean_text)
            if match:
                features[f'pattern_{name}'] = confidence
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_level = level or 0
        
        features['max_pattern_confidence'] = max_confidence
        features['detected_numbering_level'] = detected_level
        
        # Semantic indicator detection
        lower_text = clean_text.lower()
        semantic_scores = []
        
        for indicator, confidence in self.SEMANTIC_INDICATORS.items():
            if indicator in lower_text:
                features[f'semantic_{indicator.replace(" ", "_")}'] = confidence
                semantic_scores.append(confidence)
        
        features['max_semantic_confidence'] = max(semantic_scores) if semantic_scores else 0.0
        
        # Multilingual detection
        lang = self._detect_language(clean_text)
        features['language'] = lang
        
        if lang in self.multilingual_compiled:
            for pattern, name, confidence, level in self.multilingual_compiled[lang]:
                if pattern.search(clean_text):
                    features[f'multilingual_{name}'] = confidence
        
        return features
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on Unicode ranges"""
        # Japanese
        if re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text):
            return 'japanese'
        # Arabic  
        elif re.search(r'[\u0600-\u06FF]', text):
            return 'arabic'
        # Chinese
        elif re.search(r'[\u4E00-\u9FFF]', text):
            return 'chinese'
        # Korean
        elif re.search(r'[\uAC00-\uD7AF]', text):
            return 'korean'
        else:
            return 'latin'


class LayoutAnalyzer:
    """Analyzes document layout and spatial relationships"""
    
    def __init__(self, blocks: List[TextBlock], pages_metadata: List[PageMetadata]):
        self.blocks = blocks
        self.pages_metadata = pages_metadata
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial index for efficient neighbor queries"""
        self.page_blocks = defaultdict(list)
        for i, block in enumerate(self.blocks):
            self.page_blocks[block.page].append((i, block))
    
    def compute_layout_features(self, block_idx: int) -> Dict[str, float]:
        """Compute layout-based features for a block"""
        block = self.blocks[block_idx]
        page_metadata = self.pages_metadata[block.page - 1]
        
        features = {}
        
        # Position normalization
        features['x_position'] = block.center_x / page_metadata.width
        features['y_position'] = block.center_y / page_metadata.height
        features['distance_from_left'] = block.x0 - page_metadata.left_margin
        features['distance_from_top'] = block.y0 - page_metadata.top_margin
        
        # Centering detection
        page_center_x = page_metadata.width / 2
        distance_from_center = abs(block.center_x - page_center_x)
        features['is_centered'] = 1.0 if distance_from_center < page_metadata.width * 0.1 else 0.0
        
        # Isolation score
        features['isolation_score'] = self._compute_isolation_score(block_idx)
        
        # Column detection
        features['column_position'] = self._detect_column_position(block, page_metadata)
        
        # Indentation analysis
        features['indentation_level'] = self._compute_indentation_level(block, page_metadata)
        
        return features
    
    def _compute_isolation_score(self, block_idx: int) -> float:
        """Compute how isolated a block is from surrounding text"""
        block = self.blocks[block_idx]
        same_page_blocks = [b for i, b in self.page_blocks[block.page]]
        
        if len(same_page_blocks) <= 1:
            return 1.0
        
        # Find nearest neighbors
        above_blocks = [b for b in same_page_blocks 
                       if b.y1 < block.y0 and abs(b.center_x - block.center_x) < block.width]
        below_blocks = [b for b in same_page_blocks 
                       if b.y0 > block.y1 and abs(b.center_x - block.center_x) < block.width]
        
        # Compute distances
        space_above = min([block.y0 - b.y1 for b in above_blocks], default=100)
        space_below = min([b.y0 - block.y1 for b in below_blocks], default=100)
        
        # Normalize isolation score
        avg_line_height = np.mean([b.height for b in same_page_blocks])
        isolation = (space_above + space_below) / (2 * avg_line_height)
        return min(isolation, 2.0) / 2.0  # Cap at 2 line heights
    
    def _detect_column_position(self, block: TextBlock, page_metadata: PageMetadata) -> float:
        """Detect which column the block belongs to"""
        if page_metadata.num_columns == 1:
            return 0.5  # Single column, centered
        
        # Simple column detection based on x-position
        column_width = page_metadata.width / page_metadata.num_columns
        column_idx = int(block.x0 / column_width)
        return (column_idx + 0.5) / page_metadata.num_columns
    
    def _compute_indentation_level(self, block: TextBlock, page_metadata: PageMetadata) -> float:
        """Compute indentation level relative to page margins"""
        base_margin = page_metadata.left_margin
        indentation = block.x0 - base_margin
        
        # Normalize by typical indentation unit (assume 20 points)
        return max(indentation, 0) / 20.0


class MultiModalFeatureExtractor:
    """Main feature extraction system combining all analyzers"""
    
    def __init__(self):
        self.pattern_detector = PatternDetector()
        self.layout_analyzer = None
        self.doc_stats = None
        
    def extract_features(self, blocks: List[TextBlock], 
                        pages_metadata: List[PageMetadata]) -> List[BlockFeatures]:
        """Extract comprehensive features for all blocks"""
        
        # Initialize analyzers
        self.layout_analyzer = LayoutAnalyzer(blocks, pages_metadata)
        self.doc_stats = self._compute_document_statistics(blocks)
        
        features_list = []
        
        for i, block in enumerate(blocks):
            # Text-based features
            text_features = self._extract_text_features(block)
            
            # Font and style features
            font_features = self._extract_font_features(block, blocks)
            
            # Pattern-based features
            pattern_features = self.pattern_detector.detect_patterns(block.text)
            
            # Layout-based features
            layout_features = self.layout_analyzer.compute_layout_features(i)
            
            # Spatial relationship features
            spatial_features = self._extract_spatial_features(block, blocks)
            
            # Combine all features into BlockFeatures object
            block_features = self._combine_features(
                text_features, font_features, pattern_features, 
                layout_features, spatial_features
            )
            
            features_list.append(block_features)
        
        return features_list
    
    def _extract_text_features(self, block: TextBlock) -> Dict[str, float]:
        """Extract text-based features"""
        text = block.text.strip()
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'starts_with_number': 1.0 if re.match(r'^\d', text) else 0.0,
            'ends_with_colon': 1.0 if text.endswith(':') else 0.0,
            'is_all_caps': 1.0 if text.isupper() and len(text) > 2 else 0.0,
            'is_title_case': 1.0 if text.istitle() else 0.0,
        }
    
    def _extract_font_features(self, block: TextBlock, all_blocks: List[TextBlock]) -> Dict[str, float]:
        """Extract font and styling features"""
        page_blocks = [b for b in all_blocks if b.page == block.page]
        
        # Page-level font statistics
        page_font_sizes = [b.font_size for b in page_blocks]
        avg_page_font = np.mean(page_font_sizes)
        
        # Document-level percentile
        all_font_sizes = [b.font_size for b in all_blocks]
        percentile = (sum(1 for size in all_font_sizes if size < block.font_size) / 
                     len(all_font_sizes))
        
        return {
            'font_size': block.font_size,
            'font_size_ratio': block.font_size / avg_page_font if avg_page_font > 0 else 1.0,
            'font_size_percentile': percentile,
            'is_bold': 1.0 if block.is_bold else 0.0,
            'is_italic': 1.0 if block.is_italic else 0.0,
            'is_underlined': 1.0 if block.is_underlined else 0.0,
        }
    
    def _extract_spatial_features(self, block: TextBlock, all_blocks: List[TextBlock]) -> Dict[str, float]:
        """Extract spatial relationship features"""
        # Count spatial relationships
        above_count = len(block.blocks_above)
        below_count = len(block.blocks_below)
        
        # Average font sizes of neighbors
        avg_font_above = 0.0
        avg_font_below = 0.0
        
        if block.blocks_above:
            above_fonts = [all_blocks[idx].font_size for idx in block.blocks_above 
                          if idx < len(all_blocks)]
            avg_font_above = np.mean(above_fonts) if above_fonts else 0.0
        
        if block.blocks_below:
            below_fonts = [all_blocks[idx].font_size for idx in block.blocks_below 
                          if idx < len(all_blocks)]
            avg_font_below = np.mean(below_fonts) if below_fonts else 0.0
        
        return {
            'blocks_above_count': above_count,
            'blocks_below_count': below_count,
            'avg_font_size_above': avg_font_above,
            'avg_font_size_below': avg_font_below,
        }
    
    def _combine_features(self, text_features: Dict, font_features: Dict,
                         pattern_features: Dict, layout_features: Dict,
                         spatial_features: Dict) -> BlockFeatures:
        """Combine all feature dictionaries into BlockFeatures object"""
        
        # Extract numbering information from patterns
        numbering_level = pattern_features.get('detected_numbering_level', 0)
        has_numbering = any(k.startswith('pattern_') and 'number' in k 
                           for k in pattern_features.keys())
        
        # Extract bullet information
        has_bullet = any(k.startswith('pattern_') and 'bullet' in k 
                        for k in pattern_features.keys())
        
        return BlockFeatures(
            # Text features
            text_length=text_features['text_length'],
            word_count=text_features['word_count'],
            sentence_count=text_features['sentence_count'],
            avg_word_length=text_features['avg_word_length'],
            
            # Font features
            font_size=font_features['font_size'],
            font_size_ratio=font_features['font_size_ratio'],
            font_size_percentile=font_features['font_size_percentile'],
            is_bold=font_features['is_bold'],
            is_italic=font_features['is_italic'],
            is_underlined=font_features['is_underlined'],
            
            # Layout features
            x_position=layout_features['x_position'],
            y_position=layout_features['y_position'],
            distance_from_left=layout_features['distance_from_left'],
            distance_from_top=layout_features['distance_from_top'],
            is_centered=layout_features['is_centered'],
            isolation_score=layout_features['isolation_score'],
            column_position=layout_features['column_position'],
            indentation_level=layout_features['indentation_level'],
            
            # Spatial features
            blocks_above_count=spatial_features['blocks_above_count'],
            blocks_below_count=spatial_features['blocks_below_count'],
            avg_font_size_above=spatial_features['avg_font_size_above'],
            avg_font_size_below=spatial_features['avg_font_size_below'],
            
            # Pattern features
            starts_with_number=text_features['starts_with_number'],
            has_numbering_pattern=1.0 if has_numbering else 0.0,
            numbering_level=numbering_level,
            has_bullet_pattern=1.0 if has_bullet else 0.0,
        )
    
    def _compute_document_statistics(self, blocks: List[TextBlock]) -> Dict:
        """Compute document-level statistics for normalization"""
        font_sizes = [block.font_size for block in blocks]
        text_lengths = [len(block.text) for block in blocks]
        
        return {
            'font_size_stats': {
                'mean': np.mean(font_sizes),
                'std': np.std(font_sizes),
                'min': min(font_sizes),
                'max': max(font_sizes),
            },
            'text_length_stats': {
                'mean': np.mean(text_lengths),
                'std': np.std(text_lengths),
                'median': np.median(text_lengths),
            },
            'total_blocks': len(blocks),
        }


# Example usage
if __name__ == "__main__":
    # This would be used in conjunction with the document parser
    extractor = MultiModalFeatureExtractor()
    
    print("Multi-Modal Feature Extractor Ready")
    print("Features: Text analysis, font analysis, pattern detection, layout analysis")
    print("Supports: Multilingual text, complex numbering schemes, spatial relationships")