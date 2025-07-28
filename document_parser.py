
import fitz
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import cv2
from PIL import Image, ImageDraw
import io

@dataclass
class TextBlock:
    """Enhanced text block with comprehensive metadata"""
    text: str
    page: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    font_size: float
    font_name: str
    font_flags: int
    color: int
    
    # Computed properties
    line_height: float = 0.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    
    # Spatial relationships
    blocks_above: List[int] = None
    blocks_below: List[int] = None
    blocks_left: List[int] = None
    blocks_right: List[int] = None
    
    # Visual features
    is_bold: bool = False
    is_italic: bool = False
    is_underlined: bool = False
    visual_prominence: float = 0.0
    
    def __post_init__(self):
        if self.blocks_above is None:
            self.blocks_above = []
        if self.blocks_below is None:
            self.blocks_below = []
        if self.blocks_left is None:
            self.blocks_left = []
        if self.blocks_right is None:
            self.blocks_right = []
            
        # Extract font style from flags
        self.is_bold = bool(self.font_flags & 2**4)
        self.is_italic = bool(self.font_flags & 2**1)
        self.is_underlined = bool(self.font_flags & 2**0)
    
    @property
    def x0(self) -> float:
        return self.bbox[0]
    
    @property
    def y0(self) -> float:
        return self.bbox[1]
    
    @property
    def x1(self) -> float:
        return self.bbox[2]
    
    @property
    def y1(self) -> float:
        return self.bbox[3]
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2
    
    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

@dataclass 
class PageMetadata:
    """Page-level metadata for layout analysis"""
    page_num: int
    width: float
    height: float
    text_blocks: List[TextBlock]
    
    # Layout statistics
    avg_font_size: float = 0.0
    dominant_font_size: float = 0.0
    font_size_variance: float = 0.0
    
    # Margin detection
    left_margin: float = 0.0
    right_margin: float = 0.0
    top_margin: float = 0.0
    bottom_margin: float = 0.0
    
    # Column detection
    num_columns: int = 1
    column_boundaries: List[float] = None
    
    def __post_init__(self):
        if self.column_boundaries is None:
            self.column_boundaries = []
        if self.text_blocks:
            self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute page-level statistics"""
        font_sizes = [block.font_size for block in self.text_blocks]
        
        self.avg_font_size = np.mean(font_sizes)
        self.font_size_variance = np.var(font_sizes)
        
        # Find dominant font size (most common)
        from collections import Counter
        size_counts = Counter(font_sizes)
        self.dominant_font_size = size_counts.most_common(1)[0][0]
        
        # Detect margins
        x_coords = [block.x0 for block in self.text_blocks]
        y_coords = [block.y0 for block in self.text_blocks]
        
        self.left_margin = min(x_coords) if x_coords else 0
        self.top_margin = min(y_coords) if y_coords else 0
        self.right_margin = max(block.x1 for block in self.text_blocks) if self.text_blocks else self.width
        self.bottom_margin = max(block.y1 for block in self.text_blocks) if self.text_blocks else self.height


class EnhancedDocumentParser:
    """Enhanced PDF parser with spatial intelligence and layout analysis"""
    
    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr
        self.pages_metadata: List[PageMetadata] = []
        self.all_blocks: List[TextBlock] = []
        
    def parse_document(self, pdf_path: str) -> Tuple[List[TextBlock], List[PageMetadata]]:
        """Main parsing function with enhanced text extraction"""
        
        doc = fitz.open(pdf_path)
        self.pages_metadata = []
        self.all_blocks = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text blocks with detailed information
                page_blocks = self._extract_page_blocks(page, page_num + 1)
                
                # Create page metadata
                page_metadata = PageMetadata(
                    page_num=page_num + 1,
                    width=page.rect.width,
                    height=page.rect.height,
                    text_blocks=page_blocks
                )
                
                self.pages_metadata.append(page_metadata)
                self.all_blocks.extend(page_blocks)
                
            # Post-process: compute spatial relationships
            self._compute_spatial_relationships()
            
            # Detect reading order
            self._detect_reading_order()
            
        finally:
            doc.close()
            
        return self.all_blocks, self.pages_metadata
    
    def _extract_page_blocks(self, page: fitz.Page, page_num: int) -> List[TextBlock]:
        """Extract text blocks from a single page with enhanced metadata"""
        
        page_blocks = []
        
        # Get detailed text information
        text_dict = page.get_text("dict")
        
        for block in text_dict["blocks"]:
            if block["type"] != 0:  # Skip image blocks
                continue
                
            for line in block["lines"]:
                # Combine spans that are likely part of the same semantic unit
                line_spans = line["spans"]
                if not line_spans:
                    continue
                
                # Group consecutive spans with similar properties
                grouped_spans = self._group_similar_spans(line_spans)
                
                for span_group in grouped_spans:
                    if not span_group:
                        continue
                        
                    # Create text block from span group
                    text_block = self._create_text_block_from_spans(span_group, page_num)
                    
                    if text_block and text_block.text.strip():
                        page_blocks.append(text_block)
        
        return page_blocks
    
    def _group_similar_spans(self, spans: List[Dict]) -> List[List[Dict]]:
        """Group consecutive spans with similar formatting"""
        if not spans:
            return []
        
        groups = []
        current_group = [spans[0]]
        
        for i in range(1, len(spans)):
            current_span = spans[i]
            prev_span = spans[i-1]
            
            # Check if spans should be grouped
            font_similar = abs(current_span["size"] - prev_span["size"]) < 0.5
            flags_similar = current_span["flags"] == prev_span["flags"]
            font_similar_name = current_span["font"] == prev_span["font"]
            
            # Check spatial proximity (same line)
            x_gap = current_span["bbox"][0] - prev_span["bbox"][2]
            y_overlap = min(current_span["bbox"][3], prev_span["bbox"][3]) - max(current_span["bbox"][1], prev_span["bbox"][1])
            
            if (font_similar and flags_similar and font_similar_name and 
                x_gap < 20 and y_overlap > 0):  # Same line, reasonable gap
                current_group.append(current_span)
            else:
                groups.append(current_group)
                current_group = [current_span]
        
        groups.append(current_group)
        return groups
    
    def _create_text_block_from_spans(self, spans: List[Dict], page_num: int) -> Optional[TextBlock]:
        """Create a TextBlock from a group of spans"""
        if not spans:
            return None
        
        # Combine text
        text = "".join(span["text"] for span in spans)
        
        # Compute bounding box
        x0 = min(span["bbox"][0] for span in spans)
        y0 = min(span["bbox"][1] for span in spans)
        x1 = max(span["bbox"][2] for span in spans)
        y1 = max(span["bbox"][3] for span in spans)
        
        # Use properties from the first span (they should be similar)
        first_span = spans[0]
        
        # Compute additional metrics
        char_count = sum(len(span["text"]) for span in spans)
        total_width = x1 - x0
        char_spacing = total_width / max(char_count, 1)
        
        # Estimate line height
        line_height = y1 - y0
        
        return TextBlock(
            text=text,
            page=page_num,
            bbox=(x0, y0, x1, y1),
            font_size=first_span["size"],
            font_name=first_span["font"],
            font_flags=first_span["flags"],
            color=first_span.get("color", 0),
            line_height=line_height,
            char_spacing=char_spacing
        )
    
    def _compute_spatial_relationships(self):
        """Compute spatial relationships between text blocks"""
        
        # Group blocks by page for efficiency
        page_blocks = {}
        for i, block in enumerate(self.all_blocks):
            if block.page not in page_blocks:
                page_blocks[block.page] = []
            page_blocks[block.page].append((i, block))
        
        # Compute relationships within each page
        for page_num, blocks_with_idx in page_blocks.items():
            self._compute_page_relationships(blocks_with_idx)
    
    def _compute_page_relationships(self, blocks_with_idx: List[Tuple[int, TextBlock]]):
        """Compute spatial relationships within a page"""
        
        for i, (idx_i, block_i) in enumerate(blocks_with_idx):
            for j, (idx_j, block_j) in enumerate(blocks_with_idx):
                if i == j:
                    continue
                
                # Compute relative positions
                relation = self._get_spatial_relation(block_i, block_j)
                
                if relation == "above":
                    self.all_blocks[idx_i].blocks_below.append(idx_j)
                    self.all_blocks[idx_j].blocks_above.append(idx_i)
                elif relation == "left":
                    self.all_blocks[idx_i].blocks_right.append(idx_j)
                    self.all_blocks[idx_j].blocks_left.append(idx_i)
    
    def _get_spatial_relation(self, block1: TextBlock, block2: TextBlock) -> Optional[str]:
        """Determine spatial relationship between two blocks"""
        
        # Vertical relationship
        if block1.y1 < block2.y0 - 5:  # block1 is above block2
            # Check for horizontal overlap
            h_overlap = min(block1.x1, block2.x1) - max(block1.x0, block2.x0)
            if h_overlap > min(block1.width, block2.width) * 0.3:  # Significant overlap
                return "above"
        
        # Horizontal relationship  
        elif block1.x1 < block2.x0 - 5:  # block1 is left of block2
            # Check for vertical overlap
            v_overlap = min(block1.y1, block2.y1) - max(block1.y0, block2.y0)
            if v_overlap > min(block1.height, block2.height) * 0.3:  # Significant overlap
                return "left"
        
        return None
    
    def _detect_reading_order(self):
        """Detect and assign reading order to blocks"""
        
        # Group by page and sort by reading order
        for page_metadata in self.pages_metadata:
            page_blocks = page_metadata.text_blocks
            
            # Sort by Y coordinate first (top to bottom), then X coordinate (left to right)
            page_blocks.sort(key=lambda b: (b.y0, b.x0))
            
            # Assign reading order numbers
            for i, block in enumerate(page_blocks):
                block.reading_order = i
    
    def get_document_statistics(self) -> Dict:
        """Get overall document statistics for feature engineering"""
        
        if not self.all_blocks:
            return {}
        
        font_sizes = [block.font_size for block in self.all_blocks]
        
        return {
            "total_blocks": len(self.all_blocks),
            "total_pages": len(self.pages_metadata),
            "font_size_stats": {
                "min": min(font_sizes),
                "max": max(font_sizes),
                "mean": np.mean(font_sizes),
                "std": np.std(font_sizes),
                "percentiles": {
                    "p25": np.percentile(font_sizes, 25),
                    "p50": np.percentile(font_sizes, 50),
                    "p75": np.percentile(font_sizes, 75),
                    "p90": np.percentile(font_sizes, 90),
                    "p95": np.percentile(font_sizes, 95)
                }
            },
            "layout_stats": {
                "avg_blocks_per_page": len(self.all_blocks) / len(self.pages_metadata),
                "pages_with_multiple_columns": sum(1 for p in self.pages_metadata if p.num_columns > 1)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    parser = EnhancedDocumentParser()
    
    # This would be called from main processing
    # blocks, page_metadata = parser.parse_document("sample.pdf")
    # stats = parser.get_document_statistics()
    
    print("Enhanced Document Parser Module Ready")
    print("Features: Spatial relationships, layout analysis, enhanced text extraction")