import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import pickle
from functools import lru_cache

from document_parser import TextBlock
from feature_extractor import BlockFeatures


class HeadingLevel(Enum):
    """Heading level enumeration"""
    TITLE = "TITLE"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


@dataclass
class HeadingPrediction:
    """Prediction for a single text block"""
    block_idx: int
    is_heading: bool
    confidence: float
    predicted_level: HeadingLevel
    level_confidence: float
    
    # Individual scorer contributions
    pattern_score: float = 0.0
    semantic_score: float = 0.0
    layout_score: float = 0.0
    font_score: float = 0.0
    ensemble_score: float = 0.0
    
    # Feature importance for interpretability
    top_features: List[Tuple[str, float]] = None
    
    def __post_init__(self):
        if self.top_features is None:
            self.top_features = []


class PatternScorer:
    """Scores blocks based on pattern recognition"""
    
    def __init__(self):
        # Weights for different pattern types
        self.pattern_weights = {
            # High-confidence patterns
            'pattern_numbered_section': 0.95,
            'pattern_chapter': 0.98,
            'pattern_section': 0.90,
            'pattern_part': 0.95,
            'pattern_appendix': 0.90,
            
            # Medium-confidence patterns
            'pattern_numbered_subsection': 0.85,
            'pattern_roman_upper': 0.80,
            'pattern_alpha_upper': 0.75,
            
            # Lower-confidence patterns
            'pattern_numbered_subsubsection': 0.80,
            'pattern_roman_lower': 0.70,
            'pattern_alpha_lower': 0.65,
            'pattern_bullet': 0.60,
            'pattern_question': 0.75,
            
            # Multilingual patterns
            'multilingual_chapter': 0.95,
            'multilingual_section': 0.85,
            'multilingual_indicator': 0.80,
            
            # Semantic patterns
            'semantic_introduction': 0.85,
            'semantic_conclusion': 0.85,
            'semantic_abstract': 0.90,
            'semantic_references': 0.95,
            'semantic_methodology': 0.80,
        }
    
    def score_block(self, features: BlockFeatures) -> Tuple[float, Dict[str, float]]:
        """Score a block based on pattern features"""
        
        if not hasattr(features, 'extended_features'):
            return 0.0, {}
        
        pattern_features = features.extended_features
        score = 0.0
        contributing_patterns = {}
        
        # Score based on detected patterns
        for pattern_name, weight in self.pattern_weights.items():
            if pattern_name in pattern_features and pattern_features[pattern_name] > 0:
                pattern_score = pattern_features[pattern_name] * weight
                score = max(score, pattern_score)  # Take maximum, not sum
                contributing_patterns[pattern_name] = pattern_score
        
        # Boost for strong numbering patterns
        if features.has_numbering_pattern > 0.5:
            numbering_boost = 0.1 * features.numbering_level / 3.0  # Normalize by max level
            score += numbering_boost
            contributing_patterns['numbering_boost'] = numbering_boost
        
        return min(score, 1.0), contributing_patterns


class SemanticScorer:
    """Scores blocks based on semantic analysis"""
    
    def __init__(self):
        self.semantic_weights = {
            'heading_probability': 0.4,
            'topic_coherence': 0.25,
            'section_boundary_score': 0.2,
            'discourse_markers': 0.15,
        }
    
    def score_block(self, features: BlockFeatures) -> Tuple[float, Dict[str, float]]:
        """Score a block based on semantic features"""
        
        score = 0.0
        contributing_features = {}
        
        # Base semantic score
        score += features.semantic_importance * 0.3
        contributing_features['semantic_importance'] = features.semantic_importance * 0.3
        
        # Topic coherence (high coherence = likely heading)
        coherence_score = features.topic_coherence * 0.25
        score += coherence_score
        contributing_features['topic_coherence'] = coherence_score
        
        # Keyword density
        keyword_score = features.keyword_density * 0.2
        score += keyword_score
        contributing_features['keyword_density'] = keyword_score
        
        # Extended semantic features if available
        if hasattr(features, 'extended_features'):
            ext_features = features.extended_features
            
            for feature_name, weight in self.semantic_weights.items():
                if feature_name in ext_features:
                    feature_score = ext_features[feature_name] * weight
                    score += feature_score
                    contributing_features[feature_name] = feature_score
        
        return min(score, 1.0), contributing_features


class LayoutScorer:
    """Scores blocks based on layout and positioning"""
    
    def __init__(self):
        pass
    
    def score_block(self, features: BlockFeatures) -> Tuple[float, Dict[str, float]]:
        """Score a block based on layout features"""
        
        score = 0.0
        contributing_features = {}
        
        # Isolation score (isolated blocks more likely to be headings)
        isolation_contribution = features.isolation_score * 0.3
        score += isolation_contribution
        contributing_features['isolation_score'] = isolation_contribution
        
        # Centering (centered text often headings)
        centering_contribution = features.is_centered * 0.2
        score += centering_contribution
        contributing_features['is_centered'] = centering_contribution
        
        # Left alignment with common margins
        # Assume we have margin consistency info in extended features
        if hasattr(features, 'extended_features'):
            margin_consistency = features.extended_features.get('margin_consistency', 0.0)
            margin_contribution = margin_consistency * 0.15
            score += margin_contribution
            contributing_features['margin_consistency'] = margin_contribution
        
        # Position on page (headings often at top)
        if features.y_position < 0.2:  # Top 20% of page
            top_position_bonus = 0.1
            score += top_position_bonus
            contributing_features['top_position_bonus'] = top_position_bonus
        
        # Indentation (headings usually not heavily indented)
        if features.indentation_level < 1.0:  # Low indentation
            indentation_bonus = (1.0 - features.indentation_level) * 0.1
            score += indentation_bonus
            contributing_features['indentation_bonus'] = indentation_bonus
        
        return min(score, 1.0), contributing_features


class FontScorer:
    """Scores blocks based on font characteristics"""
    
    def __init__(self):
        pass
    
    def score_block(self, features: BlockFeatures) -> Tuple[float, Dict[str, float]]:
        """Score a block based on font features"""
        
        score = 0.0
        contributing_features = {}
        
        # Font size ratio (larger fonts more likely headings)
        if features.font_size_ratio > 1.0:
            size_contribution = min((features.font_size_ratio - 1.0) * 0.5, 0.4)
            score += size_contribution
            contributing_features['font_size_ratio'] = size_contribution
        
        # Font size percentile
        percentile_contribution = features.font_size_percentile * 0.3
        score += percentile_contribution
        contributing_features['font_size_percentile'] = percentile_contribution
        
        # Bold formatting
        bold_contribution = features.is_bold * 0.2
        score += bold_contribution
        contributing_features['is_bold'] = bold_contribution
        
        # Italic (less common for headings, small bonus)
        italic_contribution = features.is_italic * 0.05
        score += italic_contribution
        contributing_features['is_italic'] = italic_contribution
        
        # Underlined (moderate indicator)
        underline_contribution = features.is_underlined * 0.1
        score += underline_contribution
        contributing_features['is_underlined'] = underline_contribution
        
        return min(score, 1.0), contributing_features


class LevelClassifier:
    """Classifies heading level based on features"""
    
    def __init__(self):
        # Simple rule-based level classification
        # In production, this could be a trained classifier
        pass
    
    def classify_level(self, features: BlockFeatures, 
                      heading_confidence: float) -> Tuple[HeadingLevel, float]:
        """Classify the heading level"""
        
        # Pattern-based level detection (highest priority)
        if hasattr(features, 'extended_features'):
            ext_features = features.extended_features
            
            # Check for explicit level patterns
            if ext_features.get('pattern_chapter', 0) > 0.5:
                return HeadingLevel.H1, 0.95
            elif ext_features.get('pattern_section', 0) > 0.5:
                return HeadingLevel.H1, 0.90
            elif ext_features.get('pattern_numbered_section', 0) > 0.5:
                return HeadingLevel.H1, 0.85
            elif ext_features.get('pattern_numbered_subsection', 0) > 0.5:
                return HeadingLevel.H2, 0.85
            elif ext_features.get('pattern_numbered_subsubsection', 0) > 0.5:
                return HeadingLevel.H3, 0.80
        
        # Numbering level detection
        if features.numbering_level > 0:
            level_map = {1: HeadingLevel.H1, 2: HeadingLevel.H2, 3: HeadingLevel.H3}
            if features.numbering_level <= 3:
                return level_map[int(features.numbering_level)], 0.8
        
        # Font-based level detection
        if features.font_size_ratio > 1.5 and heading_confidence > 0.7:
            return HeadingLevel.H1, 0.7
        elif features.font_size_ratio > 1.2 and heading_confidence > 0.6:
            return HeadingLevel.H2, 0.6
        elif features.font_size_ratio > 1.0 and heading_confidence > 0.5:
            return HeadingLevel.H3, 0.5
        
        # Default based on confidence
        if heading_confidence > 0.8:
            return HeadingLevel.H1, 0.6
        elif heading_confidence > 0.6:
            return HeadingLevel.H2, 0.5
        else:
            return HeadingLevel.H3, 0.4


class EnsembleScorer:
    """Main ensemble scoring system"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Initialize individual scorers
        self.pattern_scorer = PatternScorer()
        self.semantic_scorer = SemanticScorer()
        self.layout_scorer = LayoutScorer()
        self.font_scorer = FontScorer()
        self.level_classifier = LevelClassifier()
        
        # Ensemble weights (can be tuned based on validation)
        self.weights = weights or {
            'pattern': 0.35,    # Highest weight - patterns are most reliable
            'semantic': 0.25,   # Semantic analysis
            'font': 0.25,      # Font characteristics
            'layout': 0.15,    # Layout features
        }
        
        # Confidence threshold for heading detection
        self.heading_threshold = 0.4
        
    def score_blocks(self, blocks: List[TextBlock], 
                    features_list: List[BlockFeatures]) -> List[HeadingPrediction]:
        """Score all blocks and generate predictions"""
        
        predictions = []
        
        for i, (block, features) in enumerate(zip(blocks, features_list)):
            # Get scores from individual scorers
            pattern_score, pattern_details = self.pattern_scorer.score_block(features)
            semantic_score, semantic_details = self.semantic_scorer.score_block(features)
            layout_score, layout_details = self.layout_scorer.score_block(features)
            font_score, font_details = self.font_scorer.score_block(features)
            
            # Compute ensemble score
            ensemble_score = (
                pattern_score * self.weights['pattern'] +
                semantic_score * self.weights['semantic'] +
                font_score * self.weights['font'] +
                layout_score * self.weights['layout']
            )
            
            # Apply text length penalty for very long blocks
            if features.text_length > 300:  # Headings are usually shorter
                length_penalty = min((features.text_length - 300) / 1000, 0.3)
                ensemble_score *= (1.0 - length_penalty)
            
            # Apply word count bonus for reasonable heading lengths
            if 2 <= features.word_count <= 15:
                ensemble_score *= 1.1
            
            # Determine if it's a heading
            is_heading = ensemble_score > self.heading_threshold
            
            # Classify level if it's a heading
            if is_heading:
                predicted_level, level_confidence = self.level_classifier.classify_level(
                    features, ensemble_score
                )
            else:
                predicted_level = HeadingLevel.H3  # Default
                level_confidence = 0.0
            
            # Collect top contributing features for interpretability
            all_feature_contributions = {}
            all_feature_contributions.update({f"pattern_{k}": v for k, v in pattern_details.items()})
            all_feature_contributions.update({f"semantic_{k}": v for k, v in semantic_details.items()})
            all_feature_contributions.update({f"layout_{k}": v for k, v in layout_details.items()})
            all_feature_contributions.update({f"font_{k}": v for k, v in font_details.items()})
            
            top_features = sorted(all_feature_contributions.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            prediction = HeadingPrediction(
                block_idx=i,
                is_heading=is_heading,
                confidence=ensemble_score,
                predicted_level=predicted_level,
                level_confidence=level_confidence,
                pattern_score=pattern_score,
                semantic_score=semantic_score,
                layout_score=layout_score,
                font_score=font_score,
                ensemble_score=ensemble_score,
                top_features=top_features
            )
            
            predictions.append(prediction)
        
        # Post-process predictions for consistency
        predictions = self._post_process_predictions(predictions, blocks)
        
        return predictions
    
    def _post_process_predictions(self, predictions: List[HeadingPrediction],
                                blocks: List[TextBlock]) -> List[HeadingPrediction]:
        """Post-process predictions for consistency and hierarchy validation"""
        
        # Filter out very low confidence predictions
        for pred in predictions:
            if pred.confidence < 0.3:
                pred.is_heading = False
        
        # Ensure reasonable heading density (not too many headings)
        heading_count = sum(1 for p in predictions if p.is_heading)
        total_blocks = len(predictions)
        
        if heading_count > total_blocks * 0.3:  # More than 30% headings is suspicious
            # Keep only the highest confidence headings
            heading_predictions = [(i, p) for i, p in enumerate(predictions) if p.is_heading]
            heading_predictions.sort(key=lambda x: x[1].confidence, reverse=True)
            
            # Keep top 30% as headings
            keep_count = int(total_blocks * 0.3)
            keep_indices = set(i for i, _ in heading_predictions[:keep_count])
            
            for i, pred in enumerate(predictions):
                if pred.is_heading and i not in keep_indices:
                    pred.is_heading = False
        
        # Validate heading hierarchy
        predictions = self._validate_hierarchy(predictions)
        
        return predictions
    
    def _validate_hierarchy(self, predictions: List[HeadingPrediction]) -> List[HeadingPrediction]:
        """Validate and correct heading hierarchy"""
        
        heading_predictions = [p for p in predictions if p.is_heading]
        
        if not heading_predictions:
            return predictions
        
        # Simple hierarchy correction
        level_order = [HeadingLevel.H1, HeadingLevel.H2, HeadingLevel.H3]
        last_level_index = -1
        
        for pred in heading_predictions:
            current_level_index = level_order.index(pred.predicted_level)
            
            # Don't allow jumps of more than 1 level
            if current_level_index > last_level_index + 1:
                # Adjust to next logical level
                corrected_index = min(last_level_index + 1, len(level_order) - 1)
                pred.predicted_level = level_order[corrected_index]
                pred.level_confidence *= 0.9  # Reduce confidence for corrections
            
            last_level_index = level_order.index(pred.predicted_level)
        
        return predictions
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update ensemble weights (for tuning)"""
        self.weights.update(new_weights)
    
    def set_threshold(self, threshold: float):
        """Update heading detection threshold"""
        self.heading_threshold = threshold


# Utility functions for evaluation and tuning
def evaluate_predictions(predictions: List[HeadingPrediction], 
                        ground_truth: List[Dict]) -> Dict[str, float]:
    """Evaluate predictions against ground truth"""
    
    # Convert predictions to simple format for comparison
    pred_headings = []
    for pred in predictions:
        if pred.is_heading:
            pred_headings.append({
                'level': pred.predicted_level.value,
                'confidence': pred.confidence
            })
    
    # Simple precision/recall calculation
    # (This would be more sophisticated in practice)
    tp = len([p for p in pred_headings if p['confidence'] > 0.5])
    fp = len([p for p in pred_headings if p['confidence'] <= 0.5])
    fn = max(0, len(ground_truth) - tp)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'total_predictions': len(pred_headings),
        'total_ground_truth': len(ground_truth)
    }


# Example usage
if __name__ == "__main__":
    ensemble = EnsembleScorer()
    
    print("Multi-Signal Ensemble Scorer Ready")
    print("Components: Pattern scorer, Semantic scorer, Layout scorer, Font scorer")
    print("Features: Weighted ensemble, hierarchy validation, confidence calibration")
    print("Tunable: Weights, thresholds, post-processing rules")