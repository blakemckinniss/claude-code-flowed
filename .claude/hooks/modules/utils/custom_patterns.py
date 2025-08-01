"""Custom pattern loader for user-defined patterns."""

import json
import os
from typing import List, Dict, Any, Optional
from ..core import Analyzer


class CustomPatternAnalyzer(Analyzer):
    """Analyzer for custom user-defined patterns."""
    
    def __init__(self, patterns: List[Dict[str, Any]], priority: int = 0):
        self.custom_patterns = patterns
        super().__init__(priority)
    
    def get_name(self) -> str:
        return "custom"
    
    def _initialize_patterns(self) -> None:
        """Initialize custom patterns."""
        for pattern_config in self.custom_patterns:
            pattern = pattern_config.get("pattern", "")
            message = pattern_config.get("message", "")
            metadata = pattern_config.get("metadata", {})
            
            if pattern and message:
                self.add_pattern(pattern, message, metadata)


class CustomPatternLoader:
    """Loads custom patterns from configuration."""
    
    @staticmethod
    def load_from_config(config: Dict[str, Any]) -> Optional[CustomPatternAnalyzer]:
        """Load custom patterns from configuration."""
        custom_patterns = config.get("custom_patterns", [])
        
        if not custom_patterns:
            return None
        
        priority = config.get("custom_pattern_priority", 75)
        return CustomPatternAnalyzer(custom_patterns, priority)
    
    @staticmethod
    def load_from_file(file_path: str) -> Optional[CustomPatternAnalyzer]:
        """Load custom patterns from a JSON file."""
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                patterns = json.load(f)
            
            if isinstance(patterns, list):
                return CustomPatternAnalyzer(patterns)
            elif isinstance(patterns, dict) and "patterns" in patterns:
                priority = patterns.get("priority", 75)
                return CustomPatternAnalyzer(patterns["patterns"], priority)
        except Exception:
            return None
        
        return None