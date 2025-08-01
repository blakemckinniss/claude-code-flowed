"""Base analyzer for pattern matching and context generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import re


@dataclass
class PatternMatch:
    """Represents a matched pattern with metadata."""
    pattern: str
    message: str
    priority: int = 0
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Analyzer(ABC):
    """Abstract base class for analyzers."""
    
    def __init__(self, priority: int = 0):
        self.priority = priority
        self.patterns: List[Tuple[str, str, Dict[str, Any]]] = []
        self._initialize_patterns()
    
    @abstractmethod
    def _initialize_patterns(self) -> None:
        """Initialize patterns for this analyzer."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this analyzer."""
        pass
    
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze the prompt and return matches."""
        matches = []
        
        for pattern, message, metadata in self.patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                match = PatternMatch(
                    pattern=pattern,
                    message=message,
                    priority=self.priority,
                    metadata=metadata
                )
                matches.append(match)
        
        return matches
    
    def add_pattern(self, pattern: str, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a pattern to this analyzer."""
        if metadata is None:
            metadata = {}
        self.patterns.append((pattern, message, metadata))