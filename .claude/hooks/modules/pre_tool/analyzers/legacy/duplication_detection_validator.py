"""Duplication detection validator to prevent AI from creating duplicate code/files.

Detects when AI attempts to:
- Create files that already exist with similar names
- Implement functions/classes that duplicate existing functionality
- Create parallel systems that overlap with existing code
"""

import os
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
import json

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class DuplicationDetectionValidator(HiveWorkflowValidator):
    """Prevents AI from creating duplicate code/files."""
    
    # Common patterns that indicate duplication
    DUPLICATE_PATTERNS = {
        'file_suffixes': ['_copy', '_new', '_v2', '_temp', '_backup', '_old', '_duplicate'],
        'function_prefixes': ['new_', 'my_', 'custom_', 'alt_', 'another_'],
        'similar_names': {
            'index': ['main', 'app', 'entry'],
            'config': ['settings', 'configuration', 'setup', 'env'],
            'utils': ['helpers', 'utilities', 'tools', 'common'],
            'service': ['manager', 'handler', 'controller', 'provider']
        }
    }
    
    def get_validator_name(self) -> str:
        return "duplication_detection_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for potential duplication."""
        
        # Check file creation operations
        if tool_name in ["Write", "mcp__filesystem__write_file"]:
            return self._validate_file_duplication(tool_input)
        
        # Check code editing operations
        elif tool_name in ["Edit", "MultiEdit", "mcp__filesystem__edit_file"]:
            return self._validate_code_duplication(tool_input)
        
        # Check task descriptions for duplicate functionality
        elif tool_name == "Task":
            return self._validate_task_duplication(tool_input)
        
        return None
    
    def _validate_file_duplication(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if a file being created might be a duplicate."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        
        if not file_path:
            return None
        
        path = Path(file_path)
        file_name = path.stem
        file_ext = path.suffix
        
        # Check for duplicate suffixes
        for suffix in self.DUPLICATE_PATTERNS['file_suffixes']:
            if suffix in file_name.lower():
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"🚨 DUPLICATION DETECTED: File '{path.name}' appears to be a duplicate (contains '{suffix}')",
                    suggested_alternative="Update the existing file instead of creating a duplicate",
                    blocking_reason="Creating duplicate files causes confusion and maintenance issues",
                    hive_guidance="Queen ZEN recommends checking existing files before creating new ones",
                    priority_score=85
                )
        
        # Check if similar file already exists
        if path.parent.exists():
            existing_files = list(path.parent.glob(f"*{file_ext}"))
            similar_files = self._find_similar_files(file_name, existing_files)
            
            if similar_files:
                similar_names = ", ".join([f.name for f in similar_files[:3]])
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"⚠️ Similar files already exist: {similar_names}",
                    suggested_alternative="Consider updating one of the existing files instead",
                    hive_guidance="Check if functionality can be added to existing files",
                    priority_score=60
                )
        
        return None
    
    def _validate_code_duplication(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if code being added might duplicate existing functionality."""
        content = tool_input.get("content") or tool_input.get("new_string", "")
        
        # Check for function/class definitions with duplicate prefixes
        function_pattern = r'(?:def|function|const|class)\s+(\w+)'
        matches = re.findall(function_pattern, content)
        
        for match in matches:
            func_name = match.lower()
            
            # Check for duplicate prefixes
            for prefix in self.DUPLICATE_PATTERNS['function_prefixes']:
                if func_name.startswith(prefix):
                    return ValidationResult(
                        severity=ValidationSeverity.WARN,
                        violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                        message=f"⚠️ Function '{match}' has duplicate prefix '{prefix}' - might duplicate existing functionality",
                        suggested_alternative="Search for existing implementations before creating new functions",
                        hive_guidance="Use mcp__zen__chat to analyze existing code patterns first",
                        priority_score=50
                    )
        
        return None
    
    def _validate_task_duplication(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if a task might duplicate existing functionality."""
        description = tool_input.get("description", "").lower()
        
        # Keywords that suggest reimplementation
        duplication_keywords = [
            "reimplement", "recreate", "rebuild", "rewrite from scratch",
            "create another", "make a new", "build a second", "duplicate"
        ]
        
        for keyword in duplication_keywords:
            if keyword in description:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"⚠️ Task suggests reimplementation: '{keyword}' detected",
                    suggested_alternative="mcp__zen__thinkdeep to analyze existing implementation first",
                    hive_guidance="Queen ZEN can identify existing functionality to extend rather than duplicate",
                    priority_score=55
                )
        
        return None
    
    def _find_similar_files(self, target_name: str, existing_files: List[Path]) -> List[Path]:
        """Find files with similar names using various heuristics."""
        similar_files = []
        target_lower = target_name.lower()
        
        for file in existing_files:
            file_stem = file.stem.lower()
            
            # Check exact match (ignoring case)
            if file_stem == target_lower:
                similar_files.append(file)
                continue
            
            # Check if one contains the other
            if target_lower in file_stem or file_stem in target_lower:
                similar_files.append(file)
                continue
            
            # Check semantic similarity
            for _category, similar_terms in self.DUPLICATE_PATTERNS['similar_names'].items():
                if target_lower in similar_terms and file_stem in similar_terms:
                    similar_files.append(file)
                    break
            
            # Check Levenshtein distance for typos
            if self._levenshtein_distance(target_lower, file_stem) <= 2:
                similar_files.append(file)
        
        return similar_files
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]