#!/usr/bin/env python3
"""Duplication Detection Validator - Refactored Version.

Prevents AI from creating duplicate code/files using base validator classes.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from .base_validators import DuplicationDetector, FileOperationValidator, TaskAnalysisValidator
from ..core.workflow_validator import (
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class RefactoredDuplicationDetectionValidator(DuplicationDetector):
    """Prevents AI from creating duplicate code/files using base class functionality."""
    
    def __init__(self, priority: int = 860):
        super().__init__(priority)
        
        # Add task duplication patterns
        self.duplication_keywords = [
            "reimplement", "recreate", "rebuild", "rewrite from scratch",
            "create another", "make a new", "build a second", "duplicate"
        ]
    
    def get_validator_name(self) -> str:
        return "refactored_duplication_detection_validator"
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
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
        
        # Check for duplicate suffixes using base class method
        duplicate_suffix = self.has_duplicate_suffix(file_name)
        if duplicate_suffix:
            return self.create_blocking_result(
                message=f"🚨 DUPLICATION DETECTED: File '{path.name}' appears to be a duplicate (contains '{duplicate_suffix}')",
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                blocking_reason="Creating duplicate files causes confusion and maintenance issues",
                alternative="Update the existing file instead of creating a duplicate",
                guidance="Queen ZEN recommends checking existing files before creating new ones",
                priority=85
            )
        
        # Check if similar file already exists using base class method
        if path.parent.exists():
            existing_files = list(path.parent.glob(f"*{path.suffix}"))
            similar_files = self.find_similar_files(file_name, existing_files)
            
            if similar_files:
                similar_names = ", ".join([f.name for f in similar_files[:3]])
                return self.create_warning_result(
                    message=f"⚠️ Similar files already exist: {similar_names}",
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    alternative="Consider updating one of the existing files instead",
                    guidance="Check if functionality can be added to existing files",
                    priority=60
                )
        
        return None
    
    def _validate_code_duplication(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if code being added might duplicate existing functionality."""
        content = tool_input.get("content") or tool_input.get("new_string", "")
        
        if not content:
            return None
        
        # Extract function/class names
        import re
        function_pattern = r'(?:def|function|const|class)\s+(\w+)'
        matches = re.findall(function_pattern, content)
        
        for match in matches:
            # Check for duplicate prefixes using base class method
            duplicate_prefix = self.has_duplicate_prefix(match)
            if duplicate_prefix:
                return self.create_warning_result(
                    message=f"⚠️ Function '{match}' has duplicate prefix '{duplicate_prefix}' - might duplicate existing functionality",
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    alternative="Search for existing implementations before creating new functions",
                    guidance="Use mcp__zen__chat to analyze existing code patterns first",
                    priority=50
                )
        
        return None
    
    def _validate_task_duplication(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if a task might duplicate existing functionality."""
        description = tool_input.get("description", "").lower()
        
        # Check for duplication keywords
        for keyword in self.duplication_keywords:
            if keyword in description:
                return self.create_warning_result(
                    message=f"⚠️ Task suggests reimplementation: '{keyword}' detected",
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    alternative="mcp__zen__thinkdeep to analyze existing implementation first",
                    guidance="Queen ZEN can identify existing functionality to extend rather than duplicate",
                    priority=55
                )
        
        return None