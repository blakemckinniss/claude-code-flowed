#!/usr/bin/env python3
"""Base validator classes for common patterns across analyzers.

This module provides reusable base classes that eliminate duplication
across the various validator implementations.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class BaseHiveValidator(HiveWorkflowValidator, ABC):
    """Enhanced base class for all hive validators with common functionality."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self._enabled = True
        self._name = self.get_validator_name()
    
    @property
    def enabled(self) -> bool:
        """Check if validator is enabled."""
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable validator."""
        self._enabled = value
    
    def create_suggestion_result(self, message: str, guidance: Optional[str] = None, 
                               alternative: Optional[str] = None, priority: int = 50) -> ValidationResult:
        """Create a standard suggestion result."""
        return ValidationResult(
            severity=ValidationSeverity.SUGGEST,
            violation_type=None,
            message=message,
            suggested_alternative=alternative,
            hive_guidance=guidance,
            priority_score=priority
        )
    
    def create_warning_result(self, message: str, violation_type: WorkflowViolationType,
                            guidance: Optional[str] = None, alternative: Optional[str] = None, 
                            priority: int = 70) -> ValidationResult:
        """Create a standard warning result."""
        return ValidationResult(
            severity=ValidationSeverity.WARN,
            violation_type=violation_type,
            message=message,
            suggested_alternative=alternative,
            hive_guidance=guidance,
            priority_score=priority
        )
    
    def create_blocking_result(self, message: str, violation_type: WorkflowViolationType,
                             blocking_reason: str, guidance: Optional[str] = None, 
                             alternative: Optional[str] = None, priority: int = 90) -> ValidationResult:
        """Create a standard blocking result."""
        return ValidationResult(
            severity=ValidationSeverity.BLOCK,
            violation_type=violation_type,
            message=message,
            suggested_alternative=alternative,
            blocking_reason=blocking_reason,
            hive_guidance=guidance,
            priority_score=priority
        )
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Template method that checks enabled status before validating."""
        if not self.enabled:
            return None
        
        return self._validate_workflow_impl(tool_name, tool_input, context)
    
    @abstractmethod
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Implement the actual validation logic."""
        pass


class PatternMatchingValidator(BaseHiveValidator):
    """Base class for validators that use pattern matching."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self._patterns: Dict[str, List[str]] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
    
    def add_patterns(self, category: str, patterns: List[str]) -> None:
        """Add patterns for a category."""
        self._patterns[category] = patterns
        self._compiled_patterns[category] = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def match_patterns(self, text: str, category: str) -> List[Tuple[str, re.Match]]:
        """Match text against patterns in a category."""
        matches = []
        if category not in self._compiled_patterns:
            return matches
        
        for i, pattern in enumerate(self._compiled_patterns[category]):
            match = pattern.search(text)
            if match:
                matches.append((self._patterns[category][i], match))
        
        return matches
    
    def has_pattern_match(self, text: str, category: str) -> bool:
        """Check if text matches any pattern in category."""
        return len(self.match_patterns(text, category)) > 0


class ToolSpecificValidator(BaseHiveValidator):
    """Base class for validators that target specific tools."""
    
    def __init__(self, target_tools: Set[str], priority: int = 500):
        super().__init__(priority)
        self.target_tools = target_tools
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Only validate targeted tools."""
        if tool_name not in self.target_tools:
            return None
        
        return self._validate_target_tool(tool_name, tool_input, context)
    
    @abstractmethod
    def _validate_target_tool(self, tool_name: str, tool_input: Dict[str, Any], 
                            context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate the specific target tool."""
        pass


class FileOperationValidator(ToolSpecificValidator):
    """Base class for validators that work with file operations."""
    
    def __init__(self, priority: int = 500):
        super().__init__({"Write", "Edit", "MultiEdit", "mcp__filesystem__write_file", 
                         "mcp__filesystem__edit_file"}, priority)
    
    def get_file_path(self, tool_input: Dict[str, Any]) -> Optional[str]:
        """Extract file path from tool input."""
        return tool_input.get("file_path") or tool_input.get("path")
    
    def get_file_content(self, tool_input: Dict[str, Any]) -> Optional[str]:
        """Extract file content from tool input."""
        return tool_input.get("content") or tool_input.get("new_string")
    
    def is_write_operation(self, tool_name: str) -> bool:
        """Check if tool is a write operation."""
        return tool_name in {"Write", "mcp__filesystem__write_file"}
    
    def is_edit_operation(self, tool_name: str) -> bool:
        """Check if tool is an edit operation."""
        return tool_name in {"Edit", "MultiEdit", "mcp__filesystem__edit_file"}


class BatchingValidator(BaseHiveValidator):
    """Base class for validators that enforce batching patterns."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self.min_batch_size = 5
        self.optimal_batch_size = 10
    
    def check_batch_violation(self, items: List[Any], operation_name: str) -> Optional[ValidationResult]:
        """Check if operation violates batching rules."""
        if len(items) < self.min_batch_size:
            return self.create_warning_result(
                message=f"🚨 BATCH VIOLATION: {operation_name} should include {self.min_batch_size}-{self.optimal_batch_size}+ items in ONE call!",
                violation_type=WorkflowViolationType.FRAGMENTED_WORKFLOW,
                alternative=f"Batch ALL {operation_name.lower()} together in ONE call",
                guidance=f"✅ CORRECT: {operation_name} with {self.min_batch_size}-{self.optimal_batch_size}+ items with all statuses/priorities",
                priority=95
            )
        return None


class MCPToolValidator(ToolSpecificValidator):
    """Base class for validators that work with MCP tools."""
    
    def __init__(self, mcp_prefixes: Set[str], priority: int = 500):
        super().__init__(set(), priority)  # Initialize with empty set, override in _validate_workflow_impl
        self.mcp_prefixes = mcp_prefixes
    
    def _validate_workflow_impl(self, tool_name: str, tool_input: Dict[str, Any], 
                               context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Check if tool is an MCP tool we should validate."""
        if not any(tool_name.startswith(prefix) for prefix in self.mcp_prefixes):
            return None
        
        return self._validate_mcp_tool(tool_name, tool_input, context)
    
    @abstractmethod
    def _validate_mcp_tool(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate the specific MCP tool."""
        pass


class TaskAnalysisValidator(ToolSpecificValidator):
    """Base class for validators that analyze task descriptions."""
    
    def __init__(self, priority: int = 500):
        super().__init__({"Task"}, priority)
        self.task_patterns: Dict[str, List[str]] = {}
    
    def add_task_patterns(self, category: str, patterns: List[str]) -> None:
        """Add task analysis patterns."""
        self.task_patterns[category] = patterns
    
    def detect_task_type(self, description: str) -> Optional[str]:
        """Detect task type from description."""
        if not description:
            return None
        
        desc_lower = description.lower()
        for category, patterns in self.task_patterns.items():
            for pattern in patterns:
                if re.search(pattern, desc_lower, re.IGNORECASE):
                    return category
        
        return None
    
    def get_task_description(self, tool_input: Dict[str, Any]) -> str:
        """Extract task description from tool input."""
        return tool_input.get("description", "") or tool_input.get("prompt", "")


class VisualFormatProvider(BaseHiveValidator):
    """Base class for validators that provide visual formatting templates."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self.format_templates: Dict[str, str] = {}
    
    def add_format_template(self, name: str, template: str) -> None:
        """Add a visual format template."""
        self.format_templates[name] = template
    
    def get_format_template(self, name: str) -> Optional[str]:
        """Get a visual format template."""
        return self.format_templates.get(name)
    
    def create_format_suggestion(self, message: str, template_name: str, 
                               priority: int = 50) -> ValidationResult:
        """Create a formatting suggestion with template."""
        template = self.get_format_template(template_name)
        return self.create_suggestion_result(
            message=message,
            guidance=template,
            priority=priority
        )


class DuplicationDetector(BaseHiveValidator):
    """Base class for validators that detect code/file duplication."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self.duplicate_suffixes = ['_copy', '_new', '_v2', '_temp', '_backup', '_old']
        self.duplicate_prefixes = ['new_', 'my_', 'custom_', 'alt_', 'another_']
        self.similar_names = {
            'index': ['main', 'app', 'entry'],
            'config': ['settings', 'configuration', 'setup', 'env'],
            'utils': ['helpers', 'utilities', 'tools', 'common'],
            'service': ['manager', 'handler', 'controller', 'provider']
        }
    
    def has_duplicate_suffix(self, name: str) -> Optional[str]:
        """Check if name has a duplicate suffix."""
        name_lower = name.lower()
        for suffix in self.duplicate_suffixes:
            if suffix in name_lower:
                return suffix
        return None
    
    def has_duplicate_prefix(self, name: str) -> Optional[str]:
        """Check if name has a duplicate prefix."""
        name_lower = name.lower()
        for prefix in self.duplicate_prefixes:
            if name_lower.startswith(prefix):
                return prefix
        return None
    
    def find_similar_files(self, target_name: str, existing_files: List[Path]) -> List[Path]:
        """Find files with similar names."""
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
            for _category, similar_terms in self.similar_names.items():
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


class ConfigurableValidator(BaseHiveValidator):
    """Base class for validators that support configuration."""
    
    def __init__(self, priority: int = 500):
        super().__init__(priority)
        self._config = {}
    
    def get_config(self) -> Dict[str, Any]:
        """Get validator configuration."""
        return {
            "name": self.get_validator_name(),
            "enabled": self.enabled,
            "priority": self.priority,
            "config": self._config
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Update validator configuration."""
        if "enabled" in config:
            self.enabled = bool(config["enabled"])
        if "priority" in config:
            self.priority = int(config["priority"])
        if "config" in config:
            self._config.update(config["config"])
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self._config[key] = value