"""Overwrite protection validator to prevent accidental file overwrites.

Detects when AI attempts to:
- Overwrite critical configuration files
- Replace important system files without backup
- Overwrite files without checking existing content
- Delete and recreate files unnecessarily
"""

import os
import re
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
import hashlib

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class OverwriteProtectionValidator(HiveWorkflowValidator):
    """Prevents accidental overwrites of important files."""
    
    # Critical files that should never be overwritten without explicit confirmation
    CRITICAL_FILES = {
        'package.json': "Project dependencies and configuration",
        'package-lock.json': "Dependency lock file",
        'yarn.lock': "Yarn dependency lock file",
        'pnpm-lock.yaml': "PNPM dependency lock file",
        '.env': "Environment variables",
        '.env.local': "Local environment variables",
        '.env.production': "Production environment variables",
        'tsconfig.json': "TypeScript configuration",
        'webpack.config.js': "Webpack configuration",
        'vite.config.js': "Vite configuration",
        'babel.config.js': "Babel configuration",
        '.gitignore': "Git ignore patterns",
        'docker-compose.yml': "Docker compose configuration",
        'Dockerfile': "Docker configuration",
        'requirements.txt': "Python dependencies",
        'Cargo.toml': "Rust project configuration",
        'go.mod': "Go module configuration",
        'composer.json': "PHP dependencies",
        'Gemfile': "Ruby dependencies",
        'pubspec.yaml': "Flutter/Dart configuration",
    }
    
    # File patterns that indicate important files
    IMPORTANT_PATTERNS = [
        r'.*\.config\.(js|ts|json)$',  # Config files
        r'.*rc\.(js|json|yml|yaml)$',   # RC files
        r'^\.',                          # Dotfiles
        r'.*\.(key|pem|cert|crt)$',    # Security files
        r'.*\.(db|sqlite|sql)$',        # Database files
        r'(main|index|app)\.(js|ts|py|go|rs)$',  # Entry points
    ]
    
    def __init__(self, priority: int = 900):
        super().__init__(priority)
        self._file_checksums: Dict[str, str] = {}
        self._recent_reads: Set[str] = set()
        self._write_without_read_threshold = 3  # Number of writes without reads before warning
        self._writes_without_reads = 0
    
    def get_validator_name(self) -> str:
        return "overwrite_protection_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for potential dangerous overwrites."""
        
        # Track file reads
        if tool_name in ["Read", "mcp__filesystem__read_text_file", "mcp__filesystem__read_file"]:
            self._track_file_read(tool_input)
        
        # Validate write operations
        elif tool_name in ["Write", "mcp__filesystem__write_file"]:
            return self._validate_file_overwrite(tool_input)
        
        # Check for delete operations
        elif tool_name == "Bash":
            return self._validate_bash_delete(tool_input)
        
        # Check multi-file operations
        elif tool_name == "mcp__github__push_files":
            return self._validate_multi_file_push(tool_input)
        
        return None
    
    def _track_file_read(self, tool_input: Dict[str, Any]) -> None:
        """Track that a file was read."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if file_path:
            self._recent_reads.add(file_path)
            # Keep only last 50 reads
            if len(self._recent_reads) > 50:
                self._recent_reads.pop()
    
    def _validate_file_overwrite(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate file write operations for dangerous overwrites."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        content = tool_input.get("content", "")
        
        if not file_path:
            return None
        
        path = Path(file_path)
        file_name = path.name
        
        # Check if this is a critical file
        if file_name in self.CRITICAL_FILES:
            return self._handle_critical_file_overwrite(path, file_name)
        
        # Check if file matches important patterns
        for pattern in self.IMPORTANT_PATTERNS:
            if re.match(pattern, file_name):
                return self._handle_important_file_overwrite(path, pattern)
        
        # Check if file exists and wasn't read first
        if path.exists() and str(path) not in self._recent_reads:
            self._writes_without_reads += 1
            
            # Generate warning after threshold
            if self._writes_without_reads >= self._write_without_read_threshold:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                    message=f"⚠️ Multiple files being overwritten without reading first ({self._writes_without_reads} files)",
                    suggested_alternative="Read files before overwriting to understand existing content",
                    hive_guidance="Queen ZEN recommends analyzing files before modification",
                    priority_score=70
                )
        else:
            # Reset counter on successful read-before-write
            self._writes_without_reads = 0
        
        # Check if content is being completely replaced
        if path.exists() and len(content) > 0:
            return self._check_content_replacement(path, content)
        
        return None
    
    def _handle_critical_file_overwrite(self, path: Path, file_name: str) -> ValidationResult:
        """Handle attempt to overwrite a critical file."""
        description = self.CRITICAL_FILES[file_name]
        
        return ValidationResult(
            severity=ValidationSeverity.BLOCK,
            violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
            message=f"🚨 CRITICAL FILE OVERWRITE BLOCKED: '{file_name}' - {description}",
            suggested_alternative="Use Edit to modify specific parts or create a backup first",
            blocking_reason=f"Overwriting {file_name} could break the entire project",
            hive_guidance="Critical files require careful modification with Queen ZEN's oversight",
            priority_score=100
        )
    
    def _handle_important_file_overwrite(self, path: Path, pattern: str) -> ValidationResult:
        """Handle attempt to overwrite an important file."""
        if not path.exists():
            return None  # Creating new file is OK
        
        # Check if file was read recently
        if str(path) not in self._recent_reads:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                message=f"⚠️ Overwriting important file '{path.name}' without reading it first",
                suggested_alternative="Read the file first to understand its current content",
                hive_guidance="Understanding existing content prevents accidental data loss",
                priority_score=75
            )
        
        return None
    
    def _check_content_replacement(self, path: Path, new_content: str) -> Optional[ValidationResult]:
        """Check if content is being completely replaced vs updated."""
        try:
            existing_content = path.read_text()
            
            # Calculate similarity
            if len(existing_content) > 100 and len(new_content) > 100:
                # Simple check: if new content doesn't contain any substantial part of old content
                overlap = self._calculate_content_overlap(existing_content, new_content)
                
                if overlap < 0.1:  # Less than 10% similarity
                    return ValidationResult(
                        severity=ValidationSeverity.WARN,
                        violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                        message=f"⚠️ Complete file replacement detected for '{path.name}'",
                        suggested_alternative="Use Edit to preserve existing content structure",
                        hive_guidance="Incremental changes are safer than complete replacements",
                        priority_score=65
                    )
        except Exception:
            pass
        
        return None
    
    def _validate_bash_delete(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate bash commands for dangerous delete operations."""
        command = tool_input.get("command", "")
        
        # Dangerous delete patterns
        dangerous_patterns = [
            (r'rm\s+-rf\s+(?!(/tmp|/var/tmp))', "Recursive force delete"),
            (r'rm\s+.*\.(json|yml|yaml|toml|lock)(\s|$)', "Deleting configuration files"),
            (r'rm\s+.*\*', "Wildcard deletion"),
            (r'find.*-delete', "Find and delete operation"),
            (r'truncate.*-s\s*0', "File truncation"),
        ]
        
        for pattern, description in dangerous_patterns:
            if re.search(pattern, command):
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                    message=f"🚨 DANGEROUS DELETE OPERATION: {description}",
                    suggested_alternative="Create backups before deletion or use safer alternatives",
                    blocking_reason="Irreversible deletion operations require explicit confirmation",
                    hive_guidance="Queen ZEN recommends backup strategies before destructive operations",
                    priority_score=95
                )
        
        return None
    
    def _validate_multi_file_push(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate multi-file push operations."""
        files = tool_input.get("files", [])
        
        critical_files_in_push = []
        for file_info in files:
            file_path = file_info.get("path", "")
            file_name = Path(file_path).name
            
            if file_name in self.CRITICAL_FILES:
                critical_files_in_push.append(file_name)
        
        if critical_files_in_push:
            files_list = ", ".join(critical_files_in_push[:3])
            return ValidationResult(
                severity=ValidationSeverity.BLOCK,
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                message=f"🚨 CRITICAL FILES IN PUSH: {files_list}",
                suggested_alternative="Review and push critical files separately with proper testing",
                blocking_reason="Pushing multiple critical files requires careful review",
                hive_guidance="Critical configuration changes should be isolated and tested",
                priority_score=90
            )
        
        # Check for large number of files
        if len(files) > 10:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.DANGEROUS_OPERATION,
                message=f"⚠️ Large batch push detected: {len(files)} files",
                suggested_alternative="Consider pushing files in smaller, logical groups",
                hive_guidance="Smaller commits are easier to review and revert if needed",
                priority_score=60
            )
        
        return None
    
    def _calculate_content_overlap(self, content1: str, content2: str) -> float:
        """Calculate rough overlap between two pieces of content."""
        # Simple line-based overlap calculation
        lines1 = set(content1.strip().split('\n'))
        lines2 = set(content2.strip().split('\n'))
        
        if not lines1:
            return 0.0
        
        common_lines = lines1.intersection(lines2)
        return len(common_lines) / len(lines1)