"""Rogue system validator to prevent AI from creating parallel/competing systems.

Detects when AI attempts to:
- Create new frameworks alongside existing ones
- Build redundant service/API layers
- Implement duplicate state management systems
- Create competing authentication/authorization systems
"""

import re
from typing import Dict, Any, Optional, List, Set
from pathlib import Path

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class RogueSystemValidator(HiveWorkflowValidator):
    """Prevents AI from creating parallel competing systems."""
    
    # Known system patterns and their common variants
    SYSTEM_PATTERNS = {
        'state_management': {
            'patterns': ['store', 'state', 'redux', 'mobx', 'context', 'provider', 'vuex', 'pinia'],
            'files': ['store.', 'state.', 'context.', 'provider.'],
            'message': "State management system already exists"
        },
        'authentication': {
            'patterns': ['auth', 'login', 'session', 'jwt', 'oauth', 'passport', 'authentication'],
            'files': ['auth.', 'login.', 'session.', 'authentication.'],
            'message': "Authentication system already exists"
        },
        'routing': {
            'patterns': ['router', 'routes', 'routing', 'navigation', 'route'],
            'files': ['router.', 'routes.', 'routing.'],
            'message': "Routing system already exists"
        },
        'api_layer': {
            'patterns': ['api', 'service', 'client', 'fetch', 'axios', 'request'],
            'files': ['api.', 'service.', 'client.'],
            'message': "API service layer already exists"
        },
        'database': {
            'patterns': ['database', 'db', 'orm', 'model', 'schema', 'migration'],
            'files': ['database.', 'db.', 'models/', 'migrations/'],
            'message': "Database layer already exists"
        },
        'configuration': {
            'patterns': ['config', 'settings', 'env', 'environment', 'constants'],
            'files': ['config.', 'settings.', '.env', 'constants.'],
            'message': "Configuration system already exists"
        },
        'logging': {
            'patterns': ['logger', 'log', 'logging', 'winston', 'pino', 'bunyan'],
            'files': ['logger.', 'logging.', 'log.'],
            'message': "Logging system already exists"
        },
        'event_system': {
            'patterns': ['event', 'emitter', 'pubsub', 'observer', 'bus', 'dispatcher'],
            'files': ['events.', 'emitter.', 'bus.'],
            'message': "Event system already exists"
        }
    }
    
    def __init__(self, priority: int = 850):
        super().__init__(priority)
        self._existing_systems: Dict[str, Set[str]] = {}
        self._project_analyzed = False
    
    def get_validator_name(self) -> str:
        return "rogue_system_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for potential rogue system creation."""
        
        # Analyze project structure on first run
        if not self._project_analyzed:
            self._analyze_project_structure()
        
        # Check file creation for new systems
        if tool_name in ["Write", "mcp__filesystem__write_file"]:
            return self._validate_new_system_creation(tool_input)
        
        # Check task descriptions for system creation
        elif tool_name == "Task":
            return self._validate_task_system_creation(tool_input)
        
        # Check multi-file operations for framework creation
        elif tool_name == "mcp__claude-flow__task_orchestrate":
            return self._validate_framework_creation(tool_input)
        
        return None
    
    def _analyze_project_structure(self) -> None:
        """Analyze existing project structure to identify current systems."""
        # This is a simplified version - in production, would do deeper analysis
        project_root = Path("/home/devcontainers/flowed")
        
        for system_type, config in self.SYSTEM_PATTERNS.items():
            self._existing_systems[system_type] = set()
            
            # Check for files indicating this system exists
            for pattern in config['files']:
                if pattern.endswith('/'):
                    # Directory pattern
                    dirs = list(project_root.rglob(pattern.rstrip('/')))
                    if dirs:
                        self._existing_systems[system_type].add(pattern)
                else:
                    # File pattern
                    files = list(project_root.rglob(f"*{pattern}*"))
                    if files:
                        self._existing_systems[system_type].add(pattern)
        
        self._project_analyzed = True
    
    def _validate_new_system_creation(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if a new file represents a competing system."""
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        content = tool_input.get("content", "")
        
        if not file_path:
            return None
        
        path = Path(file_path)
        file_name = path.name.lower()
        
        # Check each system type
        for system_type, config in self.SYSTEM_PATTERNS.items():
            # Skip if this system doesn't exist yet
            if not self._existing_systems.get(system_type):
                continue
            
            # Check if file name suggests new system
            for pattern in config['patterns']:
                if pattern in file_name and not self._is_extending_existing(file_name, content):
                    return ValidationResult(
                        severity=ValidationSeverity.BLOCK,
                        violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                        message=f"🚨 ROGUE SYSTEM DETECTED: {config['message']} - attempting to create '{path.name}'",
                        suggested_alternative=f"Extend existing {system_type} instead of creating a new one",
                        blocking_reason="Creating parallel systems causes architectural conflicts and maintenance nightmares",
                        hive_guidance=f"Queen ZEN recommends analyzing existing {system_type} with mcp__zen__thinkdeep first",
                        priority_score=90
                    )
            
            # Check content for system initialization patterns
            if self._contains_system_initialization(content, config['patterns']):
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"⚠️ Potential parallel {system_type} initialization detected",
                    suggested_alternative=f"Use existing {system_type} infrastructure",
                    hive_guidance="Check existing patterns before creating new systems",
                    priority_score=70
                )
        
        return None
    
    def _validate_task_system_creation(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if task description suggests creating a competing system."""
        description = tool_input.get("description", "").lower()
        
        # Keywords suggesting new system creation
        system_keywords = [
            "build a new", "create a new", "implement a new", "design a new",
            "set up a new", "establish a new", "framework", "architecture"
        ]
        
        for keyword in system_keywords:
            if keyword in description:
                # Check which systems might be duplicated
                for system_type, config in self.SYSTEM_PATTERNS.items():
                    if self._existing_systems.get(system_type) and \
                       any(pattern in description for pattern in config['patterns']):
                        return ValidationResult(
                            severity=ValidationSeverity.WARN,
                            violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                            message=f"⚠️ Task suggests creating new {system_type} - {config['message']}",
                            suggested_alternative=f"Task agent to analyze and extend existing {system_type}",
                            hive_guidance="Use existing infrastructure rather than creating parallel systems",
                            priority_score=65
                        )
        
        return None
    
    def _validate_framework_creation(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Check if orchestrated task might create a competing framework."""
        task = tool_input.get("task", "").lower()
        
        # Patterns suggesting framework creation
        framework_patterns = [
            r"create.*framework", r"build.*architecture", r"implement.*system",
            r"design.*infrastructure", r"set up.*platform"
        ]
        
        for pattern in framework_patterns:
            if re.search(pattern, task):
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message="🚨 FRAMEWORK CREATION DETECTED: Attempting to create new architectural framework",
                    suggested_alternative="mcp__zen__analyze to understand existing architecture first",
                    blocking_reason="New frameworks conflict with existing architecture and create technical debt",
                    hive_guidance="Queen ZEN must approve all architectural decisions to maintain hive coherence",
                    priority_score=95
                )
        
        return None
    
    def _is_extending_existing(self, file_name: str, content: str) -> bool:
        """Check if file is extending existing system rather than creating new one."""
        extension_patterns = [
            'plugin', 'extension', 'addon', 'module', 'component',
            'middleware', 'interceptor', 'decorator', 'mixin'
        ]
        
        # Check file name
        for pattern in extension_patterns:
            if pattern in file_name:
                return True
        
        # Check content for extension patterns
        extension_code_patterns = [
            r'extends\s+\w+', r'implements\s+\w+', r'plugin\s*\(',
            r'middleware\s*\(', r'decorator\s*\(', r'mixin\s*\('
        ]
        
        for pattern in extension_code_patterns:
            if re.search(pattern, content):
                return True
        
        return False
    
    def _contains_system_initialization(self, content: str, patterns: List[str]) -> bool:
        """Check if content contains system initialization patterns."""
        init_patterns = [
            r'new\s+\w*(?:' + '|'.join(patterns) + ')',
            r'create(?:' + '|'.join(patterns) + ')',
            r'init(?:' + '|'.join(patterns) + ')',
            r'setup(?:' + '|'.join(patterns) + ')'
        ]
        
        for pattern in init_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        
        return False