"""Conflicting architecture validator to ensure consistency with project patterns.

Detects when AI attempts to:
- Use inconsistent coding patterns or styles
- Mix incompatible frameworks or libraries
- Violate established architectural patterns
- Create inconsistent API designs
"""

import re
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
import json

from ..core.workflow_validator import (
    HiveWorkflowValidator,
    ValidationResult,
    ValidationSeverity,
    WorkflowViolationType,
    WorkflowContextTracker
)


class ConflictingArchitectureValidator(HiveWorkflowValidator):
    """Ensures consistency with established project patterns."""
    
    def __init__(self, priority: int = 825):
        super().__init__(priority)
        self._project_patterns: Dict[str, Any] = {}
        self._analyzed = False
    
    def get_validator_name(self) -> str:
        return "conflicting_architecture_validator"
    
    def validate_workflow(self, tool_name: str, tool_input: Dict[str, Any], 
                         context: WorkflowContextTracker) -> Optional[ValidationResult]:
        """Validate for architectural conflicts."""
        
        # Analyze project patterns on first run
        if not self._analyzed:
            self._analyze_project_patterns()
        
        # Check file operations for pattern violations
        if tool_name in ["Write", "Edit", "MultiEdit", "mcp__filesystem__write_file", "mcp__filesystem__edit_file"]:
            return self._validate_code_patterns(tool_name, tool_input)
        
        # Check task descriptions for architectural violations
        elif tool_name == "Task":
            return self._validate_task_architecture(tool_input)
        
        return None
    
    def _analyze_project_patterns(self) -> None:
        """Analyze project to detect established patterns."""
        project_root = Path("/home/devcontainers/flowed")
        
        # Detect import style (ES6 vs CommonJS)
        self._project_patterns['import_style'] = self._detect_import_style(project_root)
        
        # Detect async patterns (callbacks vs promises vs async/await)
        self._project_patterns['async_pattern'] = self._detect_async_pattern(project_root)
        
        # Detect naming conventions
        self._project_patterns['naming_convention'] = self._detect_naming_convention(project_root)
        
        # Detect framework usage
        self._project_patterns['frameworks'] = self._detect_frameworks(project_root)
        
        # Detect API patterns
        self._project_patterns['api_pattern'] = self._detect_api_patterns(project_root)
        
        # Detect test patterns
        self._project_patterns['test_pattern'] = self._detect_test_patterns(project_root)
        
        self._analyzed = True
    
    def _validate_code_patterns(self, tool_name: str, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate code against established patterns."""
        content = tool_input.get("content") or tool_input.get("new_string", "")
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        
        # Check import style consistency
        import_violation = self._check_import_consistency(content)
        if import_violation:
            return import_violation
        
        # Check async pattern consistency
        async_violation = self._check_async_consistency(content)
        if async_violation:
            return async_violation
        
        # Check naming convention consistency
        naming_violation = self._check_naming_consistency(content, file_path)
        if naming_violation:
            return naming_violation
        
        # Check framework mixing
        framework_violation = self._check_framework_consistency(content)
        if framework_violation:
            return framework_violation
        
        # Check API pattern consistency
        if self._is_api_file(file_path):
            api_violation = self._check_api_consistency(content)
            if api_violation:
                return api_violation
        
        return None
    
    def _check_import_consistency(self, content: str) -> Optional[ValidationResult]:
        """Check if imports match project style."""
        current_style = self._project_patterns.get('import_style', 'unknown')
        
        # Detect style in new content
        has_es6 = bool(re.search(r'import\s+.*\s+from\s+[\'"]', content))
        has_commonjs = bool(re.search(r'require\s*\([\'"]', content))
        
        if current_style == 'es6' and has_commonjs:
            return ValidationResult(
                severity=ValidationSeverity.BLOCK,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="🚨 ARCHITECTURE CONFLICT: Using CommonJS require() in ES6 module project",
                suggested_alternative="Use ES6 import syntax: import X from 'module'",
                blocking_reason="Mixing module systems causes compatibility issues",
                hive_guidance="Maintain consistent ES6 module syntax throughout the project",
                priority_score=85
            )
        elif current_style == 'commonjs' and has_es6:
            return ValidationResult(
                severity=ValidationSeverity.BLOCK,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="🚨 ARCHITECTURE CONFLICT: Using ES6 imports in CommonJS project",
                suggested_alternative="Use CommonJS syntax: const X = require('module')",
                blocking_reason="Mixing module systems causes compatibility issues",
                hive_guidance="Maintain consistent CommonJS syntax throughout the project",
                priority_score=85
            )
        
        return None
    
    def _check_async_consistency(self, content: str) -> Optional[ValidationResult]:
        """Check if async patterns match project style."""
        current_pattern = self._project_patterns.get('async_pattern', 'unknown')
        
        # Detect patterns in new content
        has_callbacks = bool(re.search(r'function\s*\([^)]*callback', content))
        bool(re.search(r'\.then\s*\(|\.catch\s*\(|new\s+Promise', content))
        bool(re.search(r'async\s+function|async\s*\(|await\s+', content))
        
        if current_pattern == 'async_await' and has_callbacks:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="⚠️ Using callbacks in async/await project",
                suggested_alternative="Use async/await pattern for consistency",
                hive_guidance="Modern async/await is cleaner and more maintainable",
                priority_score=60
            )
        elif current_pattern == 'promises' and has_callbacks:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="⚠️ Using callbacks in Promise-based project",
                suggested_alternative="Use Promises for consistency",
                hive_guidance="Promises provide better error handling than callbacks",
                priority_score=55
            )
        
        return None
    
    def _check_naming_consistency(self, content: str, file_path: str) -> Optional[ValidationResult]:
        """Check if naming conventions match project style."""
        current_convention = self._project_patterns.get('naming_convention', {})
        
        # Check file naming
        if file_path:
            path = Path(file_path)
            file_name = path.stem
            
            # Check case style
            if current_convention.get('files') == 'kebab-case' and '_' in file_name:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"⚠️ File '{path.name}' uses snake_case in kebab-case project",
                    suggested_alternative="Use kebab-case: " + file_name.replace('_', '-'),
                    hive_guidance="Consistent naming improves project maintainability",
                    priority_score=45
                )
        
        # Check function/variable naming in content
        if current_convention.get('functions') == 'camelCase':
            snake_functions = re.findall(r'function\s+([a-z]+_[a-z_]+)', content)
            if snake_functions:
                return ValidationResult(
                    severity=ValidationSeverity.WARN,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"⚠️ Function '{snake_functions[0]}' uses snake_case in camelCase project",
                    suggested_alternative="Use camelCase for function names",
                    hive_guidance="Consistent naming conventions improve code readability",
                    priority_score=40
                )
        
        return None
    
    def _check_framework_consistency(self, content: str) -> Optional[ValidationResult]:
        """Check for conflicting framework usage."""
        current_frameworks = self._project_patterns.get('frameworks', set())
        
        # Framework detection patterns
        framework_patterns = {
            'react': [r'import.*from\s+[\'"]react', r'React\.', r'useState', r'useEffect'],
            'vue': [r'import.*from\s+[\'"]vue', r'Vue\.', r'createApp', r'ref\(', r'reactive\('],
            'angular': [r'import.*from\s+[\'"]@angular', r'@Component', r'@Injectable'],
            'express': [r'express\(\)', r'app\.get\(', r'app\.post\(', r'router\.'],
            'fastify': [r'fastify\(\)', r'fastify\.get\(', r'fastify\.post\('],
            'jest': [r'describe\(', r'test\(', r'expect\(', r'jest\.'],
            'mocha': [r'describe\(', r'it\(', r'chai', r'mocha'],
        }
        
        # Detect frameworks in new content
        detected_frameworks = set()
        for framework, patterns in framework_patterns.items():
            if any(re.search(pattern, content) for pattern in patterns):
                detected_frameworks.add(framework)
        
        # Check for conflicts
        conflicting_pairs = [
            ('react', 'vue'), ('react', 'angular'), ('vue', 'angular'),
            ('express', 'fastify'), ('jest', 'mocha')
        ]
        
        for framework in detected_frameworks:
            for f1, f2 in conflicting_pairs:
                if framework == f1 and f2 in current_frameworks:
                    return ValidationResult(
                        severity=ValidationSeverity.BLOCK,
                        violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                        message=f"🚨 FRAMEWORK CONFLICT: Introducing {f1} in {f2} project",
                        suggested_alternative=f"Use existing {f2} framework",
                        blocking_reason="Mixing competing frameworks creates conflicts and bloat",
                        hive_guidance="One framework per concern maintains architectural clarity",
                        priority_score=90
                    )
                elif framework == f2 and f1 in current_frameworks:
                    return ValidationResult(
                        severity=ValidationSeverity.BLOCK,
                        violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                        message=f"🚨 FRAMEWORK CONFLICT: Introducing {f2} in {f1} project",
                        suggested_alternative=f"Use existing {f1} framework",
                        blocking_reason="Mixing competing frameworks creates conflicts and bloat",
                        hive_guidance="One framework per concern maintains architectural clarity",
                        priority_score=90
                    )
        
        return None
    
    def _check_api_consistency(self, content: str) -> Optional[ValidationResult]:
        """Check if API patterns match project style."""
        current_api_pattern = self._project_patterns.get('api_pattern', 'unknown')
        
        # Detect API patterns
        has_rest = bool(re.search(r'(GET|POST|PUT|DELETE|PATCH)\s*[\'"]?/api/', content, re.IGNORECASE))
        has_graphql = bool(re.search(r'(query|mutation|subscription)\s*{|graphql|gql`', content))
        bool(re.search(r'\.call\(|\.invoke\(|jsonrpc|rpc\.', content))
        
        if current_api_pattern == 'rest' and has_graphql:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="⚠️ Introducing GraphQL in REST API project",
                suggested_alternative="Use existing REST patterns",
                hive_guidance="Mixing API paradigms increases complexity",
                priority_score=65
            )
        elif current_api_pattern == 'graphql' and has_rest:
            return ValidationResult(
                severity=ValidationSeverity.WARN,
                violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                message="⚠️ Creating REST endpoints in GraphQL project",
                suggested_alternative="Use GraphQL resolvers",
                hive_guidance="Maintain consistent API architecture",
                priority_score=65
            )
        
        return None
    
    def _validate_task_architecture(self, tool_input: Dict[str, Any]) -> Optional[ValidationResult]:
        """Validate task descriptions for architectural violations."""
        description = tool_input.get("description", "").lower()
        
        # Keywords suggesting architectural changes
        violation_keywords = {
            'switch to': "Attempting to change established patterns",
            'replace with': "Trying to replace existing architecture",
            'migrate to': "Proposing architectural migration",
            'convert to': "Suggesting pattern conversion",
            'rewrite in': "Proposing language/framework change"
        }
        
        for keyword, message in violation_keywords.items():
            if keyword in description:
                return ValidationResult(
                    severity=ValidationSeverity.BLOCK,
                    violation_type=WorkflowViolationType.INEFFICIENT_EXECUTION,
                    message=f"🚨 ARCHITECTURE VIOLATION: {message}",
                    suggested_alternative="Work within existing architectural patterns",
                    blocking_reason="Architectural changes require careful planning and approval",
                    hive_guidance="Queen ZEN must approve all architectural decisions",
                    priority_score=95
                )
        
        return None
    
    # Helper methods for pattern detection
    def _detect_import_style(self, project_root: Path) -> str:
        """Detect predominant import style in project."""
        es6_count = 0
        commonjs_count = 0
        
        for js_file in project_root.rglob("*.js"):
            try:
                content = js_file.read_text()
                if re.search(r'import\s+.*\s+from\s+[\'"]', content):
                    es6_count += 1
                if re.search(r'require\s*\([\'"]', content):
                    commonjs_count += 1
            except Exception:
                continue
        
        if es6_count > commonjs_count:
            return 'es6'
        elif commonjs_count > 0:
            return 'commonjs'
        return 'unknown'
    
    def _detect_async_pattern(self, project_root: Path) -> str:
        """Detect predominant async pattern."""
        patterns_count = {'callbacks': 0, 'promises': 0, 'async_await': 0}
        
        for file in project_root.rglob("*.js"):
            try:
                content = file.read_text()
                if re.search(r'async\s+function|async\s*\(|await\s+', content):
                    patterns_count['async_await'] += 1
                elif re.search(r'\.then\s*\(|\.catch\s*\(|new\s+Promise', content):
                    patterns_count['promises'] += 1
                elif re.search(r'function\s*\([^)]*callback', content):
                    patterns_count['callbacks'] += 1
            except Exception:
                continue
        
        return max(patterns_count, key=patterns_count.get)
    
    def _detect_naming_convention(self, project_root: Path) -> Dict[str, str]:
        """Detect naming conventions."""
        conventions = {}
        
        # File naming
        file_names = [f.stem for f in project_root.rglob("*.js") if f.stem]
        if file_names:
            if all('-' in name and '_' not in name for name in file_names[:10]):
                conventions['files'] = 'kebab-case'
            elif all('_' in name and '-' not in name for name in file_names[:10]):
                conventions['files'] = 'snake_case'
            else:
                conventions['files'] = 'camelCase'
        
        # Function naming (simplified check)
        conventions['functions'] = 'camelCase'  # Default for JS
        
        return conventions
    
    def _detect_frameworks(self, project_root: Path) -> Set[str]:
        """Detect frameworks used in project."""
        frameworks = set()
        
        # Check package.json
        package_json = project_root / "package.json"
        if package_json.exists():
            try:
                data = json.loads(package_json.read_text())
                deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                
                framework_map = {
                    'react': 'react',
                    'vue': 'vue',
                    '@angular/core': 'angular',
                    'express': 'express',
                    'fastify': 'fastify',
                    'jest': 'jest',
                    'mocha': 'mocha'
                }
                
                for dep, framework in framework_map.items():
                    if dep in deps:
                        frameworks.add(framework)
            except Exception:
                pass
        
        return frameworks
    
    def _detect_api_patterns(self, project_root: Path) -> str:
        """Detect API patterns used."""
        # Check for GraphQL schema files
        if list(project_root.rglob("*.graphql")) or list(project_root.rglob("*schema.gql")):
            return 'graphql'
        
        # Check for REST patterns in route files
        for file in project_root.rglob("*route*.js"):
            try:
                content = file.read_text()
                if re.search(r'(app|router)\.(get|post|put|delete)\s*\(', content):
                    return 'rest'
            except Exception:
                continue
        
        return 'unknown'
    
    def _detect_test_patterns(self, project_root: Path) -> str:
        """Detect testing framework patterns."""
        # Check test files
        for test_file in project_root.rglob("*.test.js"):
            try:
                content = test_file.read_text()
                if 'jest' in content or 'expect(' in content:
                    return 'jest'
                elif 'mocha' in content or 'chai' in content:
                    return 'mocha'
            except Exception:
                continue
        
        return 'unknown'
    
    def _is_api_file(self, file_path: str) -> bool:
        """Check if file is likely an API file."""
        api_indicators = ['api', 'route', 'controller', 'endpoint', 'resolver', 'handler']
        return any(indicator in file_path.lower() for indicator in api_indicators)