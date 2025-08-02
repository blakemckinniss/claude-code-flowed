"""Execution Safety Analyzer.

Specialized analyzer for execution tools (Bash, subprocess commands)
with focus on security, safety, and performance implications.
"""

import re
import shlex
from typing import Dict, Any, List, Optional, Set

from ...core.tool_analyzer_base import (
    BaseToolAnalyzer, ToolContext, FeedbackResult, FeedbackSeverity, ToolCategory
)


class ExecutionSafetyAnalyzer(BaseToolAnalyzer):
    """Analyzer for execution safety and security patterns."""
    
    def __init__(self, priority: int = 950):
        """Initialize execution safety analyzer."""
        super().__init__(priority)
        
        # Dangerous command patterns
        self.dangerous_patterns = [
            {
                "pattern": r'\brm\s+-rf\s+/',
                "severity": FeedbackSeverity.CRITICAL,
                "message": "Dangerous recursive delete command detected",
                "category": "destructive"
            },
            {
                "pattern": r'\bsudo\s+rm\s',
                "severity": FeedbackSeverity.ERROR,
                "message": "Sudo with rm command - high risk operation",
                "category": "privileged_destructive"
            },
            {
                "pattern": r'\bchmod\s+777\b',
                "severity": FeedbackSeverity.WARNING,
                "message": "Overly permissive file permissions (777)",
                "category": "security"
            },
            {
                "pattern": r'\bcurl\s+.*\|\s*sh\b',
                "severity": FeedbackSeverity.ERROR,
                "message": "Piping curl output to shell - security risk",
                "category": "remote_execution"
            },
            {
                "pattern": r'\bwget\s+.*\|\s*sh\b',
                "severity": FeedbackSeverity.ERROR,
                "message": "Piping wget output to shell - security risk",
                "category": "remote_execution"
            }
        ]
        
        # Privileged operations
        self.privileged_commands = [
            "sudo", "su", "passwd", "usermod", "groupmod", 
            "chown", "chmod", "mount", "umount"
        ]
        
        # Network operations
        self.network_commands = [
            "curl", "wget", "nc", "netcat", "ssh", "scp", "rsync"
        ]
        
        # System modification commands
        self.system_commands = [
            "systemctl", "service", "crontab", "iptables", 
            "ufw", "firewall-cmd"
        ]
        
        # Package managers
        self.package_managers = [
            "apt", "apt-get", "yum", "dnf", "pacman", 
            "npm", "pip", "cargo", "composer"
        ]
    
    def get_analyzer_name(self) -> str:
        return "execution_safety_analyzer"
    
    def get_supported_tools(self) -> List[str]:
        return ["Bash", "subprocess", "mcp__terminal__*"]
    
    def get_tool_categories(self) -> List[ToolCategory]:
        return [ToolCategory.EXECUTION]
    
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze execution commands for safety and security."""
        tool_input = context.tool_input
        command = tool_input.get("command", "")
        
        if not command:
            return None
        
        # Check for dangerous patterns
        danger_result = self._check_dangerous_patterns(command)
        if danger_result:
            return danger_result
        
        # Check for privileged operations
        privilege_result = self._check_privileged_operations(command, context)
        if privilege_result:
            return privilege_result
        
        # Check for network operations
        network_result = self._check_network_operations(command, context)
        if network_result:
            return network_result
        
        # Check for command injection risks
        injection_result = self._check_command_injection(command)
        if injection_result:
            return injection_result
        
        # Check for performance implications
        performance_result = self._check_performance_implications(command, context)
        if performance_result:
            return performance_result
        
        return None
    
    def _check_dangerous_patterns(self, command: str) -> Optional[FeedbackResult]:
        """Check for dangerous command patterns."""
        for pattern_info in self.dangerous_patterns:
            if re.search(pattern_info["pattern"], command, re.IGNORECASE):
                suggestions = self._get_safety_suggestions(pattern_info["category"])
                
                return FeedbackResult(
                    severity=pattern_info["severity"],
                    message=f"{pattern_info['message']}: {command[:100]}...",
                    suggestions=suggestions,
                    metadata={
                        "command": command,
                        "risk_category": pattern_info["category"],
                        "pattern": pattern_info["pattern"]
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _check_privileged_operations(self, command: str, context: ToolContext) -> Optional[FeedbackResult]:
        """Check for privileged operations that need special attention."""
        command_parts = shlex.split(command) if command else []
        if not command_parts:
            return None
        
        base_command = command_parts[0]
        
        if base_command in self.privileged_commands:
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message=f"Privileged operation detected: {base_command}",
                suggestions=[
                    "Ensure you have necessary permissions",
                    "Consider the security implications",
                    "Verify the command is necessary for the task",
                    "Test in safe environment first"
                ],
                metadata={
                    "command": command,
                    "privileged_command": base_command
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _check_network_operations(self, command: str, context: ToolContext) -> Optional[FeedbackResult]:
        """Check for network operations and their security implications."""
        command_parts = shlex.split(command) if command else []
        if not command_parts:
            return None
        
        base_command = command_parts[0]
        
        if base_command in self.network_commands:
            # Check for insecure patterns
            if "http://" in command and "localhost" not in command:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message="Insecure HTTP connection detected in network operation",
                    suggestions=[
                        "Use HTTPS instead of HTTP when possible",
                        "Verify the source is trustworthy", 
                        "Consider security implications of unencrypted data"
                    ],
                    metadata={
                        "command": command,
                        "security_issue": "insecure_http"
                    },
                    analyzer_name=self.get_analyzer_name()
                )
            
            # General network operation guidance
            return FeedbackResult(
                severity=FeedbackSeverity.INFO,
                message=f"Network operation detected: {base_command}",
                suggestions=[
                    "Ensure network connectivity is available",
                    "Consider timeout settings for network operations",
                    "Handle network errors gracefully"
                ],
                metadata={
                    "command": command,
                    "network_command": base_command
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _check_command_injection(self, command: str) -> Optional[FeedbackResult]:
        """Check for potential command injection vulnerabilities."""
        # Look for suspicious patterns that might indicate injection
        injection_patterns = [
            r'[;&|`$()]',  # Command separators and substitution
            r'\$\{.*\}',   # Parameter expansion
            r'`.*`',       # Command substitution
            r'\$\(.*\)',   # Command substitution
        ]
        
        risk_indicators = []
        for pattern in injection_patterns:
            if re.search(pattern, command):
                risk_indicators.append(pattern)
        
        if len(risk_indicators) >= 2:  # Multiple indicators suggest higher risk
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message="Command contains patterns that may indicate injection risk",
                suggestions=[
                    "Validate all user input before using in commands",
                    "Use parameterized commands when possible",
                    "Sanitize special characters in command arguments",
                    "Consider using subprocess with argument lists instead of shell=True"
                ],
                metadata={
                    "command": command,
                    "risk_patterns": risk_indicators
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _check_performance_implications(self, command: str, context: ToolContext) -> Optional[FeedbackResult]:
        """Check for performance implications of commands."""
        # Long-running operations
        long_running_patterns = [
            r'\bfind\s+/\s',           # Find from root
            r'\bgrep\s+-r\s+.*\s+/\s', # Recursive grep from root
            r'\bcp\s+-r\s+.*\s+.*',    # Recursive copy
            r'\btar\s+.*\s+.*\.tar',   # Tar operations
            r'\bzip\s+.*',             # Compression
            r'\bgit\s+clone\s+.*',     # Git clone
        ]
        
        for pattern in long_running_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return FeedbackResult(
                    severity=FeedbackSeverity.INFO,
                    message="Potentially long-running operation detected",
                    suggestions=[
                        "Consider adding timeout parameters",
                        "Monitor system resources during execution",
                        "Consider running in background for very long operations",
                        "Provide progress indication if possible"
                    ],
                    metadata={
                        "command": command,
                        "performance_concern": "long_running"
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        # Resource intensive operations
        resource_patterns = [
            r'\bmake\s+-j\s*$',  # Make without job limit
            r'\bnpm\s+install\s+--global',  # Global npm install
        ]
        
        for pattern in resource_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message="Resource-intensive operation detected",
                    suggestions=[
                        "Monitor system resources (CPU, memory, disk)",
                        "Consider limiting concurrent operations",
                        "Ensure sufficient system resources are available"
                    ],
                    metadata={
                        "command": command,
                        "performance_concern": "resource_intensive"
                    },
                    analyzer_name=self.get_analyzer_name()
                )
        
        return None
    
    def _get_safety_suggestions(self, category: str) -> List[str]:
        """Get safety suggestions based on risk category."""
        suggestions_map = {
            "destructive": [
                "⚠️  CRITICAL: This is a destructive operation",
                "Create backups before proceeding",
                "Double-check the target paths",
                "Consider using --dry-run or --preview flags first"
            ],
            "privileged_destructive": [
                "🚨 EXTREME CAUTION: Privileged destructive operation",
                "Verify you have the correct permissions",
                "Ensure this is absolutely necessary",
                "Consider non-destructive alternatives"
            ], 
            "security": [
                "🔒 Security consideration required",
                "Review the security implications",
                "Use more restrictive permissions when possible",
                "Document the security decision"
            ],
            "remote_execution": [
                "🌐 Remote execution security risk",
                "Verify the source is trusted and secure",
                "Consider downloading and inspecting first",
                "Use package managers when available"
            ]
        }
        
        return suggestions_map.get(category, [
            "Review the command for safety",
            "Consider the implications before executing",
            "Test in a safe environment first"
        ])


class PackageManagerAnalyzer(BaseToolAnalyzer):
    """Specialized analyzer for package manager operations."""
    
    def __init__(self, priority: int = 750):
        super().__init__(priority)
        
        self.package_managers = {
            "npm": {
                "install_commands": ["install", "i"],
                "global_flag": "-g",
                "security_commands": ["audit", "audit fix"]
            },
            "pip": {
                "install_commands": ["install"],
                "global_flag": "--user",
                "security_commands": ["check"]
            },
            "cargo": {
                "install_commands": ["install"],
                "global_flag": None,
                "security_commands": ["audit"]
            }
        }
    
    def get_analyzer_name(self) -> str:
        return "package_manager_analyzer"
    
    def get_supported_tools(self) -> List[str]:
        return ["Bash"]
    
    def get_tool_categories(self) -> List[ToolCategory]:
        return [ToolCategory.PACKAGE_MANAGEMENT]
    
    def should_analyze(self, context: ToolContext) -> bool:
        """Only analyze package manager commands."""
        command = context.tool_input.get("command", "")
        return any(pm in command for pm in self.package_managers.keys())
    
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze package manager operations."""
        command = context.tool_input.get("command", "")
        
        # Detect package manager
        detected_pm = None
        for pm in self.package_managers.keys():
            if command.startswith(pm + " "):
                detected_pm = pm
                break
        
        if not detected_pm:
            return None
        
        # Check for security considerations
        security_result = self._check_package_security(command, detected_pm)
        if security_result:
            return security_result
        
        # Check for dependency management best practices
        dependency_result = self._check_dependency_management(command, detected_pm, context)
        if dependency_result:
            return dependency_result
        
        return None
    
    def _check_package_security(self, command: str, package_manager: str) -> Optional[FeedbackResult]:
        """Check package installation for security considerations."""
        pm_config = self.package_managers[package_manager]
        
        # Check for global installations
        global_flag = pm_config.get("global_flag")
        if global_flag and global_flag in command:
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message=f"Global {package_manager} package installation detected",
                suggestions=[
                    "Consider local installation instead of global",
                    "Use virtual environments or project-specific dependencies",
                    "Verify the package is from a trusted source"
                ],
                metadata={
                    "package_manager": package_manager,
                    "installation_type": "global"
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        # Check for untrusted sources (simplified check)
        if "--registry" in command or "http://" in command:
            return FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message="Package installation from custom source detected",
                suggestions=[
                    "Verify the package source is trusted",
                    "Check package integrity and signatures",
                    "Consider using official package registries"
                ],
                metadata={
                    "package_manager": package_manager,
                    "security_concern": "custom_source"
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None
    
    def _check_dependency_management(
        self, 
        command: str, 
        package_manager: str, 
        context: ToolContext
    ) -> Optional[FeedbackResult]:
        """Check for dependency management best practices."""
        pm_config = self.package_managers[package_manager]
        
        # Check if installing without lockfile update consideration
        install_commands = pm_config["install_commands"]
        if any(cmd in command for cmd in install_commands):
            return FeedbackResult(
                severity=FeedbackSeverity.INFO,
                message=f"{package_manager} package installation detected",
                suggestions=[
                    "Ensure package-lock.json/requirements.txt is updated",
                    "Consider running security audit after installation",
                    "Document new dependencies in project documentation"
                ],
                metadata={
                    "package_manager": package_manager,
                    "operation": "install"
                },
                analyzer_name=self.get_analyzer_name()
            )
        
        return None