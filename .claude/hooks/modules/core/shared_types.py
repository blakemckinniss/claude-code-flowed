#!/usr/bin/env python3
"""Shared type definitions to avoid circular imports."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any
from enum import Enum


class TechStack(Enum):
    """Supported technology stacks."""
    NODEJS = "Node.js"
    PYTHON = "Python" 
    RUST = "Rust"
    JAVA = "Java"
    DOTNET = ".NET"
    GO = "Go"
    REACT = "React"
    VUE = "Vue.js"
    ANGULAR = "Angular"
    TYPESCRIPT = "TypeScript"
    UNKNOWN = "Unknown"


@dataclass
class GitContext:
    """Git repository context information."""
    is_repo: bool
    current_branch: str
    uncommitted_changes: int
    recent_commits: List[Dict[str, str]]
    branch_health: float  # 0.0 to 1.0
    last_activity: datetime
    repository_age_days: int
    commit_frequency: float  # commits per day


@dataclass
class ProjectContext:
    """Complete project context analysis."""
    git_context: GitContext
    tech_stacks: List[TechStack]
    complexity_indicators: Dict[str, Any]
    file_structure: Dict[str, int]  # file type -> count
    project_size: str  # small, medium, large, enterprise
    dependencies_count: int
    test_coverage_estimate: float
    documentation_quality: float