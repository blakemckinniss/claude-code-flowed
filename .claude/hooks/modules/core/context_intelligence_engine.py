#!/usr/bin/env python3
"""Context Intelligence Engine for ZEN Co-pilot Phase 1.

Builds on existing ZenConsultant prototype to provide intelligent context analysis,
tech stack detection, smart prompt enhancement, and progressive verbosity.
Integrates with existing memory system and hook validation framework.
"""

import json
import subprocess
import sys
import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

# Import shared types first to avoid circular imports
from .shared_types import TechStack, GitContext, ProjectContext

# Import existing infrastructure
from .zen_consultant import ZenConsultant, ComplexityLevel, CoordinationType
from ..memory.zen_memory_integration import get_zen_memory_manager, ZenMemoryManager
from .bmad_integration import BMADIntegration, BMADRoleDetector, BMADStoryGenerator
from .hive_coordinator import HiveCoordinator, HiveStructure, create_hive_enhanced_directive


class UserExpertiseLevel(Enum):
    """User expertise levels for progressive verbosity."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"  
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class EnhancedPrompt:
    """Enhanced prompt with context suggestions."""
    original_prompt: str
    enhanced_prompt: str
    missing_context: List[str]
    suggestions: List[str] 
    confidence: float
    improvement_score: float


class GitContextAnalyzer:
    """Analyzes git repository context for intelligent decision making."""
    
    def __init__(self):
        """Initialize git context analyzer."""
        self.git_available = self._check_git_availability()
        
    def _check_git_availability(self) -> bool:
        """Check if git is available and this is a git repository."""
        try:
            result = subprocess.run(['git', 'status'], 
                                  check=False, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def analyze_repository_context(self, project_dir: str = ".") -> GitContext:
        """Analyze git repository context."""
        if not self.git_available:
            return GitContext(
                is_repo=False,
                current_branch="unknown",
                uncommitted_changes=0,
                recent_commits=[],
                branch_health=0.0,
                last_activity=datetime.now(),
                repository_age_days=0,
                commit_frequency=0.0
            )
        
        try:
            # Get current branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         check=False, capture_output=True, text=True, timeout=5)
            current_branch = branch_result.stdout.strip() or "main"
            
            # Count uncommitted changes
            status_result = subprocess.run(['git', 'status', '--porcelain'], 
                                         check=False, capture_output=True, text=True, timeout=5)
            uncommitted_changes = len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0
            
            # Get recent commits (last 10)
            log_result = subprocess.run([
                'git', 'log', '--oneline', '--format=%h|%s|%an|%ad', 
                '--date=iso', '-10'
            ], check=False, capture_output=True, text=True, timeout=10)
            
            recent_commits = []
            if log_result.stdout.strip():
                for line in log_result.stdout.strip().split('\n'):
                    parts = line.split('|', 3)
                    if len(parts) == 4:
                        recent_commits.append({
                            'hash': parts[0],
                            'message': parts[1],
                            'author': parts[2],
                            'date': parts[3]
                        })
            
            # Calculate branch health and activity metrics
            branch_health = self._calculate_branch_health(recent_commits, uncommitted_changes)
            last_activity = self._get_last_activity_date(recent_commits)
            repository_age_days = self._calculate_repository_age()
            commit_frequency = self._calculate_commit_frequency(recent_commits, repository_age_days)
            
            return GitContext(
                is_repo=True,
                current_branch=current_branch,
                uncommitted_changes=uncommitted_changes,
                recent_commits=recent_commits,
                branch_health=branch_health,
                last_activity=last_activity,
                repository_age_days=repository_age_days,
                commit_frequency=commit_frequency
            )
            
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            # Fallback context for git command failures
            return GitContext(
                is_repo=True,
                current_branch="unknown",
                uncommitted_changes=0,
                recent_commits=[],
                branch_health=0.5,
                last_activity=datetime.now(),
                repository_age_days=0,
                commit_frequency=0.0
            )
    
    def _calculate_branch_health(self, commits: List[Dict], uncommitted: int) -> float:
        """Calculate branch health score (0.0 to 1.0)."""
        score = 1.0
        
        # Penalty for uncommitted changes
        if uncommitted > 10:
            score -= 0.3
        elif uncommitted > 5:
            score -= 0.1
            
        # Penalty for infrequent commits
        if len(commits) < 3:
            score -= 0.2
            
        # Bonus for regular commit patterns
        if len(commits) >= 5:
            score += 0.1
            
        return max(0.0, min(1.0, score))
    
    def _get_last_activity_date(self, commits: List[Dict]) -> datetime:
        """Get last activity date from commits."""
        if not commits:
            return datetime.now()
        
        try:
            # Parse the ISO date from the first (most recent) commit
            date_str = commits[0]['date'].split(' ')[0]  # Get date part only
            return datetime.fromisoformat(date_str)
        except (ValueError, KeyError, IndexError):
            return datetime.now()
    
    def _calculate_repository_age(self) -> int:
        """Calculate repository age in days."""
        try:
            # Get first commit date
            result = subprocess.run([
                'git', 'log', '--reverse', '--format=%ad', '--date=iso', '-1'
            ], check=False, capture_output=True, text=True, timeout=5)
            
            if result.stdout.strip():
                first_commit_date = datetime.fromisoformat(result.stdout.strip().split(' ')[0])
                return (datetime.now() - first_commit_date).days
        except (subprocess.SubprocessError, ValueError, subprocess.TimeoutExpired):
            pass
        
        return 0
    
    def _calculate_commit_frequency(self, commits: List[Dict], repo_age: int) -> float:
        """Calculate average commits per day."""
        if repo_age <= 0 or not commits:
            return 0.0
        
        # Estimate based on recent commits and repo age
        recent_commits_count = len(commits)
        
        # Use a weighted average favoring recent activity
        if repo_age < 30:
            return recent_commits_count / max(1, repo_age)
        else:
            # For older repos, estimate based on recent activity
            return (recent_commits_count * 3) / 30  # Extrapolate from last 10 commits


class TechStackDetector:
    """Detects technology stacks using file analysis and MCP tool integration."""
    
    # Technology indicators
    TECH_INDICATORS = {
        TechStack.NODEJS: {
            'files': ['package.json', 'yarn.lock', 'npm-shrinkwrap.json'],
            'extensions': ['.js', '.mjs'],
            'directories': ['node_modules'],
            'content_patterns': [r'require\(', r'module\.exports', r'npm']
        },
        TechStack.TYPESCRIPT: {
            'files': ['tsconfig.json', 'package.json'],
            'extensions': ['.ts', '.tsx'],
            'directories': ['node_modules'],
            'content_patterns': [r'interface\s+\w+', r'type\s+\w+\s*=', r'import.*from']
        },
        TechStack.PYTHON: {
            'files': ['requirements.txt', 'setup.py', 'pyproject.toml', 'Pipfile'],
            'extensions': ['.py', '.pyw'],
            'directories': ['__pycache__', 'venv', '.venv'],
            'content_patterns': [r'import\s+\w+', r'from\s+\w+\s+import', r'def\s+\w+']
        },
        TechStack.RUST: {
            'files': ['Cargo.toml', 'Cargo.lock'],
            'extensions': ['.rs'],
            'directories': ['target'],
            'content_patterns': [r'fn\s+\w+', r'use\s+\w+', r'cargo']
        },
        TechStack.JAVA: {
            'files': ['pom.xml', 'build.gradle', 'gradlew'],
            'extensions': ['.java', '.jar'],
            'directories': ['target', 'build', '.gradle'],
            'content_patterns': [r'public\s+class', r'import\s+java', r'@\w+']
        },
        TechStack.GO: {
            'files': ['go.mod', 'go.sum'],
            'extensions': ['.go'],
            'directories': ['vendor'],
            'content_patterns': [r'func\s+\w+', r'package\s+\w+', r'import\s+']
        },
        TechStack.REACT: {
            'files': ['package.json'],
            'extensions': ['.jsx', '.tsx'],
            'directories': ['node_modules'],
            'content_patterns': [r'React\.', r'useState', r'useEffect', r'jsx']
        },
        TechStack.VUE: {
            'files': ['package.json', 'vue.config.js'],
            'extensions': ['.vue'],
            'directories': ['node_modules'],
            'content_patterns': [r'<template>', r'Vue\.', r'vue']
        },
        TechStack.ANGULAR: {
            'files': ['angular.json', 'package.json'],
            'extensions': ['.ts'],
            'directories': ['node_modules', 'dist'],
            'content_patterns': [r'@Component', r'@Injectable', r'angular']
        }
    }
    
    def __init__(self, project_dir: str = "."):
        """Initialize tech stack detector."""
        self.project_dir = Path(project_dir)
        
    def detect_technology_stacks(self) -> List[TechStack]:
        """Detect technology stacks in the project."""
        detected_stacks = []
        stack_scores = {}
        
        for tech_stack, indicators in self.TECH_INDICATORS.items():
            score = self._calculate_tech_score(tech_stack, indicators)
            stack_scores[tech_stack] = score
            
            # Threshold for detection (lowered for better sensitivity)
            if score > 0.25:
                detected_stacks.append(tech_stack)
        
        # If no stacks detected, return unknown
        if not detected_stacks:
            detected_stacks.append(TechStack.UNKNOWN)
            
        # Sort by confidence score
        detected_stacks.sort(key=lambda x: stack_scores.get(x, 0), reverse=True)
        
        return detected_stacks
    
    def _calculate_tech_score(self, tech_stack: TechStack, indicators: Dict) -> float:
        """Calculate confidence score for a technology stack."""
        score = 0.0
        
        # Check for indicator files (high weight for definitive indicators)
        file_score = 0.0
        for file_pattern in indicators.get('files', []):
            if self._file_exists(file_pattern):
                file_score += 0.5  # Increased weight for config files
        
        # Check for file extensions
        ext_score = 0.0
        total_files = 0
        matching_files = 0
        
        for ext in indicators.get('extensions', []):
            count = self._count_files_with_extension(ext)
            total_files += count
            if count > 0:
                matching_files += count
                ext_score += min(0.4, count * 0.1)  # Increased weight and multiplier
        
        # Check for directories
        dir_score = 0.0
        for directory in indicators.get('directories', []):
            if self._directory_exists(directory):
                dir_score += 0.2  # Increased weight
        
        # Check content patterns (sample a few files)
        content_score = 0.0
        if matching_files > 0:
            content_score = self._analyze_content_patterns(
                indicators.get('content_patterns', []),
                indicators.get('extensions', [])
            )
        
        # Combine scores with adjusted weights
        score = (file_score * 0.5 + ext_score * 0.3 + 
                dir_score * 0.15 + content_score * 0.05)
        
        return min(1.0, score)
    
    def _file_exists(self, filename: str) -> bool:
        """Check if file exists in project directory."""
        return (self.project_dir / filename).exists()
    
    def _directory_exists(self, dirname: str) -> bool:
        """Check if directory exists in project directory."""
        return (self.project_dir / dirname).is_dir()
    
    def _count_files_with_extension(self, extension: str) -> int:
        """Count files with specific extension."""
        try:
            return len(list(self.project_dir.rglob(f"*{extension}")))
        except (OSError, PermissionError):
            return 0
    
    def _analyze_content_patterns(self, patterns: List[str], extensions: List[str]) -> float:
        """Analyze content patterns in sample files."""
        if not patterns or not extensions:
            return 0.0
        
        try:
            # Sample up to 5 files for content analysis
            sample_files = []
            for ext in extensions[:2]:  # Limit to 2 extensions
                files = list(self.project_dir.rglob(f"*{ext}"))[:3]  # Max 3 files per extension
                sample_files.extend(files)
            
            if not sample_files:
                return 0.0
            
            pattern_matches = 0
            len(patterns)
            
            for file_path in sample_files[:5]:  # Max 5 files total
                try:
                    with open(file_path, encoding='utf-8', errors='ignore') as f:
                        content = f.read(10000)  # Read first 10KB only
                        
                    for pattern in patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            pattern_matches += 1
                            break  # Count each file once per pattern match
                            
                except (OSError, UnicodeDecodeError, PermissionError):
                    continue
            
            return min(0.3, pattern_matches / max(1, len(sample_files)))
            
        except (OSError, PermissionError):
            return 0.0


class SmartPromptEnhancer:
    """Enhances user prompts with context awareness and learning."""
    
    def __init__(self, memory_manager: ZenMemoryManager):
        """Initialize prompt enhancer with memory integration."""
        self.memory_manager = memory_manager
        self.enhancement_patterns = self._load_enhancement_patterns()
    
    def _load_enhancement_patterns(self) -> Dict[str, List[str]]:
        """Load prompt enhancement patterns."""
        return {
            'vague_indicators': [
                'help', 'fix', 'improve', 'update', 'change', 'make better',
                'work on', 'handle', 'deal with', 'something'
            ],
            'missing_context_indicators': [
                'the system', 'the app', 'the code', 'this project', 'it',
                'that thing', 'the problem', 'the issue'
            ],
            'enhancement_templates': {
                'development': "Implement {feature} in {tech_stack} with {requirements}",
                'debugging': "Debug {specific_issue} in {component} affecting {functionality}", 
                'testing': "Create {test_type} tests for {component} covering {scenarios}",
                'refactoring': "Refactor {component} to improve {quality_aspect} while maintaining {constraints}",
                'architecture': "Design {system_component} architecture for {requirements} with {constraints}"
            }
        }
    
    async def enhance_prompt(self, prompt: str, project_context: ProjectContext) -> EnhancedPrompt:
        """Enhance user prompt with context and suggestions."""
        # Analyze prompt for vagueness and missing context
        vagueness_score = self._calculate_vagueness_score(prompt)
        missing_context = self._identify_missing_context(prompt, project_context)
        
        # Get recommendations from memory
        recommendations = await self.memory_manager.get_recommendations_for_prompt(
            prompt, "medium"  # Default complexity for enhancement
        )
        
        # Generate enhanced prompt
        enhanced_prompt = await self._generate_enhanced_prompt(
            prompt, project_context, missing_context, recommendations
        )
        
        # Calculate improvement metrics
        improvement_score = self._calculate_improvement_score(
            prompt, enhanced_prompt, vagueness_score
        )
        
        confidence = max(0.1, min(0.95, 0.7 - vagueness_score + len(missing_context) * 0.1))
        
        return EnhancedPrompt(
            original_prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            missing_context=missing_context,
            suggestions=self._generate_suggestions(prompt, project_context),
            confidence=confidence,
            improvement_score=improvement_score
        )
    
    def _calculate_vagueness_score(self, prompt: str) -> float:
        """Calculate how vague the prompt is (0.0 = specific, 1.0 = very vague)."""
        prompt_lower = prompt.lower()
        
        # Count vague indicators
        vague_count = sum(1 for indicator in self.enhancement_patterns['vague_indicators']
                         if indicator in prompt_lower)
        
        # Count missing context indicators
        missing_context_count = sum(1 for indicator in self.enhancement_patterns['missing_context_indicators']
                                  if indicator in prompt_lower)
        
        # Calculate vagueness based on word count and indicators
        word_count = len(prompt.split())
        
        vagueness = 0.0
        
        # Base vagueness from word count
        if word_count < 3:
            vagueness += 0.7  # Very short prompts are very vague
        elif word_count < 5:
            vagueness += 0.5
        elif word_count < 10:
            vagueness += 0.2
        
        # Add vagueness from indicators (increased weights)
        vagueness += min(0.5, vague_count * 0.15)
        vagueness += min(0.4, missing_context_count * 0.2)
        
        return min(1.0, vagueness)
    
    def _identify_missing_context(self, prompt: str, project_context: ProjectContext) -> List[str]:
        """Identify missing context that should be clarified."""
        missing = []
        prompt_lower = prompt.lower()
        
        # Check for technology stack context
        if any(word in prompt_lower for word in ['code', 'implement', 'build', 'develop']):
            if not any(stack.value.lower() in prompt_lower for stack in project_context.tech_stacks):
                missing.append(f"Technology stack (detected: {', '.join([s.value for s in project_context.tech_stacks[:2]])})")
        
        # Check for specific component/file context
        if any(indicator in prompt_lower for indicator in self.enhancement_patterns['missing_context_indicators']):
            missing.append("Specific component or file names")
        
        # Check for requirements context
        if any(word in prompt_lower for word in ['fix', 'improve', 'optimize']):
            if not any(word in prompt_lower for word in ['performance', 'security', 'usability', 'maintainability']):
                missing.append("Specific quality requirements or success criteria")
        
        # Check for scope context
        if len(prompt.split()) < 10 and any(word in prompt_lower for word in ['update', 'change', 'modify']):
            missing.append("Scope and boundaries of changes")
        
        return missing
    
    async def _generate_enhanced_prompt(self, 
                                      original: str, 
                                      context: ProjectContext,
                                      missing_context: List[str],
                                      recommendations: Dict[str, Any]) -> str:
        """Generate enhanced version of the prompt."""
        enhanced = original
        
        # Add context if missing and available
        if context.tech_stacks and context.tech_stacks[0] != TechStack.UNKNOWN:
            primary_stack = context.tech_stacks[0].value
            if primary_stack.lower() not in original.lower():
                enhanced = f"{enhanced} (using {primary_stack})"
        
        # Add project context if relevant
        if context.project_size != "small" and "project" not in original.lower():
            enhanced = f"{enhanced} for this {context.project_size} project"
        
        # Add recent activity context if relevant
        if context.git_context.uncommitted_changes > 5:
            enhanced = f"{enhanced} (note: {context.git_context.uncommitted_changes} uncommitted changes)"
        
        return enhanced
    
    def _calculate_improvement_score(self, original: str, enhanced: str, vagueness: float) -> float:
        """Calculate improvement score from enhancement."""
        if original == enhanced:
            return 0.0
        
        # Base improvement from vagueness reduction
        improvement = vagueness * 0.6
        
        # Additional improvement from context addition
        word_increase = len(enhanced.split()) - len(original.split())
        improvement += min(0.3, word_increase * 0.05)
        
        return min(1.0, improvement)
    
    def _generate_suggestions(self, prompt: str, context: ProjectContext) -> List[str]:
        """Generate contextual suggestions for the prompt."""
        suggestions = []
        prompt_lower = prompt.lower()
        
        # Technology-specific suggestions
        if context.tech_stacks and context.tech_stacks[0] != TechStack.UNKNOWN:
            primary_stack = context.tech_stacks[0]
            
            if primary_stack == TechStack.NODEJS and 'test' in prompt_lower:
                suggestions.append("Consider specifying Jest, Mocha, or other Node.js testing framework")
            elif primary_stack == TechStack.PYTHON and 'test' in prompt_lower:
                suggestions.append("Consider specifying pytest, unittest, or testing approach")
            elif primary_stack == TechStack.REACT and 'component' in prompt_lower:
                suggestions.append("Specify functional or class component, and any required props")
        
        # Git context suggestions
        if context.git_context.uncommitted_changes > 0:
            suggestions.append("Consider committing current changes before major modifications")
        
        # Project size suggestions
        if context.project_size in ["large", "enterprise"]:
            suggestions.append("Consider breaking down into smaller, manageable tasks")
        
        return suggestions[:3]  # Limit to 3 suggestions


class ProgressiveVerbositySystem:
    """Manages adaptive verbosity based on user expertise and preferences."""
    
    def __init__(self, memory_manager: ZenMemoryManager):
        """Initialize verbosity system with memory integration."""
        self.memory_manager = memory_manager
        self.user_profiles = {}  # In production, this would be persistent
        self.verbosity_templates = self._load_verbosity_templates()
    
    def _load_verbosity_templates(self) -> Dict[str, Dict[str, str]]:
        """Load verbosity templates for different expertise levels."""
        return {
            UserExpertiseLevel.BEGINNER.value: {
                'directive_prefix': '🎯 BEGINNER-FRIENDLY GUIDE',
                'explanation_level': 'detailed',
                'technical_terms': 'explained',
                'step_detail': 'comprehensive',
                'examples': 'included'
            },
            UserExpertiseLevel.INTERMEDIATE.value: {
                'directive_prefix': '⚡ INTERMEDIATE TASK',
                'explanation_level': 'moderate',
                'technical_terms': 'standard',
                'step_detail': 'clear',
                'examples': 'relevant'
            },
            UserExpertiseLevel.ADVANCED.value: {
                'directive_prefix': '🚀 ADVANCED IMPLEMENTATION',
                'explanation_level': 'concise',
                'technical_terms': 'technical',
                'step_detail': 'efficient',
                'examples': 'minimal'
            },
            UserExpertiseLevel.EXPERT.value: {
                'directive_prefix': '💎 EXPERT EXECUTION',
                'explanation_level': 'minimal',
                'technical_terms': 'precise',
                'step_detail': 'essential',
                'examples': 'none'
            }
        }
    
    def detect_user_expertise(self, prompt: str, interaction_history: Optional[List[str]] = None) -> UserExpertiseLevel:
        """Detect user expertise level from prompt and history."""
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())
        
        # Technical indicators with more specific classification
        expert_indicators = [
            'refactor', 'optimize', 'architecture', 'design patterns', 'scalability',
            'microservices', 'kubernetes', 'docker', 'ci/cd', 'devops',
            'performance tuning', 'load balancing', 'caching strategies'
        ]
        
        advanced_indicators = [
            'implement', 'algorithm', 'data structure', 'api design', 'database',
            'testing strategy', 'deployment', 'monitoring', 'logging', 'microservices'
        ]
        
        intermediate_indicators = [
            'create', 'build', 'develop', 'add feature', 'integrate',
            'setup', 'configure', 'debug', 'troubleshoot'
        ]
        
        beginner_indicators = [
            'help', 'how to', 'tutorial', 'learn', 'guide', 'explain',
            'getting started', 'setup', 'install', 'basic'
        ]
        
        # Count indicators
        expert_score = sum(1 for indicator in expert_indicators if indicator in prompt_lower)
        advanced_score = sum(1 for indicator in advanced_indicators if indicator in prompt_lower)
        intermediate_score = sum(1 for indicator in intermediate_indicators if indicator in prompt_lower)
        beginner_score = sum(1 for indicator in beginner_indicators if indicator in prompt_lower)
        
        # Weight the scores
        expert_weighted = expert_score * 4
        advanced_weighted = advanced_score * 3
        intermediate_weighted = intermediate_score * 2
        beginner_score * 1
        
        # Determine expertise level with stricter criteria
        if expert_weighted >= 8 or (expert_score >= 2 and word_count > 15):
            return UserExpertiseLevel.EXPERT
        elif expert_weighted >= 4 or advanced_weighted >= 6 or (advanced_score >= 2 and word_count > 10):
            return UserExpertiseLevel.ADVANCED
        elif advanced_weighted >= 3 or intermediate_weighted >= 4 or (intermediate_score >= 2):
            return UserExpertiseLevel.INTERMEDIATE
        else:
            return UserExpertiseLevel.BEGINNER
    
    def adapt_directive_verbosity(self, 
                                directive: str, 
                                expertise: UserExpertiseLevel,
                                context: ProjectContext) -> str:
        """Adapt directive verbosity based on user expertise."""
        template = self.verbosity_templates[expertise.value]
        
        # Modify directive based on expertise level
        if expertise == UserExpertiseLevel.BEGINNER:
            return self._create_beginner_directive(directive, context, template)
        elif expertise == UserExpertiseLevel.INTERMEDIATE:
            return self._create_intermediate_directive(directive, template)
        elif expertise == UserExpertiseLevel.ADVANCED:
            return self._create_advanced_directive(directive, template)
        else:  # EXPERT
            return self._create_expert_directive(directive, template)
    
    def _create_beginner_directive(self, directive: str, context: ProjectContext, template: Dict) -> str:
        """Create beginner-friendly directive with explanations."""
        enhanced = f"{template['directive_prefix']}\n\n"
        enhanced += directive + "\n\n"
        
        # Add helpful context
        enhanced += "📚 HELPFUL CONTEXT:\n"
        if context.tech_stacks and context.tech_stacks[0] != TechStack.UNKNOWN:
            enhanced += f"• Your project uses {context.tech_stacks[0].value}\n"
        enhanced += f"• Project complexity: {context.project_size}\n"
        enhanced += f"• Git status: {context.git_context.uncommitted_changes} uncommitted changes\n\n"
        
        # Add learning resources suggestion
        enhanced += "💡 TIP: Ask for explanations of any technical terms you don't understand!\n"
        
        return enhanced
    
    def _create_intermediate_directive(self, directive: str, template: Dict) -> str:
        """Create intermediate-level directive with balanced detail."""
        enhanced = f"{template['directive_prefix']}\n\n"
        enhanced += directive + "\n\n"
        enhanced += "💼 Next: Review the plan and ask for clarification on any unclear steps.\n"
        return enhanced
    
    def _create_advanced_directive(self, directive: str, template: Dict) -> str:
        """Create advanced directive with technical focus."""
        enhanced = f"{template['directive_prefix']}\n\n"
        enhanced += directive + "\n\n"
        enhanced += "⚡ Ready for immediate execution with technical precision.\n"
        return enhanced
    
    def _create_expert_directive(self, directive: str, template: Dict) -> str:
        """Create expert-level directive with minimal verbosity."""
        # Strip emojis and verbose language for experts
        cleaned_directive = re.sub(r'[🚨⚡🐝👑🤖]', '', directive)
        cleaned_directive = re.sub(r'CRITICAL|URGENT|IMPORTANT:', '', cleaned_directive)
        cleaned_directive = re.sub(r'WARNINGS?:', 'Notes:', cleaned_directive)
        
        enhanced = f"{template['directive_prefix']}: {cleaned_directive.strip()}"
        return enhanced


class ContextIntelligenceEngine:
    """Main Context Intelligence Engine orchestrating all components."""
    
    def __init__(self, project_dir: str = "."):
        """Initialize the Context Intelligence Engine."""
        self.project_dir = project_dir
        self.zen_consultant = ZenConsultant()
        self.memory_manager = get_zen_memory_manager()
        
        # Initialize analyzers
        self.git_analyzer = GitContextAnalyzer()
        self.tech_detector = TechStackDetector(project_dir)
        self.prompt_enhancer = SmartPromptEnhancer(self.memory_manager)
        self.verbosity_system = ProgressiveVerbositySystem(self.memory_manager)
        
        # Initialize BMAD components
        self.bmad_role_detector = BMADRoleDetector()
        self.bmad_story_generator = BMADStoryGenerator()
        self.bmad_integration = BMADIntegration()
        
        # Initialize Hive Coordination
        self.hive_coordinator = HiveCoordinator(self.zen_consultant)
        
        # Cache for performance
        self._context_cache = {}
        self._cache_timestamp = None
        self._cache_ttl = timedelta(minutes=5)
    
    async def analyze_full_context(self, force_refresh: bool = False) -> ProjectContext:
        """Analyze complete project context with caching."""
        # Check cache
        if (not force_refresh and 
            self._context_cache and 
            self._cache_timestamp and 
            datetime.now() - self._cache_timestamp < self._cache_ttl):
            return self._context_cache
        
        # Perform fresh analysis
        git_context = self.git_analyzer.analyze_repository_context(self.project_dir)
        tech_stacks = self.tech_detector.detect_technology_stacks()
        
        # Calculate additional metrics
        complexity_indicators = self._analyze_complexity_indicators()
        file_structure = self._analyze_file_structure()
        project_size = self._determine_project_size(file_structure, git_context)
        
        context = ProjectContext(
            git_context=git_context,
            tech_stacks=tech_stacks,
            complexity_indicators=complexity_indicators,
            file_structure=file_structure,
            project_size=project_size,
            dependencies_count=self._count_dependencies(),
            test_coverage_estimate=self._estimate_test_coverage(),
            documentation_quality=self._assess_documentation_quality()
        )
        
        # Update cache
        self._context_cache = context
        self._cache_timestamp = datetime.now()
        
        return context
    
    async def generate_intelligent_directive(self, prompt: str) -> Dict[str, Any]:
        """Generate context-aware directive using all intelligence components."""
        # Analyze project context
        project_context = await self.analyze_full_context()
        
        # Enhance prompt with context
        enhanced_prompt_data = await self.prompt_enhancer.enhance_prompt(prompt, project_context)
        
        # Detect user expertise
        user_expertise = self.verbosity_system.detect_user_expertise(prompt)
        
        # BMAD Integration: Detect roles and generate context-engineered story
        detected_roles = self.bmad_role_detector.detect_required_roles(prompt)
        primary_role = detected_roles[0] if detected_roles else None
        bmad_story = None
        
        if detected_roles:
            # Generate BMAD context-engineered story
            bmad_story = self.bmad_story_generator.generate_story(
                prompt=prompt,
                roles=detected_roles,
                project_context={
                    'tech_stacks': [stack.value for stack in project_context.tech_stacks],
                    'branch': project_context.git_context.current_branch,
                    'uncommitted_changes': project_context.git_context.uncommitted_changes,
                    'project_size': project_context.project_size
                }
            )
        
        # Detect if hive coordination is needed
        complexity, metadata = self.zen_consultant.analyze_prompt_complexity(enhanced_prompt_data.enhanced_prompt)
        coordination_type = self.zen_consultant.determine_coordination_type(complexity, metadata["categories"], prompt)
        needs_hive = complexity == ComplexityLevel.HIVE_REQUIRED or coordination_type == CoordinationType.HIVE
        
        # Generate session ID for hive tracking
        session_id = f"ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(prompt) % 1000:03d}"
        
        # Create hive structure if needed
        hive_structure = None
        if needs_hive:
            hive_structure = self.hive_coordinator.create_hive_structure(
                enhanced_prompt_data.enhanced_prompt, 
                project_context, 
                session_id
            )
        
        # Generate base directive using enhanced prompt
        base_directive = self.zen_consultant.generate_directive(enhanced_prompt_data.enhanced_prompt)
        
        # Enhance directive with BMAD concepts if roles were detected
        if detected_roles:
            base_directive = self.bmad_integration.enhance_directive_with_bmad(
                directive=base_directive,
                prompt=prompt,
                project_context={
                    'tech_stacks': [stack.value for stack in project_context.tech_stacks],
                    'branch': project_context.git_context.current_branch,
                    'uncommitted_changes': project_context.git_context.uncommitted_changes,
                    'project_size': project_context.project_size
                }
            )
        
        # Adapt verbosity for user expertise
        adapted_directive = self.verbosity_system.adapt_directive_verbosity(
            base_directive, user_expertise, project_context
        )
        
        # Create comprehensive response
        response = {
            "directive": adapted_directive,
            "context_analysis": {
                "git_status": {
                    "branch": project_context.git_context.current_branch,
                    "uncommitted_changes": project_context.git_context.uncommitted_changes,
                    "branch_health": project_context.git_context.branch_health,
                    "last_activity": project_context.git_context.last_activity.isoformat()
                },
                "technology_stacks": [stack.value for stack in project_context.tech_stacks],
                "project_size": project_context.project_size,
                "complexity_score": sum(project_context.complexity_indicators.values()) / len(project_context.complexity_indicators) if project_context.complexity_indicators else 0.0
            },
            "hive_coordination": {
                "enabled": needs_hive,
                "session_id": session_id if needs_hive else None,
                "complexity_level": complexity.value,
                "coordination_type": coordination_type.value,
                "structure": self.hive_coordinator.get_hive_status(session_id) if needs_hive else None,
                "scaling_triggers": {
                    "auto_scaling": hive_structure.auto_scaling_enabled if hive_structure else False,
                    "max_workers": hive_structure.max_workers if hive_structure else 0,
                    "current_capacity": hive_structure.current_capacity if hive_structure else 0.0
                }
            },
            "prompt_enhancement": {
                "original_prompt": enhanced_prompt_data.original_prompt,
                "enhanced_prompt": enhanced_prompt_data.enhanced_prompt,
                "missing_context": enhanced_prompt_data.missing_context,
                "suggestions": enhanced_prompt_data.suggestions,
                "improvement_score": enhanced_prompt_data.improvement_score
            },
            "user_adaptation": {
                "detected_expertise": user_expertise.value,
                "verbosity_level": user_expertise.value
            },
            "bmad_analysis": {
                "detected_roles": [role.value for role in detected_roles] if detected_roles else [],
                "primary_role": primary_role.value if primary_role else None,
                "story_generated": bmad_story is not None,
                "story_content": bmad_story if bmad_story else None
            },
            "confidence_metrics": {
                "context_confidence": min(project_context.git_context.branch_health, 0.9),
                "tech_detection_confidence": 0.8 if project_context.tech_stacks[0] != TechStack.UNKNOWN else 0.3,
                "prompt_enhancement_confidence": enhanced_prompt_data.confidence,
                "bmad_confidence": 0.85 if primary_role else 0.0,
                "hive_coordination_confidence": 0.9 if needs_hive else 0.7,
                "overall_confidence": (project_context.git_context.branch_health + enhanced_prompt_data.confidence + (0.85 if primary_role else 0.0) + (0.9 if needs_hive else 0.7)) / 4
            }
        }
        
        return response
    
    def _analyze_complexity_indicators(self) -> Dict[str, Any]:
        """Analyze project complexity indicators."""
        try:
            project_path = Path(self.project_dir)
            
            # Count different file types
            code_files = len(list(project_path.rglob("*.py"))) + len(list(project_path.rglob("*.js"))) + len(list(project_path.rglob("*.ts")))
            config_files = len(list(project_path.rglob("*.json"))) + len(list(project_path.rglob("*.yaml"))) + len(list(project_path.rglob("*.yml")))
            
            return {
                "code_files_count": code_files,
                "config_files_count": config_files,
                "directory_depth": self._calculate_max_directory_depth(),
                "has_tests": len(list(project_path.rglob("*test*"))) > 0,
                "has_docs": len(list(project_path.rglob("*.md"))) > 0
            }
        except (OSError, PermissionError):
            return {"error": "Cannot analyze complexity"}
    
    def _analyze_file_structure(self) -> Dict[str, int]:
        """Analyze file structure and types."""
        try:
            project_path = Path(self.project_dir)
            file_types = {}
            
            for file_path in project_path.rglob("*"):
                if file_path.is_file():
                    suffix = file_path.suffix.lower()
                    file_types[suffix] = file_types.get(suffix, 0) + 1
            
            return file_types
        except (OSError, PermissionError):
            return {}
    
    def _determine_project_size(self, file_structure: Dict[str, int], git_context: GitContext) -> str:
        """Determine project size based on various metrics."""
        total_files = sum(file_structure.values())
        
        if total_files < 10:
            return "small"
        elif total_files < 100:
            return "medium"
        elif total_files < 1000:
            return "large"
        else:
            return "enterprise"
    
    def _calculate_max_directory_depth(self) -> int:
        """Calculate maximum directory depth."""
        try:
            project_path = Path(self.project_dir)
            max_depth = 0
            
            for path in project_path.rglob("*"):
                if path.is_dir():
                    depth = len(path.relative_to(project_path).parts)
                    max_depth = max(max_depth, depth)
            
            return max_depth
        except (OSError, PermissionError):
            return 0
    
    def _count_dependencies(self) -> int:
        """Count project dependencies."""
        dep_count = 0
        
        # Check package.json
        package_json = Path(self.project_dir) / "package.json"
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    deps = data.get('dependencies', {})
                    dev_deps = data.get('devDependencies', {})
                    dep_count += len(deps) + len(dev_deps)
            except (json.JSONDecodeError, OSError):
                pass
        
        # Check requirements.txt
        requirements_txt = Path(self.project_dir) / "requirements.txt"
        if requirements_txt.exists():
            try:
                with open(requirements_txt) as f:
                    lines = f.readlines()
                    dep_count += len([line for line in lines if line.strip() and not line.startswith('#')])
            except OSError:
                pass
        
        return dep_count
    
    def _estimate_test_coverage(self) -> float:
        """Estimate test coverage based on file analysis."""
        try:
            project_path = Path(self.project_dir)
            test_files = len(list(project_path.rglob("*test*"))) + len(list(project_path.rglob("*spec*")))
            code_files = len(list(project_path.rglob("*.py"))) + len(list(project_path.rglob("*.js"))) + len(list(project_path.rglob("*.ts")))
            
            if code_files == 0:
                return 0.0
            
            # Rough estimate: 1 test file per 3-5 code files is decent coverage
            estimated_coverage = min(1.0, (test_files * 4) / code_files)
            return estimated_coverage
        except (OSError, PermissionError):
            return 0.0
    
    def _assess_documentation_quality(self) -> float:
        """Assess documentation quality."""
        try:
            project_path = Path(self.project_dir)
            doc_files = list(project_path.rglob("*.md"))
            
            quality_score = 0.0
            
            # Check for README
            if (project_path / "README.md").exists():
                quality_score += 0.4
            
            # Check for other documentation
            if len(doc_files) > 1:
                quality_score += 0.3
            
            # Check for inline documentation (sample a few files)
            code_files = list(project_path.rglob("*.py"))[:5] + list(project_path.rglob("*.js"))[:5]
            documented_files = 0
            
            for code_file in code_files:
                try:
                    with open(code_file, encoding='utf-8', errors='ignore') as f:
                        content = f.read(5000)  # Read first 5KB
                        if ('"""' in content or "'''" in content or 
                            '/*' in content or '//' in content):
                            documented_files += 1
                except (OSError, UnicodeDecodeError):
                    continue
            
            if code_files:
                quality_score += 0.3 * (documented_files / len(code_files))
            
            return min(1.0, quality_score)
        except (OSError, PermissionError):
            return 0.0


# Integration function for existing hook system
async def create_context_aware_directive(prompt: str, project_dir: str = ".") -> Dict[str, Any]:
    """Create context-aware directive for integration with existing hook system."""
    engine = ContextIntelligenceEngine(project_dir)
    
    try:
        result = await engine.generate_intelligent_directive(prompt)
        
        # Format for existing hook system compatibility
        return {
            "hookSpecificOutput": {
                "hookEventName": "ContextIntelligentDirective",
                "additionalContext": result["directive"],
                "contextAnalysis": result["context_analysis"],
                "promptEnhancement": result["prompt_enhancement"], 
                "userAdaptation": result["user_adaptation"],
                "bmadAnalysis": result["bmad_analysis"],
                "confidenceMetrics": result["confidence_metrics"]
            }
        }
    except Exception as e:
        # Fallback to existing ZEN consultant
        zen_consultant = ZenConsultant()
        fallback_directive = zen_consultant.generate_directive(prompt)
        
        return {
            "hookSpecificOutput": {
                "hookEventName": "ContextIntelligentDirective",
                "additionalContext": fallback_directive,
                "fallback": True,
                "error": str(e)
            }
        }


if __name__ == "__main__":
    # Test the Context Intelligence Engine
    async def test_engine():
        print("🧠 Context Intelligence Engine Test")
        print("=" * 50)
        
        engine = ContextIntelligenceEngine()
        
        # Test context analysis
        context = await engine.analyze_full_context()
        print("📊 Project Analysis:")
        print(f"  • Tech Stacks: {[s.value for s in context.tech_stacks]}")
        print(f"  • Project Size: {context.project_size}")
        print(f"  • Git Branch: {context.git_context.current_branch}")
        print(f"  • Uncommitted Changes: {context.git_context.uncommitted_changes}")
        print(f"  • Dependencies: {context.dependencies_count}")
        
        # Test intelligent directive generation
        test_prompt = "Fix the authentication bug in the login system"
        result = await engine.generate_intelligent_directive(test_prompt)
        
        print("\n🎯 Intelligent Directive:")
        print(result["directive"])
        
        print("\n📈 Confidence Metrics:")
        for metric, value in result["confidence_metrics"].items():
            print(f"  • {metric}: {value:.2f}")
    
    asyncio.run(test_engine())