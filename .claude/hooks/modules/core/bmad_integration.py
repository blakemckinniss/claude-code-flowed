#!/usr/bin/env python3
"""BMAD-METHOD Integration for Claude Hook System.

Enhances Context Intelligence Engine with BMAD's Agentic Planning roles
and Context-Engineered Development Stories for better task orchestration.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class BMADRole(Enum):
    """BMAD Agentic Planning roles."""
    ANALYST = "requirements-analyst"
    PM = "project-manager-agent" 
    ARCHITECT = "system-architect"
    SCRUM_MASTER = "workflow-agent"
    DEVELOPER = "spec-developer"
    QA = "qa-expert"


@dataclass
class BMADStory:
    """BMAD Context-Engineered Development Story."""
    story_id: str
    title: str
    context: str
    background: str
    acceptance_criteria: List[str]
    implementation_hints: List[str]
    edge_cases: List[str]
    dependencies: List[str]
    estimated_complexity: str
    required_roles: List[BMADRole]


class BMADRoleDetector:
    """Detects when BMAD Agentic Planning roles are needed."""
    
    # Role detection patterns - maps prompt patterns to BMAD roles
    ROLE_PATTERNS = {
        BMADRole.ANALYST: [
            r"(analyze|requirement|spec|define|gather|understand|investigate|research|study)",
            r"(what do we need|what are the requirements|business needs|user needs)",
            r"(analyze the problem|understand the system|investigate the issue)"
        ],
        BMADRole.PM: [
            r"(project|plan|manage|coordinate|timeline|milestone|schedule|prioritize)",
            r"(project plan|roadmap|timeline|deliverable|scope|phase)",
            r"(manage the|coordinate with|organize the|plan out)"
        ],
        BMADRole.ARCHITECT: [
            r"(architecture|design|system|structure|pattern|scalable|framework)",
            r"(system design|architecture|technical design|design pattern|framework)",
            r"(design the system|architect the|design patterns|scalability)"
        ],
        BMADRole.SCRUM_MASTER: [
            r"(workflow|process|methodology|agile|scrum|sprint|story|backlog)",
            r"(development process|workflow|methodology|team process)",
            r"(organize the work|break down|workflow|process improvement)"
        ],
        BMADRole.DEVELOPER: [
            r"(implement|build|create|code|develop|feature|function|component)",
            r"(build the|implement the|create a|develop a|code the)",
            r"(implementation|coding|development|building)"
        ],
        BMADRole.QA: [
            r"(test|quality|validate|verify|check|ensure|qa|testing)",
            r"(test the|quality assurance|testing strategy|validation)",
            r"(ensure quality|test coverage|validation|verification)"
        ]
    }
    
    def detect_required_roles(self, prompt: str) -> List[BMADRole]:
        """Detect which BMAD roles are needed for the prompt."""
        prompt_lower = prompt.lower()
        detected_roles = []
        role_scores = {}
        
        # Calculate scores for each role
        for role, patterns in self.ROLE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, prompt_lower))
                score += matches
            
            role_scores[role] = score
            
            # Threshold for role detection
            if score > 0:
                detected_roles.append(role)
        
        # Sort by confidence (highest score first)
        detected_roles.sort(key=lambda role: role_scores[role], reverse=True)
        
        # For complex tasks, ensure we have core planning roles
        if self._is_complex_task(prompt_lower):
            self._ensure_core_planning_roles(detected_roles)
        
        # Limit to top 4 roles to avoid over-allocation
        return detected_roles[:4]
    
    def _is_complex_task(self, prompt_lower: str) -> bool:
        """Determine if task requires complex planning."""
        complex_indicators = [
            "entire system", "full stack", "complete solution", "end-to-end",
            "architecture", "scalable", "enterprise", "production-ready",
            "microservices", "distributed", "multi-tenant", "integration",
            "migration", "refactor", "modernize", "large scale"
        ]
        
        return any(indicator in prompt_lower for indicator in complex_indicators)
    
    def _ensure_core_planning_roles(self, detected_roles: List[BMADRole]) -> None:
        """Ensure core planning roles are present for complex tasks."""
        core_roles = [BMADRole.ANALYST, BMADRole.ARCHITECT, BMADRole.PM]
        
        for role in core_roles:
            if role not in detected_roles:
                detected_roles.insert(-1, role)  # Insert before last role


class BMADStoryGenerator:
    """Generates BMAD Context-Engineered Development Stories."""
    
    def __init__(self):
        """Initialize story generator."""
        self.story_templates = {
            "development": {
                "context": "Development task requiring implementation of new functionality",
                "background": "User needs new feature implemented with proper testing and documentation",
                "default_criteria": ["Feature works as specified", "Tests pass", "Code reviewed"]
            },
            "architecture": {
                "context": "System architecture design task requiring technical planning",
                "background": "System needs architectural design or improvement for scalability/maintainability",
                "default_criteria": ["Architecture documented", "Scalability considered", "Performance analyzed"]
            },
            "analysis": {
                "context": "Analysis task requiring investigation and requirements gathering",
                "background": "Problem or requirement needs thorough analysis before implementation",
                "default_criteria": ["Analysis complete", "Requirements documented", "Options evaluated"]
            }
        }
    
    def generate_story(self, 
                      prompt: str, 
                      roles: List[BMADRole],
                      project_context: Dict[str, Any]) -> BMADStory:
        """Generate context-engineered story from prompt and roles."""
        
        # Determine story category
        category = self._categorize_task(prompt, roles)
        template = self.story_templates.get(category, self.story_templates["development"])
        
        # Generate story components
        story_id = f"BMAD-{len(prompt.split())}-{hash(prompt) % 1000:03d}"
        title = self._generate_title(prompt)
        context = self._enhance_context(template["context"], project_context)
        background = self._enhance_background(template["background"], prompt, project_context)
        
        # Generate acceptance criteria
        acceptance_criteria = self._generate_acceptance_criteria(prompt, template["default_criteria"])
        
        # Generate implementation hints
        implementation_hints = self._generate_implementation_hints(prompt, roles, project_context)
        
        # Generate edge cases
        edge_cases = self._generate_edge_cases(prompt, project_context)
        
        # Generate dependencies
        dependencies = self._generate_dependencies(prompt, roles)
        
        # Estimate complexity
        complexity = self._estimate_complexity(prompt, roles, project_context)
        
        return BMADStory(
            story_id=story_id,
            title=title,
            context=context,
            background=background,
            acceptance_criteria=acceptance_criteria,
            implementation_hints=implementation_hints,
            edge_cases=edge_cases,
            dependencies=dependencies,
            estimated_complexity=complexity,
            required_roles=roles
        )
    
    def _categorize_task(self, prompt: str, roles: List[BMADRole]) -> str:
        """Categorize task based on prompt and roles."""
        if BMADRole.ARCHITECT in roles or "architecture" in prompt.lower():
            return "architecture"
        elif BMADRole.ANALYST in roles or any(word in prompt.lower() for word in ["analyze", "research", "investigate"]):
            return "analysis"
        else:
            return "development"
    
    def _generate_title(self, prompt: str) -> str:
        """Generate concise title from prompt."""
        # Extract key action and object
        words = prompt.split()
        if len(words) <= 6:
            return prompt.title()
        
        # Find the main verb and object
        important_words = []
        for word in words[:8]:  # Look at first 8 words
            if word.lower() in ["implement", "build", "create", "design", "analyze", "develop", "fix", "add", "update"]:
                important_words.append(word)
            elif len(word) > 3 and word.lower() not in ["the", "and", "for", "with", "this", "that"]:
                important_words.append(word)
        
        return " ".join(important_words[:5]).title()
    
    def _enhance_context(self, base_context: str, project_context: Dict[str, Any]) -> str:
        """Enhance context with project-specific information."""
        tech_stacks = project_context.get("technology_stacks", [])
        project_size = project_context.get("project_size", "unknown")
        
        enhanced = base_context
        if tech_stacks:
            enhanced += f" using {', '.join(tech_stacks[:2])}"
        if project_size != "unknown":
            enhanced += f" in a {project_size} project"
        
        return enhanced
    
    def _enhance_background(self, base_background: str, prompt: str, project_context: Dict[str, Any]) -> str:
        """Enhance background with prompt and project context."""
        git_status = project_context.get("git_status", {})
        
        enhanced = f"{base_background}. Original request: '{prompt[:100]}...'"
        
        if git_status.get("uncommitted_changes", 0) > 0:
            enhanced += f" Note: {git_status['uncommitted_changes']} uncommitted changes in repository."
        
        return enhanced
    
    def _generate_acceptance_criteria(self, prompt: str, default_criteria: List[str]) -> List[str]:
        """Generate acceptance criteria based on prompt."""
        criteria = default_criteria.copy()
        
        prompt_lower = prompt.lower()
        
        # Add specific criteria based on prompt content
        if "test" in prompt_lower:
            criteria.append("Comprehensive tests implemented")
        if "performance" in prompt_lower:
            criteria.append("Performance requirements met")
        if "security" in prompt_lower:
            criteria.append("Security review passed")
        if "documentation" in prompt_lower:
            criteria.append("Documentation updated")
        if "api" in prompt_lower:
            criteria.append("API contract validated")
        
        return criteria
    
    def _generate_implementation_hints(self, 
                                     prompt: str, 
                                     roles: List[BMADRole],
                                     project_context: Dict[str, Any]) -> List[str]:
        """Generate implementation hints based on context."""
        hints = []
        tech_stacks = project_context.get("technology_stacks", [])
        
        # Technology-specific hints
        if "React" in tech_stacks:
            hints.append("Use functional components and hooks for React implementation")
        if "Python" in tech_stacks:
            hints.append("Follow PEP 8 style guidelines and use type hints")
        if "Node.js" in tech_stacks:
            hints.append("Use async/await patterns for asynchronous operations")
        
        # Role-specific hints
        if BMADRole.ARCHITECT in roles:
            hints.append("Consider scalability and maintainability in design decisions")
        if BMADRole.QA in roles:
            hints.append("Implement comprehensive test coverage including edge cases")
        if BMADRole.DEVELOPER in roles:
            hints.append("Follow existing code patterns and conventions in the project")
        
        return hints[:3]  # Limit to 3 hints
    
    def _generate_edge_cases(self, prompt: str, project_context: Dict[str, Any]) -> List[str]:
        """Generate edge cases to consider."""
        edge_cases = []
        prompt_lower = prompt.lower()
        
        # Common edge cases based on task type
        if any(word in prompt_lower for word in ["user", "input", "form", "data"]):
            edge_cases.append("Invalid or malformed input data")
            edge_cases.append("Empty or null values")
        
        if any(word in prompt_lower for word in ["api", "service", "network", "request"]):
            edge_cases.append("Network failures and timeouts")
            edge_cases.append("Rate limiting and throttling")
        
        if any(word in prompt_lower for word in ["database", "storage", "save", "persist"]):
            edge_cases.append("Database connection failures")
            edge_cases.append("Transaction rollback scenarios")
        
        return edge_cases[:3]  # Limit to 3 edge cases
    
    def _generate_dependencies(self, prompt: str, roles: List[BMADRole]) -> List[str]:
        """Generate dependencies for the story."""
        dependencies = []
        
        # Role-based dependencies
        if BMADRole.ANALYST in roles and BMADRole.ARCHITECT not in roles:
            dependencies.append("Requirements analysis must be completed first")
        
        if BMADRole.ARCHITECT in roles and BMADRole.DEVELOPER in roles:
            dependencies.append("Architecture design must be approved before implementation")
        
        if BMADRole.QA in roles:
            dependencies.append("Implementation must be complete before QA testing")
        
        # Task-specific dependencies
        if "integration" in prompt.lower():
            dependencies.append("Both systems must be available for integration testing")
        if "deployment" in prompt.lower():
            dependencies.append("All tests must pass before deployment")
        
        return dependencies
    
    def _estimate_complexity(self, 
                           prompt: str, 
                           roles: List[BMADRole],
                           project_context: Dict[str, Any]) -> str:
        """Estimate story complexity."""
        complexity_score = 0
        
        # Base complexity from prompt length and keywords
        word_count = len(prompt.split())
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        # Role complexity
        complexity_score += len(roles)
        
        # Context complexity
        if project_context.get("project_size") in ["large", "enterprise"]:
            complexity_score += 2
        
        # Keyword complexity
        complex_keywords = ["architecture", "integration", "migration", "scalable", "distributed"]
        complexity_score += sum(1 for keyword in complex_keywords if keyword in prompt.lower())
        
        # Map score to complexity level
        if complexity_score <= 2:
            return "Simple"
        elif complexity_score <= 5:
            return "Medium"
        elif complexity_score <= 8:
            return "Complex"
        else:
            return "Enterprise"


class BMADIntegration:
    """Main BMAD integration class for Context Intelligence Engine."""
    
    def __init__(self):
        """Initialize BMAD integration."""
        self.role_detector = BMADRoleDetector()
        self.story_generator = BMADStoryGenerator()
    
    def enhance_directive_with_bmad(self, 
                                  directive: str,
                                  prompt: str,
                                  project_context: Dict[str, Any]) -> str:
        """Enhance directive with BMAD concepts."""
        
        # Detect required BMAD roles
        required_roles = self.role_detector.detect_required_roles(prompt)
        
        # If no roles detected or simple task, return original directive
        if not required_roles or (len(required_roles) == 1 and "simple" in directive.lower()):
            return directive
        
        # Generate BMAD story
        story = self.story_generator.generate_story(prompt, required_roles, project_context)
        
        # Enhance directive with BMAD story elements
        enhanced_directive = self._format_enhanced_directive(directive, story, required_roles)
        
        return enhanced_directive
    
    def _format_enhanced_directive(self, 
                                 original_directive: str,
                                 story: BMADStory,
                                 roles: List[BMADRole]) -> str:
        """Format enhanced directive with BMAD story elements."""
        
        # Extract the original directive content
        lines = original_directive.split('\n')
        directive_content = []
        
        for line in lines:
            if line.strip() and not line.startswith('🎗') and not line.startswith('💼'):
                directive_content.append(line)
        
        # Build enhanced directive
        enhanced = []
        
        # Add BMAD header
        enhanced.append("🎯 BMAD-ENHANCED TASK ORCHESTRATION")
        enhanced.append("")
        
        # Add story context
        enhanced.append(f"📋 CONTEXT STORY: {story.story_id}")
        enhanced.append(f"• Title: {story.title}")
        enhanced.append(f"• Context: {story.context}")
        enhanced.append(f"• Complexity: {story.estimated_complexity}")
        enhanced.append("")
        
        # Add required roles
        enhanced.append("👥 REQUIRED BMAD ROLES:")
        for role in roles[:3]:  # Limit to top 3 roles
            enhanced.append(f"• {role.value}")
        enhanced.append("")
        
        # Add acceptance criteria
        if story.acceptance_criteria:
            enhanced.append("✅ ACCEPTANCE CRITERIA:")
            for criteria in story.acceptance_criteria[:3]:  # Limit to 3 criteria
                enhanced.append(f"• {criteria}")
            enhanced.append("")
        
        # Add implementation hints
        if story.implementation_hints:
            enhanced.append("💡 IMPLEMENTATION HINTS:")
            for hint in story.implementation_hints:
                enhanced.append(f"• {hint}")
            enhanced.append("")
        
        # Add original directive content
        enhanced.append("⚡ ZEN ORCHESTRATION:")
        enhanced.extend(directive_content)
        
        # Add BMAD workflow
        enhanced.append("")
        enhanced.append("🔄 BMAD WORKFLOW:")
        enhanced.append("1. Analyst → Requirements gathering")
        enhanced.append("2. Architect → System design") 
        enhanced.append("3. PM → Task coordination")
        enhanced.append("4. Developer → Implementation")
        enhanced.append("5. QA → Quality validation")
        
        return '\n'.join(enhanced)


# Integration function for Context Intelligence Engine
def enhance_context_directive_with_bmad(directive: str, 
                                      prompt: str,
                                      project_context: Dict[str, Any]) -> str:
    """Enhance context directive with BMAD integration."""
    bmad = BMADIntegration()
    return bmad.enhance_directive_with_bmad(directive, prompt, project_context)


if __name__ == "__main__":
    # Test BMAD integration
    print("🎯 BMAD Integration Test")
    print("=" * 50)
    
    # Test role detection
    detector = BMADRoleDetector()
    test_prompts = [
        "Design a scalable architecture for our microservices system",
        "Analyze the requirements for the new user management feature",
        "Implement the payment processing API with error handling",
        "Create a project plan for the mobile app development"
    ]
    
    for prompt in test_prompts:
        roles = detector.detect_required_roles(prompt)
        print(f"\nPrompt: {prompt}")
        print(f"Detected Roles: {[role.value for role in roles]}")
    
    # Test story generation
    story_gen = BMADStoryGenerator()
    context = {
        "technology_stacks": ["React", "Node.js"],
        "project_size": "medium",
        "git_status": {"uncommitted_changes": 3}
    }
    
    story = story_gen.generate_story(test_prompts[0], [BMADRole.ARCHITECT, BMADRole.DEVELOPER], context)
    print(f"\n📋 Generated Story: {story.story_id}")
    print(f"Title: {story.title}")
    print(f"Context: {story.context}")
    print(f"Acceptance Criteria: {story.acceptance_criteria}")