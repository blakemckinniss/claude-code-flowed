#!/usr/bin/env python3
"""Command Templates Analyzer for Claude Code.

Provides direct, actionable command templates for MCP operations,
agent spawning, and ZEN consultation based on task context.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from modules.core import Analyzer, PatternMatch


class CommandTemplatesAnalyzer(Analyzer):
    """Provides direct command templates for common workflows."""
    
    def __init__(self):
        super().__init__()
        self.name = "command_templates"
        self.priority = 100  # Highest priority - commands should appear first
        
        # Task pattern matchers with associated command templates
        self.task_patterns = {
            # Development tasks
            "feature_development": {
                "patterns": ["feature", "implement", "build", "create", "develop"],
                "template": self._get_feature_dev_template
            },
            "bug_fix": {
                "patterns": ["bug", "fix", "issue", "error", "problem", "debug"],
                "template": self._get_bug_fix_template
            },
            "refactor": {
                "patterns": ["refactor", "clean", "improve", "optimize", "restructure"],
                "template": self._get_refactor_template
            },
            "testing": {
                "patterns": ["test", "spec", "tdd", "unit test", "integration"],
                "template": self._get_testing_template
            },
            "code_review": {
                "patterns": ["review", "code review", "check", "analyze code"],
                "template": self._get_code_review_template
            },
            
            # GitHub workflows
            "pull_request": {
                "patterns": ["pr", "pull request", "merge", "github pr"],
                "template": self._get_pr_template
            },
            "release": {
                "patterns": ["release", "deploy", "tag", "version"],
                "template": self._get_release_template
            },
            
            # Architecture tasks
            "system_design": {
                "patterns": ["design", "architect", "system", "structure"],
                "template": self._get_system_design_template
            },
            "api_design": {
                "patterns": ["api", "endpoint", "rest", "graphql"],
                "template": self._get_api_design_template
            },
            
            # Analysis tasks
            "performance": {
                "patterns": ["performance", "slow", "optimize", "speed"],
                "template": self._get_performance_template
            },
            "security": {
                "patterns": ["security", "vulnerable", "audit", "secure"],
                "template": self._get_security_template
            }
        }
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze prompt and provide command templates."""
        matches = []
        prompt_lower = prompt.lower()
        
        # Find matching task patterns
        matched_tasks = []
        for task_type, config in self.task_patterns.items():
            if any(pattern in prompt_lower for pattern in config["patterns"]):
                matched_tasks.append((task_type, config))
        
        # Generate command templates for matched tasks
        for task_type, config in matched_tasks:
            template = config["template"]()
            matches.append(PatternMatch(
                analyzer=self.name,
                pattern=f"{task_type}_commands",
                confidence=0.95,
                metadata={
                    "message": template,
                    "category": "command_template",
                    "task_type": task_type
                }
            ))
        
        # Always add quick command reference
        matches.append(self._get_quick_reference())
        
        return matches
    
    def _get_feature_dev_template(self) -> str:
        """Template for feature development."""
        return """🚀 FEATURE DEVELOPMENT COMMAND SEQUENCE

1️⃣ CONSULT QUEEN ZEN (Strategy & Planning):
```javascript
mcp__zen__thinkdeep({
  step: "Analyze feature requirements and design approach",
  model: "anthropic/claude-opus-4",
  thinking_mode: "high",
  total_steps: 3,
  next_step_required: true
})
```

2️⃣ INITIALIZE HIVE SWARM (Worker Coordination):
```javascript
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 8,
  strategy: "adaptive"
})
```

3️⃣ SPAWN WORKER AGENTS (In ONE Message):
```javascript
Task("system-architect", "Design feature architecture", "architect")
Task("backend-dev", "Implement server logic", "backend")
Task("mobile-dev", "Build frontend UI", "frontend")
Task("api-docs", "Document API endpoints", "docs")
Task("tester", "Create test suite", "testing")
Task("reviewer", "Code quality check", "review")
```

4️⃣ BATCH TODO MANAGEMENT:
```javascript
TodoWrite({
  todos: [
    { id: "1", content: "Design feature architecture", status: "pending", priority: "high" },
    { id: "2", content: "Set up project structure", status: "pending", priority: "high" },
    { id: "3", content: "Implement data models", status: "pending", priority: "medium" },
    { id: "4", content: "Create API endpoints", status: "pending", priority: "medium" },
    { id: "5", content: "Build UI components", status: "pending", priority: "medium" },
    { id: "6", content: "Write unit tests", status: "pending", priority: "medium" },
    { id: "7", content: "Integration testing", status: "pending", priority: "low" },
    { id: "8", content: "Documentation", status: "pending", priority: "low" }
  ]
})
```"""

    def _get_bug_fix_template(self) -> str:
        """Template for bug fixing."""
        return """🐛 BUG FIX COMMAND SEQUENCE

1️⃣ DEEP INVESTIGATION WITH QUEEN ZEN:
```javascript
mcp__zen__debug({
  step: "Investigate bug root cause",
  model: "anthropic/claude-opus-4",
  thinking_mode: "max",
  total_steps: 5,
  confidence: "exploring",
  next_step_required: true
})
```

2️⃣ DEPLOY DEBUGGING SWARM:
```javascript
// Single message with all agents
Task("debugger", "Trace execution flow", "debug")
Task("error-detective", "Analyze error patterns", "detective")
Task("performance-engineer", "Check performance impact", "perf")
Task("tester", "Reproduce and validate fix", "test")
```

3️⃣ SYSTEMATIC INVESTIGATION:
```javascript
// Batch all search operations
Grep({ pattern: "error.*message", output_mode: "content", -B: 5, -A: 5 })
Grep({ pattern: "exception|fault|crash", output_mode: "files_with_matches" })
Glob({ pattern: "**/*.log" })
```

4️⃣ STORE FINDINGS IN HIVE MEMORY:
```javascript
mcp__claude-flow__memory_usage({
  action: "store",
  key: "bug_analysis_${bug_id}",
  value: JSON.stringify({
    symptoms: "...",
    root_cause: "...",
    fix_approach: "..."
  }),
  namespace: "bug_fixes",
  ttl: 2592000  // 30 days
})
```"""

    def _get_refactor_template(self) -> str:
        """Template for refactoring."""
        return """♻️ REFACTORING COMMAND SEQUENCE

1️⃣ ARCHITECTURAL REVIEW WITH QUEEN ZEN:
```javascript
mcp__zen__analyze({
  step: "Analyze code quality and refactoring opportunities",
  model: "anthropic/claude-opus-4",
  analysis_type: "quality",
  thinking_mode: "high",
  total_steps: 4
})
```

2️⃣ CONSENSUS BUILDING (Multiple Models):
```javascript
mcp__zen__consensus({
  step: "Should we refactor this module?",
  models: [
    { model: "anthropic/claude-opus-4", stance: "for" },
    { model: "openai/o3", stance: "neutral" },
    { model: "google/gemini-2.5-pro", stance: "against" }
  ],
  total_steps: 3,
  next_step_required: true
})
```

3️⃣ REFACTORING SWARM:
```javascript
// All in one message
Task("code-refactorer", "Improve code structure", "refactor")
Task("legacy-modernizer", "Update deprecated patterns", "modernize")
Task("performance-optimizer", "Optimize hot paths", "optimize")
Task("test-automator", "Update test suite", "test")
```

4️⃣ BATCH FILE OPERATIONS:
```javascript
// Read all files first
Read({ file_path: "src/old_module.js" })
Read({ file_path: "src/utils.js" })
Read({ file_path: "tests/old_module.test.js" })

// Then batch all edits
MultiEdit({
  file_path: "src/new_module.js",
  edits: [
    { old_string: "class OldPattern", new_string: "class NewPattern" },
    { old_string: "callback(err, data)", new_string: "async/await pattern" },
    // ... more edits
  ]
})
```"""

    def _get_testing_template(self) -> str:
        """Template for testing workflows."""
        return """🧪 TESTING COMMAND SEQUENCE

1️⃣ TEST STRATEGY WITH QUEEN ZEN:
```javascript
mcp__zen__planner({
  step: "Design comprehensive test strategy",
  model: "anthropic/claude-opus-4",
  total_steps: 5,
  next_step_required: true
})
```

2️⃣ TDD SPECIALIST SWARM:
```javascript
// Deploy all test specialists
Task("tdd-london-swarm", "Mock-driven development", "tdd")
Task("test-automator", "Create test suite", "auto")
Task("qa-expert", "Quality assurance", "qa")
Task("performance-benchmarker", "Performance tests", "perf")
```

3️⃣ GENERATE TESTS WITH AI:
```javascript
mcp__zen__testgen({
  step: "Generate comprehensive test cases",
  model: "anthropic/claude-opus-4",
  thinking_mode: "high",
  focus_on: ["edge cases", "error handling", "async behavior"]
})
```

4️⃣ EXECUTE TEST SUITE:
```javascript
// Batch all test commands
Bash({ command: "npm test -- --coverage" })
Bash({ command: "npm run test:integration" })
Bash({ command: "npm run test:e2e" })
```"""

    def _get_code_review_template(self) -> str:
        """Template for code review."""
        return """🔍 CODE REVIEW COMMAND SEQUENCE

1️⃣ COMPREHENSIVE REVIEW WITH QUEEN ZEN:
```javascript
mcp__zen__codereview({
  step: "Analyze code quality, security, and patterns",
  model: "anthropic/claude-opus-4",
  review_type: "full",
  thinking_mode: "high",
  confidence: "exploring",
  total_steps: 5
})
```

2️⃣ SECURITY AUDIT:
```javascript
mcp__zen__secaudit({
  step: "Security vulnerability assessment",
  model: "openai/o3",
  audit_focus: "owasp",
  threat_level: "high",
  total_steps: 4
})
```

3️⃣ REVIEW SWARM DEPLOYMENT:
```javascript
// All reviewers in one message
Task("code-reviewer", "General code quality", "review")
Task("security-auditor", "Security vulnerabilities", "security")
Task("architect-reviewer", "Architecture consistency", "arch")
Task("performance-engineer", "Performance issues", "perf")
```

4️⃣ AUTOMATED ANALYSIS:
```javascript
mcp__claude-flow__github_code_review({
  repo: "owner/repo",
  pr: 123
})
```"""

    def _get_pr_template(self) -> str:
        """Template for pull request workflows."""
        return """🔀 PULL REQUEST COMMAND SEQUENCE

1️⃣ PR STRATEGY WITH QUEEN ZEN:
```javascript
mcp__zen__chat({
  prompt: "Review PR changes and suggest improvements",
  model: "anthropic/claude-opus-4",
  use_websearch: true
})
```

2️⃣ GITHUB PR SWARM:
```javascript
// Deploy PR specialists
Task("pr-manager", "Manage PR lifecycle", "pr")
Task("code-review-swarm", "Automated review", "review")
Task("github-modes", "Batch operations", "batch")
```

3️⃣ AUTOMATED PR REVIEW:
```javascript
// Create pending review
mcp__github__create_pending_pull_request_review({
  owner: "owner",
  repo: "repo",
  pullNumber: 123
})

// Add review comments
mcp__github__add_pull_request_review_comment_to_pending_review({
  owner: "owner",
  repo: "repo",
  pullNumber: 123,
  path: "src/file.js",
  line: 42,
  body: "Consider using async/await",
  side: "RIGHT"
})

// Submit review
mcp__github__submit_pending_pull_request_review({
  owner: "owner",
  repo: "repo",
  pullNumber: 123,
  event: "APPROVE",
  body: "LGTM with minor suggestions"
})
```"""

    def _get_release_template(self) -> str:
        """Template for release workflows."""
        return """🚀 RELEASE COMMAND SEQUENCE

1️⃣ RELEASE PLANNING WITH QUEEN ZEN:
```javascript
mcp__zen__planner({
  step: "Plan release strategy and checklist",
  model: "anthropic/claude-opus-4",
  total_steps: 6,
  next_step_required: true
})
```

2️⃣ RELEASE COORDINATION SWARM:
```javascript
// Deploy release team
Task("release-manager", "Coordinate release", "release")
Task("release-swarm", "Automated release", "swarm")
Task("workflow-automation", "CI/CD automation", "cicd")
```

3️⃣ PRE-RELEASE CHECKS:
```javascript
// Batch all checks
Bash({ command: "npm test" })
Bash({ command: "npm run build" })
Bash({ command: "npm run lint" })
Bash({ command: "npm audit" })
```

4️⃣ GITHUB RELEASE:
```javascript
mcp__claude-flow__github_release_coord({
  repo: "owner/repo",
  version: "v1.2.3"
})
```"""

    def _get_system_design_template(self) -> str:
        """Template for system design."""
        return """🏗️ SYSTEM DESIGN COMMAND SEQUENCE

1️⃣ ARCHITECTURAL DEEP DIVE WITH QUEEN ZEN:
```javascript
mcp__zen__thinkdeep({
  step: "Design system architecture",
  model: "anthropic/claude-opus-4",
  thinking_mode: "max",
  focus_areas: ["scalability", "security", "performance"],
  total_steps: 5
})
```

2️⃣ MULTI-MODEL CONSENSUS:
```javascript
mcp__zen__consensus({
  step: "Evaluate architecture decisions",
  models: [
    { model: "anthropic/claude-opus-4" },
    { model: "openai/o3" },
    { model: "google/gemini-2.5-pro" }
  ],
  total_steps: 4
})
```

3️⃣ ARCHITECTURE SWARM:
```javascript
// Deploy architecture team
Task("system-architect", "High-level design", "architect")
Task("cloud-architect", "Infrastructure design", "cloud")
Task("database-optimizer", "Data architecture", "db")
Task("security-manager", "Security architecture", "security")
Task("api-architect", "API design", "api")
```

4️⃣ DOCUMENTATION GENERATION:
```javascript
mcp__zen__docgen({
  step: "Generate architecture documentation",
  model: "anthropic/claude-opus-4",
  document_flow: true,
  document_complexity: true
})
```"""

    def _get_api_design_template(self) -> str:
        """Template for API design."""
        return """🔌 API DESIGN COMMAND SEQUENCE

1️⃣ API STRATEGY WITH QUEEN ZEN:
```javascript
mcp__zen__chat({
  prompt: "Design RESTful/GraphQL API architecture",
  model: "anthropic/claude-opus-4",
  use_websearch: true
})
```

2️⃣ API SPECIALIST SWARM:
```javascript
// Deploy API team
Task("api-architect", "API contract design", "architect")
Task("api-docs", "OpenAPI/Swagger specs", "docs")
Task("graphql-architect", "GraphQL schema", "graphql")
Task("api-documenter", "Developer docs", "documenter")
```

3️⃣ GENERATE API SPECIFICATION:
```javascript
Write({
  file_path: "api/openapi.yaml",
  content: `openapi: 3.0.0
info:
  title: API Specification
  version: 1.0.0
paths:
  /users:
    get:
      summary: List users
      # ... full spec
`
})
```"""

    def _get_performance_template(self) -> str:
        """Template for performance optimization."""
        return """⚡ PERFORMANCE OPTIMIZATION SEQUENCE

1️⃣ PERFORMANCE ANALYSIS WITH QUEEN ZEN:
```javascript
mcp__zen__analyze({
  step: "Analyze performance bottlenecks",
  model: "anthropic/claude-opus-4",
  analysis_type: "performance",
  thinking_mode: "high",
  total_steps: 5
})
```

2️⃣ PERFORMANCE SWARM:
```javascript
// Deploy performance team
Task("performance-engineer", "Profile application", "perf")
Task("performance-benchmarker", "Run benchmarks", "bench")
Task("perf-analyzer", "Analyze bottlenecks", "analyze")
Task("performance-optimizer", "Implement optimizations", "optimize")
```

3️⃣ PROFILING & BENCHMARKS:
```javascript
// Batch profiling commands
Bash({ command: "npm run profile" })
Bash({ command: "ab -n 1000 -c 10 http://localhost:3000/" })
Bash({ command: "lighthouse http://localhost:3000 --output json" })
```

4️⃣ NEURAL OPTIMIZATION:
```javascript
mcp__claude-flow__neural_train({
  pattern_type: "optimization",
  training_data: "performance_metrics",
  epochs: 50
})
```"""

    def _get_security_template(self) -> str:
        """Template for security workflows."""
        return """🔒 SECURITY AUDIT COMMAND SEQUENCE

1️⃣ COMPREHENSIVE SECURITY AUDIT:
```javascript
mcp__zen__secaudit({
  step: "Full security assessment",
  model: "anthropic/claude-opus-4",
  audit_focus: "comprehensive",
  threat_level: "critical",
  compliance_requirements: ["OWASP", "SOC2", "GDPR"],
  thinking_mode: "max",
  total_steps: 6
})
```

2️⃣ SECURITY SPECIALIST SWARM:
```javascript
// Deploy security team
Task("security-auditor", "Vulnerability scan", "audit")
Task("security-manager", "Security architecture", "manager")
Task("network-engineer", "Network security", "network")
```

3️⃣ AUTOMATED SECURITY SCANS:
```javascript
// Batch security commands
Bash({ command: "npm audit --json" })
Bash({ command: "trivy fs ." })
Bash({ command: "semgrep --config=auto ." })
```

4️⃣ STORE SECURITY FINDINGS:
```javascript
mcp__claude-flow__memory_usage({
  action: "store",
  key: "security_audit_${date}",
  value: JSON.stringify(findings),
  namespace: "security",
  ttl: 7776000  // 90 days
})
```"""

    def _get_quick_reference(self) -> PatternMatch:
        """Quick reference for common commands."""
        return PatternMatch(
            analyzer=self.name,
            pattern="quick_reference",
            confidence=1.0,
            metadata={
                "message": """⚡ QUICK COMMAND REFERENCE

🧠 QUEEN ZEN CONSULTATION:
• `mcp__zen__chat` - Strategic planning & guidance
• `mcp__zen__thinkdeep` - Deep analysis (5-step workflows)
• `mcp__zen__consensus` - Multi-model decisions
• `mcp__zen__planner` - Sequential planning
• `mcp__zen__debug` - Root cause analysis

🐝 HIVE SWARM COORDINATION:
• `mcp__claude-flow__swarm_init` - Initialize hive (mesh/hierarchical)
• `mcp__claude-flow__agent_spawn` - Deploy worker agents
• `mcp__claude-flow__task_orchestrate` - Coordinate tasks
• `mcp__claude-flow__memory_usage` - Hive memory

🔧 EXECUTION (Claude Code Only):
• `Task()` - Spawn agents (batch in ONE message!)
• `TodoWrite()` - Task management (5-10+ todos!)
• `Read/Write/Edit` - File operations (batch!)
• `Bash()` - Commands (batch parallel!)

📝 REMEMBER THE HIVE LAW:
1. Queen ZEN commands first
2. Flow Workers coordinate
3. Storage Workers manage data
4. Execution Drones implement
5. ALWAYS batch operations!""",
                "category": "quick_reference",
                "priority": "always_show"
            }
        )