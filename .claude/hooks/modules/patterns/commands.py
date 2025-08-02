#!/usr/bin/env python3
"""Command Patterns Analyzer - Provides direct actionable commands.

This analyzer provides specific, copy-paste ready command sequences
for common workflows with Claude-Flow MCP and ZEN consultation.
"""

import re
from typing import List, Dict, Any, Optional
from ..core import Analyzer, PatternMatch


class CommandPatternsAnalyzer(Analyzer):
    """Provides direct command templates for immediate execution."""
    
    def __init__(self, priority: int = 2000):  # Highest priority
        super().__init__(priority)
        self.name = "command_patterns"
        
    def get_name(self) -> str:
        """Get the name of this analyzer."""
        return self.name
    
    def _initialize_patterns(self) -> None:
        """Initialize patterns for this analyzer."""
        # We'll override analyze() instead of using patterns
        
    def analyze(self, prompt: str) -> List[PatternMatch]:
        """Analyze prompt and provide direct commands."""
        matches = []
        prompt_lower = prompt.lower()
        
        # Always provide the command cheatsheet
        matches.append(self._get_command_cheatsheet())
        
        # Task-specific command sequences
        if any(word in prompt_lower for word in ["feature", "implement", "build", "create"]):
            matches.append(self._get_feature_commands())
            
        if any(word in prompt_lower for word in ["bug", "fix", "debug", "error"]):
            matches.append(self._get_debug_commands())
            
        if any(word in prompt_lower for word in ["test", "tdd", "spec"]):
            matches.append(self._get_testing_commands())
            
        if any(word in prompt_lower for word in ["review", "analyze", "check"]):
            matches.append(self._get_review_commands())
            
        if any(word in prompt_lower for word in ["pr", "pull request", "github"]):
            matches.append(self._get_github_commands())
            
        if any(word in prompt_lower for word in ["performance", "slow", "optimize"]):
            matches.append(self._get_performance_commands())
            
        if any(word in prompt_lower for word in ["security", "audit", "vulnerable"]):
            matches.append(self._get_security_commands())
            
        return matches
    
    def _get_command_cheatsheet(self) -> PatternMatch:
        """Essential command cheatsheet."""
        return PatternMatch(
            pattern="command_cheatsheet",
            message="""🎯 DIRECT COMMAND EXECUTION GUIDE

👑 QUEEN ZEN COMMANDS (Copy & Execute):
```javascript
// Strategic Planning
mcp__zen__chat({ 
  prompt: "YOUR_TASK_HERE", 
  model: "anthropic/claude-opus-4",
  use_websearch: true 
})

// Deep Analysis (5-step workflow)
mcp__zen__thinkdeep({
  step: "YOUR_ANALYSIS_HERE",
  model: "anthropic/claude-opus-4", 
  thinking_mode: "high",
  total_steps: 5,
  next_step_required: true
})

// Multi-Model Consensus
mcp__zen__consensus({
  step: "YOUR_DECISION_HERE",
  models: [
    { model: "anthropic/claude-opus-4", stance: "for" },
    { model: "openai/o3", stance: "neutral" },
    { model: "google/gemini-2.5-pro", stance: "against" }
  ],
  total_steps: 3
})
```

🐝 HIVE SWARM INITIALIZATION:
```javascript
// Initialize Swarm
mcp__claude-flow__swarm_init({
  topology: "hierarchical",  // or "mesh", "ring", "star"
  maxAgents: 8,
  strategy: "adaptive"
})

// Spawn Multiple Agents (ALWAYS in ONE message!)
Task("system-architect", "Design architecture", "architect")
Task("backend-dev", "Build API", "backend")
Task("mobile-dev", "Create UI", "frontend")
Task("tester", "Write tests", "test")
Task("reviewer", "Review code", "review")
```

📋 BATCH OPERATIONS (GOLDEN RULE):
```javascript
// TodoWrite - ALWAYS 5-10+ items!
TodoWrite({
  todos: [
    { id: "1", content: "Task 1", status: "pending", priority: "high" },
    { id: "2", content: "Task 2", status: "pending", priority: "high" },
    { id: "3", content: "Task 3", status: "pending", priority: "medium" },
    { id: "4", content: "Task 4", status: "pending", priority: "medium" },
    { id: "5", content: "Task 5", status: "pending", priority: "low" }
  ]
})

// File Operations - Batch in ONE message!
Read({ file_path: "/path/file1.js" })
Read({ file_path: "/path/file2.js" })
Write({ file_path: "/path/new.js", content: "..." })

// Bash Commands - Parallel execution!
Bash({ command: "npm test" })
Bash({ command: "npm run build" })
Bash({ command: "npm run lint" })
```

💾 HIVE MEMORY OPERATIONS:
```javascript
// Store findings
mcp__claude-flow__memory_usage({
  action: "store",
  key: "analysis_results",
  value: JSON.stringify(data),
  namespace: "project",
  ttl: 86400  // 24 hours
})

// Retrieve memory
mcp__claude-flow__memory_usage({
  action: "retrieve",
  key: "analysis_results",
  namespace: "project"
})
```""",
            priority=self.priority,
            confidence=1.0,
            metadata={
                "category": "cheatsheet",
                "priority": "always_show"
            }
        )
    
    def _get_feature_commands(self) -> PatternMatch:
        """Feature development commands."""
        return PatternMatch(
            pattern="feature_commands",
            message="""🚀 FEATURE DEVELOPMENT - COPY THESE COMMANDS:

```javascript
// 1. Queen ZEN Strategy Session
mcp__zen__thinkdeep({
  step: "Design feature architecture and implementation plan",
  model: "anthropic/claude-opus-4",
  thinking_mode: "high",
  focus_areas: ["architecture", "scalability", "user experience"],
  total_steps: 3,
  next_step_required: true
})

// 2. Initialize Development Hive
mcp__claude-flow__swarm_init({
  topology: "hierarchical",
  maxAgents: 8,
  strategy: "adaptive"
})

// 3. Deploy Feature Team (ONE MESSAGE!)
Task("system-architect", "Design feature architecture", "architect")
Task("backend-dev", "Implement server logic", "backend")
Task("mobile-dev", "Build UI components", "frontend")
Task("api-architect", "Design API endpoints", "api")
Task("tester", "Create test suite", "test")
Task("documenter", "Write documentation", "docs")

// 4. Comprehensive Todo List
TodoWrite({
  todos: [
    { id: "feat-1", content: "Analyze requirements", status: "in_progress", priority: "high" },
    { id: "feat-2", content: "Design architecture", status: "pending", priority: "high" },
    { id: "feat-3", content: "Set up project structure", status: "pending", priority: "high" },
    { id: "feat-4", content: "Implement data models", status: "pending", priority: "medium" },
    { id: "feat-5", content: "Create API endpoints", status: "pending", priority: "medium" },
    { id: "feat-6", content: "Build UI components", status: "pending", priority: "medium" },
    { id: "feat-7", content: "Write unit tests", status: "pending", priority: "medium" },
    { id: "feat-8", content: "Integration testing", status: "pending", priority: "low" },
    { id: "feat-9", content: "Documentation", status: "pending", priority: "low" },
    { id: "feat-10", content: "Code review", status: "pending", priority: "low" }
  ]
})
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "feature_development"
            }
        )
    
    def _get_debug_commands(self) -> PatternMatch:
        """Debugging commands."""
        return PatternMatch(
            pattern="debug_commands",
            message="""🐛 DEBUG & FIX - COPY THESE COMMANDS:

```javascript
// 1. Deep Bug Investigation
mcp__zen__debug({
  step: "Investigate bug root cause and trace execution flow",
  model: "anthropic/claude-opus-4",
  thinking_mode: "max",
  confidence: "exploring",
  total_steps: 5,
  next_step_required: true
})

// 2. Deploy Debug Squad (ONE MESSAGE!)
Task("debugger", "Trace execution and find root cause", "debug")
Task("error-detective", "Analyze error patterns and logs", "detective")
Task("performance-engineer", "Check performance impact", "perf")
Task("tester", "Reproduce bug and validate fix", "test")

// 3. Systematic Search (BATCH!)
Grep({ pattern: "error|exception|fail", output_mode: "content", -B: 5, -A: 5 })
Grep({ pattern: "stack trace|traceback", output_mode: "files_with_matches" })
Glob({ pattern: "**/*.log" })
Glob({ pattern: "**/test*.js" })

// 4. Store Debug Findings
mcp__claude-flow__memory_usage({
  action: "store",
  key: "bug_analysis_" + Date.now(),
  value: JSON.stringify({
    symptoms: "Describe symptoms",
    root_cause: "Root cause analysis",
    fix_approach: "Proposed solution",
    affected_files: []
  }),
  namespace: "debugging",
  ttl: 2592000  // 30 days
})
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "debugging"
            }
        )
    
    def _get_testing_commands(self) -> PatternMatch:
        """Testing commands."""
        return PatternMatch(
            pattern="testing_commands",
            message="""🧪 TESTING SUITE - COPY THESE COMMANDS:

```javascript
// 1. Test Strategy Planning
mcp__zen__planner({
  step: "Design comprehensive test strategy with coverage goals",
  model: "anthropic/claude-opus-4",
  total_steps: 4,
  next_step_required: true
})

// 2. TDD Specialist Team (ONE MESSAGE!)
Task("tdd-london-swarm", "Mock-driven TDD approach", "tdd")
Task("test-automator", "Automated test generation", "auto")
Task("qa-expert", "Quality assurance strategy", "qa")
Task("performance-benchmarker", "Performance test suite", "perf")

// 3. Generate Test Cases
mcp__zen__testgen({
  step: "Generate comprehensive test cases with edge cases",
  model: "anthropic/claude-opus-4",
  thinking_mode: "high",
  focus_on: ["edge cases", "error handling", "async behavior", "race conditions"]
})

// 4. Execute Test Suite (BATCH!)
Bash({ command: "npm test -- --coverage" })
Bash({ command: "npm run test:unit" })
Bash({ command: "npm run test:integration" })
Bash({ command: "npm run test:e2e" })
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "testing"
            }
        )
    
    def _get_review_commands(self) -> PatternMatch:
        """Code review commands."""
        return PatternMatch(
            pattern="review_commands",
            message="""🔍 CODE REVIEW - COPY THESE COMMANDS:

```javascript
// 1. Comprehensive Code Review
mcp__zen__codereview({
  step: "Analyze code quality, patterns, and potential issues",
  model: "anthropic/claude-opus-4",
  review_type: "full",
  thinking_mode: "high",
  confidence: "exploring",
  severity_filter: "all",
  total_steps: 5
})

// 2. Security Audit
mcp__zen__secaudit({
  step: "Security vulnerability assessment",
  model: "openai/o3",
  audit_focus: "owasp",
  threat_level: "high",
  compliance_requirements: ["OWASP", "SOC2"],
  total_steps: 4
})

// 3. Review Team Deployment (ONE MESSAGE!)
Task("code-reviewer", "General code quality review", "review")
Task("security-auditor", "Security vulnerability scan", "security")
Task("architect-reviewer", "Architecture consistency", "arch")
Task("performance-engineer", "Performance analysis", "perf")
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "code_review"
            }
        )
    
    def _get_github_commands(self) -> PatternMatch:
        """GitHub workflow commands."""
        return PatternMatch(
            pattern="github_commands",
            message="""🔀 GITHUB OPERATIONS - COPY THESE COMMANDS:

```javascript
// 1. PR Review Workflow
mcp__github__get_pull_request({
  owner: "OWNER",
  repo: "REPO",
  pullNumber: PR_NUMBER
})

// 2. Create Pending Review
mcp__github__create_pending_pull_request_review({
  owner: "OWNER",
  repo: "REPO",
  pullNumber: PR_NUMBER
})

// 3. Add Review Comments
mcp__github__add_pull_request_review_comment_to_pending_review({
  owner: "OWNER",
  repo: "REPO",
  pullNumber: PR_NUMBER,
  path: "src/file.js",
  line: 42,
  body: "Consider using async/await pattern",
  side: "RIGHT",
  subjectType: "LINE"
})

// 4. Submit Review
mcp__github__submit_pending_pull_request_review({
  owner: "OWNER",
  repo: "REPO",
  pullNumber: PR_NUMBER,
  event: "APPROVE",  // or "REQUEST_CHANGES", "COMMENT"
  body: "LGTM! Great implementation."
})

// 5. GitHub Automation Team
Task("pr-manager", "Manage pull requests", "pr")
Task("code-review-swarm", "Automated reviews", "review")
Task("github-modes", "Batch operations", "batch")
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "github"
            }
        )
    
    def _get_performance_commands(self) -> PatternMatch:
        """Performance optimization commands."""
        return PatternMatch(
            pattern="performance_commands",
            message="""⚡ PERFORMANCE OPTIMIZATION - COPY THESE COMMANDS:

```javascript
// 1. Performance Analysis
mcp__zen__analyze({
  step: "Analyze performance bottlenecks and optimization opportunities",
  model: "anthropic/claude-opus-4",
  analysis_type: "performance",
  thinking_mode: "high",
  output_format: "actionable",
  total_steps: 5
})

// 2. Performance Team (ONE MESSAGE!)
Task("performance-engineer", "Profile application", "profile")
Task("performance-benchmarker", "Run benchmarks", "bench")
Task("perf-analyzer", "Analyze bottlenecks", "analyze")
Task("performance-optimizer", "Implement fixes", "optimize")

// 3. Profiling Commands (BATCH!)
Bash({ command: "npm run profile" })
Bash({ command: "npm run benchmark" })
Bash({ command: "lighthouse http://localhost:3000 --output json" })
Bash({ command: "clinic doctor -- node app.js" })

// 4. Neural Performance Learning
mcp__claude-flow__neural_train({
  pattern_type: "optimization",
  training_data: "performance_metrics",
  epochs: 50
})
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "performance"
            }
        )
    
    def _get_security_commands(self) -> PatternMatch:
        """Security audit commands."""
        return PatternMatch(
            pattern="security_commands",
            message="""🔒 SECURITY AUDIT - COPY THESE COMMANDS:

```javascript
// 1. Comprehensive Security Audit
mcp__zen__secaudit({
  step: "Full security assessment with vulnerability scanning",
  model: "anthropic/claude-opus-4",
  audit_focus: "comprehensive",
  threat_level: "critical",
  compliance_requirements: ["OWASP", "SOC2", "GDPR", "PCI-DSS"],
  thinking_mode: "max",
  use_websearch: true,
  total_steps: 6
})

// 2. Security Team Deployment (ONE MESSAGE!)
Task("security-auditor", "Vulnerability scanning", "audit")
Task("security-manager", "Security architecture review", "manager")
Task("network-engineer", "Network security analysis", "network")
Task("incident-responder", "Incident response planning", "incident")

// 3. Automated Scans (BATCH!)
Bash({ command: "npm audit --json" })
Bash({ command: "trivy fs . --severity HIGH,CRITICAL" })
Bash({ command: "semgrep --config=auto --json" })
Bash({ command: "snyk test --json" })

// 4. Store Security Report
mcp__claude-flow__memory_usage({
  action: "store",
  key: "security_audit_" + new Date().toISOString(),
  value: JSON.stringify({
    vulnerabilities: [],
    recommendations: [],
    compliance_status: {}
  }),
  namespace: "security",
  ttl: 7776000  // 90 days
})
```""",
            priority=self.priority,
            confidence=0.95,
            metadata={
                "category": "security"
            }
        )