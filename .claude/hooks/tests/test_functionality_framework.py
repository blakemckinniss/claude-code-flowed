#!/usr/bin/env python3
"""Functionality Testing Framework for ZEN Co-pilot System.

This module provides comprehensive functionality validation for:
- ZenConsultant output quality and accuracy
- Agent recommendation accuracy
- Tool selection appropriateness  
- Complexity analysis correctness
- Coordination type determination
- Output format consistency
- Semantic coherence validation
"""

import json
import time
import statistics
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib

# Set up hook paths
from modules.utils.path_resolver import setup_hook_paths
setup_hook_paths()

from modules.core.zen_consultant import (
    ZenConsultant, 
    ComplexityLevel, 
    CoordinationType,
    create_zen_consultation_response,
    create_zen_consensus_request
)


@dataclass
class FunctionalityTestCase:
    """Test case for functionality validation."""
    name: str
    prompt: str
    expected_complexity: ComplexityLevel
    expected_coordination: CoordinationType
    expected_agents: List[str]
    expected_tools: List[str]
    min_confidence: float
    categories: List[str]
    description: str


@dataclass
class FunctionalityTestResult:
    """Functionality test result container."""
    test_case: str
    passed: bool
    accuracy_score: float
    details: Dict[str, Any]
    discrepancies: List[str]
    recommendations: List[str]


class ZenFunctionalityTester:
    """Comprehensive functionality testing suite for ZEN Co-pilot system."""
    
    def __init__(self):
        self.zen_consultant = ZenConsultant()
        self.test_results: List[FunctionalityTestResult] = []
        self.functionality_score = 0.0
        self._initialize_test_cases()
        
    def _initialize_test_cases(self) -> None:
        """Initialize comprehensive test cases for functionality validation."""
        self.test_cases = [
            # Simple tasks
            FunctionalityTestCase(
                name="simple_bug_fix",
                prompt="Fix the login button styling issue",
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_coordination=CoordinationType.SWARM,
                expected_agents=["frontend-developer"],
                expected_tools=["mcp__claude-flow__agent_spawn"],
                min_confidence=0.8,
                categories=["development", "frontend"],
                description="Simple UI bug fix should trigger minimal coordination"
            ),
            
            FunctionalityTestCase(
                name="simple_test_addition",
                prompt="Add unit tests for the authentication function",
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_coordination=CoordinationType.SWARM,
                expected_agents=["tester"],
                expected_tools=["mcp__claude-flow__agent_spawn"],
                min_confidence=0.75,
                categories=["testing", "development"],
                description="Simple testing task should be straightforward"
            ),
            
            # Medium complexity tasks
            FunctionalityTestCase(
                name="medium_refactoring",
                prompt="Refactor the user authentication module to improve performance",
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_coordination=CoordinationType.SWARM,
                expected_agents=["architect", "performance-engineer"],
                expected_tools=["mcp__claude-flow__swarm_init", "mcp__claude-flow__agent_spawn"],
                min_confidence=0.65,
                categories=["architecture", "performance", "refactoring"],
                description="Module refactoring requires architectural consideration"
            ),
            
            FunctionalityTestCase(
                name="medium_api_design",
                prompt="Design REST API endpoints for user management with validation",
                expected_complexity=ComplexityLevel.MEDIUM,
                expected_coordination=CoordinationType.SWARM,
                expected_agents=["api-architect", "security-manager"],
                expected_tools=["mcp__claude-flow__swarm_init"],
                min_confidence=0.7,
                categories=["api", "architecture", "security"],
                description="API design requires security and architectural expertise"
            ),
            
            # Complex tasks
            FunctionalityTestCase(
                name="complex_microservices",
                prompt="Implement microservices architecture with service mesh and observability",
                expected_complexity=ComplexityLevel.COMPLEX,
                expected_coordination=CoordinationType.HIVE,
                expected_agents=["architect", "devops-engineer", "monitoring-specialist"],
                expected_tools=["mcp__zen__planner", "mcp__claude-flow__hive_init"],
                min_confidence=0.5,
                categories=["architecture", "microservices", "devops", "monitoring"],
                description="Complex architecture requiring multi-team coordination"
            ),
            
            FunctionalityTestCase(
                name="complex_ml_pipeline",
                prompt="Build machine learning pipeline with real-time inference and monitoring",
                expected_complexity=ComplexityLevel.COMPLEX,
                expected_coordination=CoordinationType.HIVE,
                expected_agents=["ml-developer", "data-engineer", "performance-benchmarker"],
                expected_tools=["mcp__zen__thinkdeep", "mcp__claude-flow__hive_init"],
                min_confidence=0.45,
                categories=["machine-learning", "data", "performance"],
                description="ML pipeline requires specialized expertise coordination"
            ),
            
            # Enterprise tasks
            FunctionalityTestCase(
                name="enterprise_platform",
                prompt="Design enterprise-scale multi-tenant platform with compliance, security audit, and global deployment",
                expected_complexity=ComplexityLevel.ENTERPRISE,
                expected_coordination=CoordinationType.HIVE,
                expected_agents=["enterprise-architect", "security-manager", "compliance-officer", "devops-engineer"],
                expected_tools=["mcp__zen__planner", "mcp__zen__thinkdeep", "mcp__claude-flow__hive_init"],
                min_confidence=0.3,
                categories=["enterprise", "architecture", "security", "compliance", "devops"],
                description="Enterprise platform requiring comprehensive expertise"
            ),
            
            FunctionalityTestCase(
                name="enterprise_data_platform",
                prompt="Implement enterprise data platform with real-time analytics, ML ops, compliance, and multi-cloud deployment",
                expected_complexity=ComplexityLevel.ENTERPRISE,
                expected_coordination=CoordinationType.HIVE,
                expected_agents=["data-architect", "ml-developer", "compliance-officer", "cloud-architect"],
                expected_tools=["mcp__zen__planner", "mcp__zen__consensus"],
                min_confidence=0.25,
                categories=["enterprise", "data", "machine-learning", "compliance", "cloud"],
                description="Comprehensive data platform with multiple specialized domains"
            ),
            
            # Edge cases
            FunctionalityTestCase(
                name="ambiguous_request",
                prompt="Make it better",
                expected_complexity=ComplexityLevel.SIMPLE,
                expected_coordination=CoordinationType.SWARM,
                expected_agents=[],  # Should suggest clarification
                expected_tools=["mcp__zen__chat"],
                min_confidence=0.1,
                categories=["clarification"],
                description="Ambiguous request should trigger clarification"
            ),
            
            FunctionalityTestCase(
                name="security_focused",
                prompt="Implement zero-trust security architecture with SIEM integration",
                expected_complexity=ComplexityLevel.COMPLEX,
                expected_coordination=CoordinationType.HIVE,
                expected_agents=["security-manager", "network-architect", "monitoring-specialist"],
                expected_tools=["mcp__zen__planner"],
                min_confidence=0.4,
                categories=["security", "architecture", "monitoring"],
                description="Security-focused task should prioritize security agents"
            )
        ]
        
    def run_functionality_test(self, test_case: FunctionalityTestCase) -> FunctionalityTestResult:
        """Run a single functionality test case."""
        print(f"  Testing: {test_case.name}")
        
        # Generate directive using ZenConsultant
        start_time = time.time()
        directive = self.zen_consultant.get_concise_directive(test_case.prompt)
        generation_time = (time.time() - start_time) * 1000
        
        # Analyze results
        discrepancies = []
        accuracy_scores = []
        
        # Test 1: Complexity analysis accuracy
        if "thinking_mode" in directive:
            # Extract complexity from thinking mode or infer from other indicators
            actual_complexity = self._infer_complexity_from_directive(directive, test_case.prompt)
            complexity_correct = actual_complexity == test_case.expected_complexity
            accuracy_scores.append(1.0 if complexity_correct else 0.0)
            
            if not complexity_correct:
                discrepancies.append(
                    f"Complexity mismatch: expected {test_case.expected_complexity.name}, "
                    f"got {actual_complexity.name if actual_complexity else 'UNKNOWN'}"
                )
        else:
            accuracy_scores.append(0.5)  # Partial credit if complexity not explicitly provided
            
        # Test 2: Coordination type accuracy
        hive_recommended = directive.get("hive", {}).get("recommended", False)
        swarm_recommended = directive.get("swarm", {}).get("recommended", False)
        
        if hive_recommended and test_case.expected_coordination == CoordinationType.HIVE:
            coordination_correct = True
        elif swarm_recommended and test_case.expected_coordination == CoordinationType.SWARM:
            coordination_correct = True
        else:
            coordination_correct = False
            
        accuracy_scores.append(1.0 if coordination_correct else 0.0)
        
        if not coordination_correct:
            actual_coordination = "HIVE" if hive_recommended else "SWARM" if swarm_recommended else "UNCLEAR"
            discrepancies.append(
                f"Coordination type mismatch: expected {test_case.expected_coordination.name}, "
                f"got {actual_coordination}"
            )
            
        # Test 3: Agent recommendation accuracy
        recommended_agents = directive.get("agents", [])
        agent_accuracy = self._calculate_agent_accuracy(
            recommended_agents, test_case.expected_agents, test_case.categories
        )
        accuracy_scores.append(agent_accuracy)
        
        if agent_accuracy < 0.7:
            discrepancies.append(
                f"Agent recommendations suboptimal: expected types {test_case.expected_agents}, "
                f"got {recommended_agents}"
            )
            
        # Test 4: Tool selection accuracy  
        recommended_tools = directive.get("tools", [])
        tool_accuracy = self._calculate_tool_accuracy(
            recommended_tools, test_case.expected_tools, test_case.expected_coordination
        )
        accuracy_scores.append(tool_accuracy)
        
        if tool_accuracy < 0.7:
            discrepancies.append(
                f"Tool selection suboptimal: expected {test_case.expected_tools}, "
                f"got {recommended_tools}"
            )
            
        # Test 5: Confidence appropriateness
        confidence = directive.get("confidence", 0.0)
        confidence_appropriate = confidence >= test_case.min_confidence
        accuracy_scores.append(1.0 if confidence_appropriate else 0.0)
        
        if not confidence_appropriate:
            discrepancies.append(
                f"Confidence too low: expected >= {test_case.min_confidence}, "
                f"got {confidence}"
            )
            
        # Test 6: Output format consistency
        format_score = self._validate_output_format(directive)
        accuracy_scores.append(format_score)
        
        if format_score < 1.0:
            discrepancies.append("Output format inconsistencies detected")
            
        # Calculate overall accuracy
        overall_accuracy = statistics.mean(accuracy_scores)
        passed = overall_accuracy >= 0.7 and len(discrepancies) <= 2
        
        # Generate recommendations
        recommendations = self._generate_test_recommendations(
            test_case, directive, discrepancies, overall_accuracy
        )
        
        result = FunctionalityTestResult(
            test_case=test_case.name,
            passed=passed,
            accuracy_score=overall_accuracy,
            details={
                "test_case": test_case.name,
                "prompt": test_case.prompt,
                "directive": directive,
                "generation_time_ms": generation_time,
                "complexity_analysis": {
                    "expected": test_case.expected_complexity.name,
                    "inferred": self._infer_complexity_from_directive(directive, test_case.prompt).name if self._infer_complexity_from_directive(directive, test_case.prompt) else "UNKNOWN"
                },
                "coordination_analysis": {
                    "expected": test_case.expected_coordination.name,
                    "hive_recommended": hive_recommended,
                    "swarm_recommended": swarm_recommended
                },
                "agent_analysis": {
                    "expected": test_case.expected_agents,
                    "recommended": recommended_agents,
                    "accuracy": agent_accuracy
                },
                "tool_analysis": {
                    "expected": test_case.expected_tools,
                    "recommended": recommended_tools,
                    "accuracy": tool_accuracy
                },
                "confidence_analysis": {
                    "expected_min": test_case.min_confidence,
                    "actual": confidence,
                    "appropriate": confidence_appropriate
                },
                "accuracy_breakdown": {
                    "complexity": accuracy_scores[0] if len(accuracy_scores) > 0 else 0,
                    "coordination": accuracy_scores[1] if len(accuracy_scores) > 1 else 0,
                    "agents": accuracy_scores[2] if len(accuracy_scores) > 2 else 0,
                    "tools": accuracy_scores[3] if len(accuracy_scores) > 3 else 0,
                    "confidence": accuracy_scores[4] if len(accuracy_scores) > 4 else 0,
                    "format": accuracy_scores[5] if len(accuracy_scores) > 5 else 0
                }
            },
            discrepancies=discrepancies,
            recommendations=recommendations
        )
        
        self.test_results.append(result)
        return result
        
    def _infer_complexity_from_directive(self, directive: Dict[str, Any], prompt: str) -> Optional[ComplexityLevel]:
        """Infer complexity level from directive and prompt characteristics."""
        # Check if hive is recommended (usually complex/enterprise)
        if directive.get("hive", {}).get("recommended", False):
            # Count complexity indicators
            enterprise_indicators = len([
                word for word in ["enterprise", "compliance", "audit", "governance", "multi-tenant"]
                if word in prompt.lower()
            ])
            
            if enterprise_indicators >= 2:
                return ComplexityLevel.ENTERPRISE
            else:
                return ComplexityLevel.COMPLEX
                
        # Check agent count and types
        agents = directive.get("agents", [])
        if len(agents) >= 3:
            return ComplexityLevel.COMPLEX
        elif len(agents) >= 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.SIMPLE
            
    def _calculate_agent_accuracy(self, recommended: List[str], expected: List[str], categories: List[str]) -> float:
        """Calculate accuracy of agent recommendations."""
        if not expected:  # No specific agents expected
            return 1.0 if not recommended else 0.8  # Slight penalty for over-recommendation
            
        if not recommended:
            return 0.0
            
        # Calculate semantic similarity between recommended and expected agents
        matches = 0
        for expected_agent in expected:
            for recommended_agent in recommended:
                if self._agents_semantically_similar(expected_agent, recommended_agent, categories):
                    matches += 1
                    break
                    
        accuracy = matches / max(len(expected), len(recommended))
        return min(accuracy, 1.0)
        
    def _agents_semantically_similar(self, agent1: str, agent2: str, categories: List[str]) -> bool:
        """Check if two agents are semantically similar for the given categories."""
        # Simple semantic similarity check
        agent1_lower = agent1.lower()
        agent2_lower = agent2.lower()
        
        # Direct match
        if agent1_lower == agent2_lower:
            return True
            
        # Category-based matching
        category_mappings = {
            "development": ["developer", "coder", "programmer"],
            "frontend": ["frontend", "ui", "web"],
            "backend": ["backend", "api", "server"],
            "testing": ["tester", "qa", "quality"],
            "architecture": ["architect", "designer"],
            "security": ["security", "auth", "crypto"],
            "devops": ["devops", "ops", "deployment"],
            "data": ["data", "database", "analytics"],
            "ml": ["ml", "ai", "machine-learning"]
        }
        
        for category in categories:
            if category in category_mappings:
                keywords = category_mappings[category]
                if any(keyword in agent1_lower for keyword in keywords) and \
                   any(keyword in agent2_lower for keyword in keywords):
                    return True
                    
        return False
        
    def _calculate_tool_accuracy(self, recommended: List[str], expected: List[str], coordination: CoordinationType) -> float:
        """Calculate accuracy of tool recommendations."""
        if not expected:
            # Check if tools are appropriate for coordination type
            if coordination == CoordinationType.HIVE:
                hive_tools = [tool for tool in recommended if "hive" in tool or "zen" in tool]
                return 1.0 if hive_tools else 0.5
            else:
                swarm_tools = [tool for tool in recommended if "swarm" in tool or "agent_spawn" in tool]
                return 1.0 if swarm_tools else 0.5
                
        if not recommended:
            return 0.0
            
        # Calculate overlap
        expected_set = set(expected)
        recommended_set = set(recommended)
        
        intersection = expected_set.intersection(recommended_set)
        union = expected_set.union(recommended_set)
        
        return len(intersection) / len(union) if union else 0.0
        
    def _validate_output_format(self, directive: Dict[str, Any]) -> float:
        """Validate output format consistency."""
        required_fields = ["hive", "swarm", "agents", "tools", "confidence", "session_id"]
        
        format_score = 0.0
        
        # Check required fields
        for field in required_fields:
            if field in directive:
                format_score += 1.0 / len(required_fields)
                
        # Check field types
        if isinstance(directive.get("hive"), dict):
            format_score += 0.1
        if isinstance(directive.get("swarm"), dict):
            format_score += 0.1
        if isinstance(directive.get("agents"), list):
            format_score += 0.1
        if isinstance(directive.get("tools"), list):
            format_score += 0.1
        if isinstance(directive.get("confidence"), (int, float)):
            format_score += 0.1
            
        return min(format_score, 1.0)
        
    def _generate_test_recommendations(self, test_case: FunctionalityTestCase, directive: Dict[str, Any], 
                                     discrepancies: List[str], accuracy: float) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if accuracy < 0.5:
            recommendations.append(f"Major functionality issues in {test_case.name} - review algorithm")
            
        if "Complexity mismatch" in str(discrepancies):
            recommendations.append("Improve complexity analysis algorithm")
            
        if "Coordination type mismatch" in str(discrepancies):
            recommendations.append("Review coordination type selection logic")
            
        if "Agent recommendations suboptimal" in str(discrepancies):
            recommendations.append("Enhance agent recommendation system")
            
        if "Tool selection suboptimal" in str(discrepancies):
            recommendations.append("Optimize tool selection based on task requirements")
            
        if "Confidence too low" in str(discrepancies):
            recommendations.append("Calibrate confidence scoring mechanism")
            
        if "Output format inconsistencies" in str(discrepancies):
            recommendations.append("Standardize output format validation")
            
        return recommendations
        
    def test_output_consistency(self, iterations: int = 10) -> Dict[str, Any]:
        """Test output consistency across multiple runs."""
        print("🔄 Testing Output Consistency...")
        
        test_prompt = "Implement user authentication with JWT tokens"
        results = []
        
        for _i in range(iterations):
            directive = self.zen_consultant.get_concise_directive(test_prompt)
            results.append(directive)
            
        # Analyze consistency
        consistency_metrics = {
            "confidence_stability": self._analyze_confidence_stability(results),
            "agent_consistency": self._analyze_agent_consistency(results),
            "tool_consistency": self._analyze_tool_consistency(results),
            "structure_consistency": self._analyze_structure_consistency(results)
        }
        
        overall_consistency = statistics.mean(consistency_metrics.values())
        
        return {
            "iterations": iterations,
            "consistency_metrics": consistency_metrics,
            "overall_consistency": overall_consistency,
            "consistency_rating": "excellent" if overall_consistency > 0.9 else "good" if overall_consistency > 0.7 else "needs_improvement"
        }
        
    def _analyze_confidence_stability(self, results: List[Dict[str, Any]]) -> float:
        """Analyze confidence score stability."""
        confidence_scores = [r.get("confidence", 0.0) for r in results]
        if not confidence_scores:
            return 0.0
            
        # Calculate coefficient of variation
        mean_confidence = statistics.mean(confidence_scores)
        if mean_confidence == 0:
            return 0.0
            
        std_dev = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
        cv = std_dev / mean_confidence
        
        # Convert to stability score (lower CV = higher stability)
        stability = max(0.0, 1.0 - cv)
        return stability
        
    def _analyze_agent_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Analyze agent recommendation consistency."""
        agent_sets = [set(r.get("agents", [])) for r in results]
        if not agent_sets:
            return 0.0
            
        # Calculate Jaccard similarity between consecutive results
        similarities = []
        for i in range(1, len(agent_sets)):
            intersection = agent_sets[i-1].intersection(agent_sets[i])
            union = agent_sets[i-1].union(agent_sets[i])
            similarity = len(intersection) / len(union) if union else 1.0
            similarities.append(similarity)
            
        return statistics.mean(similarities) if similarities else 1.0
        
    def _analyze_tool_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Analyze tool recommendation consistency."""
        tool_sets = [set(r.get("tools", [])) for r in results]
        if not tool_sets:
            return 0.0
            
        # Calculate consistency similar to agents
        similarities = []
        for i in range(1, len(tool_sets)):
            intersection = tool_sets[i-1].intersection(tool_sets[i])
            union = tool_sets[i-1].union(tool_sets[i])
            similarity = len(intersection) / len(union) if union else 1.0
            similarities.append(similarity)
            
        return statistics.mean(similarities) if similarities else 1.0
        
    def _analyze_structure_consistency(self, results: List[Dict[str, Any]]) -> float:
        """Analyze output structure consistency."""
        required_fields = ["hive", "swarm", "agents", "tools", "confidence"]
        
        structure_scores = []
        for result in results:
            score = sum(1 for field in required_fields if field in result) / len(required_fields)
            structure_scores.append(score)
            
        return statistics.mean(structure_scores) if structure_scores else 0.0
        
    def run_comprehensive_functionality_tests(self) -> Dict[str, Any]:
        """Run all functionality tests and generate comprehensive report."""
        print("🧪 Running Comprehensive Functionality Tests...")
        print("=" * 50)
        
        # Run all test cases
        for test_case in self.test_cases:
            self.run_functionality_test(test_case)
            
        # Run consistency tests
        consistency_results = self.test_output_consistency(10)
        
        # Calculate overall metrics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        self.functionality_score = statistics.mean([r.accuracy_score for r in self.test_results]) * 100
        
        # Categorize results
        by_complexity = {}
        for result in self.test_results:
            test_case = next(tc for tc in self.test_cases if tc.name == result.test_case)
            complexity = test_case.expected_complexity.name
            if complexity not in by_complexity:
                by_complexity[complexity] = []
            by_complexity[complexity].append(result)
            
        complexity_scores = {
            complexity: statistics.mean([r.accuracy_score for r in results]) * 100
            for complexity, results in by_complexity.items()
        }
        
        # Generate all recommendations
        all_recommendations = []
        for result in self.test_results:
            all_recommendations.extend(result.recommendations)
            
        unique_recommendations = list(set(all_recommendations))
        
        return {
            "timestamp": time.time(),
            "functionality_score": self.functionality_score,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "pass_rate": passed_tests / total_tests,
            "complexity_breakdown": complexity_scores,
            "consistency_results": consistency_results,
            "test_results": [
                {
                    "test_case": r.test_case,
                    "passed": r.passed,
                    "accuracy_score": r.accuracy_score,
                    "discrepancies": r.discrepancies,
                    "recommendations": r.recommendations
                } for r in self.test_results
            ],
            "overall_recommendations": unique_recommendations,
            "functionality_status": self._determine_functionality_status()
        }
        
    def _determine_functionality_status(self) -> str:
        """Determine overall functionality status."""
        if self.functionality_score >= 90:
            return "EXCELLENT - ZenConsultant functioning optimally"
        elif self.functionality_score >= 80:
            return "GOOD - Minor improvements needed"
        elif self.functionality_score >= 70:
            return "ACCEPTABLE - Some functionality issues present"
        elif self.functionality_score >= 50:
            return "NEEDS IMPROVEMENT - Significant functionality gaps"
        else:
            return "POOR - Major functionality overhaul required"


def run_functionality_test_suite():
    """Run complete functionality test suite and save results."""
    print("🎯 ZEN Co-pilot System - Functionality Testing Framework")
    print("=" * 60)
    
    tester = ZenFunctionalityTester()
    
    # Run comprehensive functionality tests
    report = tester.run_comprehensive_functionality_tests()
    
    # Save report
    report_path = Path("/home/devcontainers/flowed/.claude/hooks/functionality_test_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    # Print summary
    print("\n🎯 FUNCTIONALITY TEST RESULTS SUMMARY")
    print("-" * 40)
    print(f"📊 Functionality Score: {report['functionality_score']:.1f}/100")
    print(f"✅ Tests Passed: {report['passed_tests']}/{report['total_tests']} ({report['pass_rate']:.1%})")
    print(f"🏆 Status: {report['functionality_status']}")
    
    print("\n📊 COMPLEXITY BREAKDOWN")
    print("-" * 25)
    for complexity, score in report['complexity_breakdown'].items():
        print(f"• {complexity}: {score:.1f}/100")
        
    print("\n🔄 CONSISTENCY ANALYSIS")
    print("-" * 25)
    consistency = report['consistency_results']
    print(f"• Overall Consistency: {consistency['overall_consistency']:.1%}")
    print(f"• Rating: {consistency['consistency_rating']}")
    
    print(f"\n📋 Full report saved to: {report_path}")
    
    # Print top recommendations
    if report['overall_recommendations']:
        print("\n🎯 TOP RECOMMENDATIONS")
        print("-" * 25)
        for rec in report['overall_recommendations'][:5]:
            print(f"• {rec}")
            
    return report


if __name__ == "__main__":
    run_functionality_test_suite()