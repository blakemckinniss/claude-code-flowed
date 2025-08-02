#!/usr/bin/env python3
"""ZEN Memory Integration for Learning and Pattern Recognition.

Manages memory operations in the zen-copilot namespace for improved
directive generation through learning from successful patterns.
"""

import json
import asyncio
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class ZenMemoryEntry:
    """Memory entry for ZEN learning system."""
    prompt: str
    directive: Dict[str, Any]
    success: bool
    complexity: str
    categories: List[str]
    timestamp: datetime
    feedback_score: float = 0.0


@dataclass
class ConversationThread:
    """Conversation thread with UUID-based threading and 3-hour expiry."""
    thread_id: str
    created_at: datetime
    last_accessed: datetime
    entries: List[ZenMemoryEntry]
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Initialize thread with UUID if not provided."""
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())
    
    def add_entry(self, entry: ZenMemoryEntry) -> None:
        """Add memory entry to thread and update access time."""
        self.entries.append(entry)
        self.last_accessed = datetime.now()
    
    def get_context_window(self, max_tokens: int = 4000) -> List[ZenMemoryEntry]:
        """Get recent entries within token limit for context window."""
        # Simple approximation: ~50 tokens per entry on average
        # More sophisticated token counting could be added later
        max_entries = max_tokens // 50
        return self.entries[-max_entries:] if len(self.entries) > max_entries else self.entries
    
    def is_expired(self, timeout_hours: int = 3) -> bool:
        """Check if thread has expired based on last access time."""
        expiry_time = self.last_accessed + timedelta(hours=timeout_hours)
        return datetime.now() > expiry_time
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update workflow state metadata."""
        self.metadata[key] = value
        self.last_accessed = datetime.now()


class ZenMemoryManager:
    """Manages memory operations for ZEN learning system."""
    
    def __init__(self):
        """Initialize ZEN memory manager."""
        self.namespace = "zen-copilot"
        self.memory_store = {}  # In-memory store for prototype
        self.conversation_threads = {}  # Dict[str, ConversationThread] - New threading support
        self.current_thread_id = None  # Track active conversation thread
        self.learning_patterns = {
            "successful_directives": {},
            "failed_patterns": {},
            "complexity_mappings": {},
            "category_preferences": {},
            "agent_success_rates": {},
            "tool_effectiveness": {}
        }
        
    async def store_directive_outcome(self, 
                                    prompt: str, 
                                    directive: Dict[str, Any],
                                    success: bool,
                                    feedback_score: float = 0.0) -> None:
        """Store the outcome of a directive for learning."""
        entry = ZenMemoryEntry(
            prompt=prompt,
            directive=directive,
            success=success,
            complexity=directive.get("thinking_mode", "medium"),
            categories=self._extract_categories_from_prompt(prompt),
            timestamp=datetime.now(),
            feedback_score=feedback_score
        )
        
        # Store in memory
        key = f"directive_{datetime.now().timestamp()}"
        self.memory_store[key] = entry
        
        # Update learning patterns
        await self._update_learning_patterns(entry)
        
        # Add to current conversation thread if active
        if self.current_thread_id and self.current_thread_id in self.conversation_threads:
            self.conversation_threads[self.current_thread_id].add_entry(entry)
        
    async def _update_learning_patterns(self, entry: ZenMemoryEntry) -> None:
        """Update learning patterns based on outcome."""
        if entry.success:
            # Update successful patterns
            for category in entry.categories:
                if category not in self.learning_patterns["successful_directives"]:
                    self.learning_patterns["successful_directives"][category] = []
                
                self.learning_patterns["successful_directives"][category].append({
                    "agents": entry.directive.get("agents", []),
                    "tools": entry.directive.get("tools", []),
                    "coordination": entry.directive.get("hive", "SWARM"),
                    "success_score": entry.feedback_score,
                    "timestamp": entry.timestamp.isoformat()
                })
                
            # Update agent success rates
            for agent in entry.directive.get("agents", []):
                if agent not in self.learning_patterns["agent_success_rates"]:
                    self.learning_patterns["agent_success_rates"][agent] = {"successes": 0, "total": 0}
                
                self.learning_patterns["agent_success_rates"][agent]["successes"] += 1
                self.learning_patterns["agent_success_rates"][agent]["total"] += 1
                
        else:
            # Update failed patterns
            pattern_key = f"{entry.complexity}_{len(entry.categories)}"
            if pattern_key not in self.learning_patterns["failed_patterns"]:
                self.learning_patterns["failed_patterns"][pattern_key] = []
                
            self.learning_patterns["failed_patterns"][pattern_key].append({
                "prompt_categories": entry.categories,
                "directive": entry.directive,
                "timestamp": entry.timestamp.isoformat()
            })
            
            # Update agent failure rates
            for agent in entry.directive.get("agents", []):
                if agent not in self.learning_patterns["agent_success_rates"]:
                    self.learning_patterns["agent_success_rates"][agent] = {"successes": 0, "total": 0}
                
                self.learning_patterns["agent_success_rates"][agent]["total"] += 1
                
    def _extract_categories_from_prompt(self, prompt: str) -> List[str]:
        """Extract task categories from prompt for learning."""
        prompt_lower = prompt.lower()
        categories = []
        
        category_keywords = {
            "development": ["code", "implement", "build", "create", "develop"],
            "testing": ["test", "qa", "quality", "verify"],
            "debugging": ["debug", "fix", "error", "issue", "problem", "bug"],
            "architecture": ["architecture", "design", "system", "structure"],
            "refactoring": ["refactor", "clean", "improve", "optimize"],
            "security": ["security", "audit", "vulnerability", "secure"],
            "performance": ["performance", "speed", "optimize", "efficient"],
            "documentation": ["document", "docs", "readme", "guide"],
            "github": ["github", "pr", "pull request", "issue", "commit"],
            "deployment": ["deploy", "release", "production", "ci/cd"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                categories.append(category)
                
        return categories or ["general"]
    
    # === Conversation Threading Methods ===
    
    def create_conversation_thread(self, thread_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new conversation thread with UUID-based ID."""
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        now = datetime.now()
        thread = ConversationThread(
            thread_id=thread_id,
            created_at=now,
            last_accessed=now,
            entries=[],
            metadata=metadata or {}
        )
        
        self.conversation_threads[thread_id] = thread
        self.current_thread_id = thread_id
        return thread_id
    
    def set_active_thread(self, thread_id: str) -> bool:
        """Set the active conversation thread."""
        if thread_id in self.conversation_threads:
            self.current_thread_id = thread_id
            self.conversation_threads[thread_id].last_accessed = datetime.now()
            return True
        return False
    
    def get_current_thread(self) -> Optional[ConversationThread]:
        """Get the current active conversation thread."""
        if self.current_thread_id and self.current_thread_id in self.conversation_threads:
            return self.conversation_threads[self.current_thread_id]
        return None
    
    def get_thread_context_window(self, thread_id: Optional[str] = None, max_tokens: int = 4000) -> List[ZenMemoryEntry]:
        """Get context window for a specific thread or current thread."""
        target_thread_id = thread_id or self.current_thread_id
        if target_thread_id and target_thread_id in self.conversation_threads:
            return self.conversation_threads[target_thread_id].get_context_window(max_tokens)
        return []
    
    def expire_old_threads(self, timeout_hours: int = 3) -> int:
        """Remove expired conversation threads. Returns count of expired threads."""
        expired_threads = []
        
        for thread_id, thread in self.conversation_threads.items():
            if thread.is_expired(timeout_hours):
                expired_threads.append(thread_id)
        
        for thread_id in expired_threads:
            # Move entries to main memory store before deletion for learning
            thread = self.conversation_threads[thread_id]
            for entry in thread.entries:
                key = f"expired_thread_{thread_id}_{entry.timestamp.timestamp()}"
                if key not in self.memory_store:  # Avoid duplicates
                    self.memory_store[key] = entry
            
            del self.conversation_threads[thread_id]
            
            # Clear current thread if it was expired
            if self.current_thread_id == thread_id:
                self.current_thread_id = None
        
        return len(expired_threads)
    
    def get_thread_stats(self) -> Dict[str, Any]:
        """Get statistics about conversation threads."""
        active_threads = len(self.conversation_threads)
        total_entries = sum(len(thread.entries) for thread in self.conversation_threads.values())
        
        oldest_thread = None
        newest_thread = None
        
        if self.conversation_threads:
            threads_by_age = sorted(self.conversation_threads.values(), key=lambda t: t.created_at)
            oldest_thread = threads_by_age[0].created_at.isoformat()
            newest_thread = threads_by_age[-1].created_at.isoformat()
        
        return {
            "active_threads": active_threads,
            "current_thread_id": self.current_thread_id,
            "total_thread_entries": total_entries,
            "oldest_thread_created": oldest_thread,
            "newest_thread_created": newest_thread
        }
        
    async def get_recommendations_for_prompt(self, prompt: str, complexity: str) -> Dict[str, Any]:
        """Get recommendations based on learned patterns."""
        categories = self._extract_categories_from_prompt(prompt)
        recommendations = {
            "suggested_agents": [],
            "suggested_tools": [],
            "suggested_coordination": "SWARM",
            "confidence_boost": 0.0
        }
        
        # Analyze successful patterns for similar categories
        for category in categories:
            if category in self.learning_patterns["successful_directives"]:
                patterns = self.learning_patterns["successful_directives"][category]
                
                # Get most successful recent patterns
                recent_patterns = [p for p in patterns 
                                 if datetime.fromisoformat(p["timestamp"]) > datetime.now() - timedelta(days=30)]
                
                if recent_patterns:
                    # Sort by success score
                    best_pattern = max(recent_patterns, key=lambda x: x.get("success_score", 0))
                    
                    recommendations["suggested_agents"].extend(best_pattern["agents"])
                    recommendations["suggested_tools"].extend(best_pattern["tools"])
                    recommendations["suggested_coordination"] = best_pattern["coordination"]
                    recommendations["confidence_boost"] += 0.1
                    
        # Analyze agent success rates
        agent_scores = {}
        for agent, stats in self.learning_patterns["agent_success_rates"].items():
            if stats["total"] > 0:
                success_rate = stats["successes"] / stats["total"]
                agent_scores[agent] = success_rate
                
        # Recommend top-performing agents
        if agent_scores:
            top_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            for agent, score in top_agents:
                if score > 0.7 and agent not in recommendations["suggested_agents"]:
                    recommendations["suggested_agents"].append(agent)
                    
        # Remove duplicates and limit
        recommendations["suggested_agents"] = list(set(recommendations["suggested_agents"]))[:3]
        recommendations["suggested_tools"] = list(set(recommendations["suggested_tools"]))[:3]
        
        return recommendations
        
    async def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress."""
        total_directives = len(self.memory_store)
        successful_directives = sum(1 for entry in self.memory_store.values() if entry.success)
        thread_stats = self.get_thread_stats()
        
        return {
            "total_directives": total_directives,
            "successful_directives": successful_directives,
            "success_rate": successful_directives / total_directives if total_directives > 0 else 0.0,
            "categories_learned": len(self.learning_patterns["successful_directives"]),
            "agents_tracked": len(self.learning_patterns["agent_success_rates"]),
            "learning_patterns_count": sum(len(patterns) for patterns in self.learning_patterns["successful_directives"].values()),
            "namespace": self.namespace,
            "threading": thread_stats
        }
        
    async def export_learning_data(self) -> Dict[str, Any]:
        """Export learning data for backup or analysis."""
        # Convert conversation threads to serializable format
        serializable_threads = {}
        for thread_id, thread in self.conversation_threads.items():
            serializable_threads[thread_id] = {
                "thread_id": thread.thread_id,
                "created_at": thread.created_at.isoformat(),
                "last_accessed": thread.last_accessed.isoformat(),
                "metadata": thread.metadata,
                "entries_count": len(thread.entries)
            }
        
        return {
            "namespace": self.namespace,
            "export_timestamp": datetime.now().isoformat(),
            "learning_patterns": self.learning_patterns,
            "memory_entries_count": len(self.memory_store),
            "conversation_threads": serializable_threads,
            "current_thread_id": self.current_thread_id,
            "stats": await self.get_learning_stats()
        }
        
    async def import_learning_data(self, data: Dict[str, Any]) -> bool:
        """Import learning data from backup."""
        try:
            if data.get("namespace") == self.namespace:
                self.learning_patterns.update(data.get("learning_patterns", {}))
                
                # Import conversation threads (metadata only for this implementation)
                thread_data = data.get("conversation_threads", {})
                for thread_id, thread_info in thread_data.items():
                    # Recreate threads with metadata but empty entries
                    # (Full entries would need more complex serialization)
                    thread = ConversationThread(
                        thread_id=thread_id,
                        created_at=datetime.fromisoformat(thread_info["created_at"]),
                        last_accessed=datetime.fromisoformat(thread_info["last_accessed"]),
                        entries=[],  # Empty for this implementation
                        metadata=thread_info.get("metadata", {})
                    )
                    self.conversation_threads[thread_id] = thread
                
                # Restore current thread if it exists in imported data
                imported_current = data.get("current_thread_id")
                if imported_current and imported_current in self.conversation_threads:
                    self.current_thread_id = imported_current
                
                return True
            return False
        except Exception:
            return False
            
    def get_pattern_summary(self) -> str:
        """Get a summary of learned patterns for debugging."""
        stats = asyncio.run(self.get_learning_stats())
        threading_stats = stats.get('threading', {})
        
        summary = f"""
🧠 ZEN Learning Summary:
• Total Directives: {stats['total_directives']}
• Success Rate: {stats['success_rate']:.1%}
• Categories Learned: {stats['categories_learned']}
• Agents Tracked: {stats['agents_tracked']}
• Learning Patterns: {stats['learning_patterns_count']}

🧵 Conversation Threading:
• Active Threads: {threading_stats.get('active_threads', 0)}
• Current Thread: {threading_stats.get('current_thread_id', 'None')[:8] + '...' if threading_stats.get('current_thread_id') else 'None'}
• Thread Entries: {threading_stats.get('total_thread_entries', 0)}

Top Performing Agents:
"""
        
        # Add top agent performance
        agent_stats = self.learning_patterns["agent_success_rates"]
        if agent_stats:
            sorted_agents = sorted(
                [(agent, stats["successes"] / stats["total"] if stats["total"] > 0 else 0) 
                 for agent, stats in agent_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for agent, success_rate in sorted_agents:
                summary += f"• {agent}: {success_rate:.1%}\n"
                
        return summary


# Singleton instance for prototype
_zen_memory_manager = None

def get_zen_memory_manager() -> ZenMemoryManager:
    """Get singleton ZEN memory manager."""
    global _zen_memory_manager
    if _zen_memory_manager is None:
        _zen_memory_manager = ZenMemoryManager()
    return _zen_memory_manager


# Simulation functions for testing
async def simulate_learning_session():
    """Simulate a learning session with various outcomes."""
    manager = get_zen_memory_manager()
    
    # Simulate successful development task
    await manager.store_directive_outcome(
        prompt="Build user authentication system",
        directive={
            "hive": "SWARM",
            "agents": ["coder", "security-auditor"],
            "tools": ["mcp__zen__analyze", "mcp__zen__testgen"],
            "thinking_mode": "medium"
        },
        success=True,
        feedback_score=0.9
    )
    
    # Simulate failed complex task
    await manager.store_directive_outcome(
        prompt="Refactor entire enterprise architecture with microservices",
        directive={
            "hive": "HIVE",
            "agents": ["system-architect"],  # Too few agents
            "tools": ["mcp__zen__analyze"],
            "thinking_mode": "max"
        },
        success=False,
        feedback_score=0.3
    )
    
    # Simulate successful debugging task
    await manager.store_directive_outcome(
        prompt="Fix memory leak in payment processor",
        directive={
            "hive": "SWARM",
            "agents": ["debugger", "performance-optimizer"],
            "tools": ["mcp__zen__debug", "mcp__zen__analyze"],
            "thinking_mode": "high"
        },
        success=True,
        feedback_score=0.8
    )
    
    return await manager.get_learning_stats()


async def simulate_conversation_threading():
    """Simulate conversation threading with multiple threads and expiry."""
    manager = get_zen_memory_manager()
    
    print("🧵 Testing Conversation Threading...")
    
    # Create first conversation thread for authentication work
    auth_thread_id = manager.create_conversation_thread(
        metadata={"project": "user-auth", "sprint": "2024-Q1"}
    )
    print(f"Created auth thread: {auth_thread_id[:8]}...")
    
    # Add some entries to auth thread
    await manager.store_directive_outcome(
        prompt="Design user authentication system",
        directive={
            "hive": "SWARM",
            "agents": ["security-architect", "backend-developer"],
            "tools": ["mcp__zen__analyze"],
            "thinking_mode": "high"
        },
        success=True,
        feedback_score=0.9
    )
    
    await manager.store_directive_outcome(
        prompt="Implement JWT token validation",
        directive={
            "hive": "SWARM", 
            "agents": ["security-auditor", "backend-developer"],
            "tools": ["mcp__zen__testgen"],
            "thinking_mode": "medium"
        },
        success=True,
        feedback_score=0.8
    )
    
    # Create second thread for different project
    api_thread_id = manager.create_conversation_thread(
        metadata={"project": "api-gateway", "sprint": "2024-Q1"}
    )
    print(f"Created API thread: {api_thread_id[:8]}...")
    
    # Add entry to API thread
    await manager.store_directive_outcome(
        prompt="Build REST API endpoints",
        directive={
            "hive": "SWARM",
            "agents": ["api-developer", "documentation-writer"],
            "tools": ["mcp__zen__planner"],
            "thinking_mode": "medium"
        },
        success=True,
        feedback_score=0.7
    )
    
    # Switch back to auth thread and add more entries
    manager.set_active_thread(auth_thread_id)
    await manager.store_directive_outcome(
        prompt="Add OAuth2 integration",
        directive={
            "hive": "HIVE",
            "agents": ["security-architect", "integration-specialist"],
            "tools": ["mcp__zen__analyze", "mcp__zen__debug"],
            "thinking_mode": "high"
        },
        success=True,
        feedback_score=0.9
    )
    
    # Test context window retrieval
    auth_context = manager.get_thread_context_window(auth_thread_id, max_tokens=2000)
    api_context = manager.get_thread_context_window(api_thread_id, max_tokens=2000)
    
    print(f"Auth thread context: {len(auth_context)} entries")
    print(f"API thread context: {len(api_context)} entries")
    
    # Test thread expiry (simulate by manipulating timestamps)
    import time
    old_thread = manager.conversation_threads[api_thread_id]
    old_thread.last_accessed = datetime.now() - timedelta(hours=4)  # Make it expired
    
    expired_count = manager.expire_old_threads(timeout_hours=3)
    print(f"Expired {expired_count} old threads")
    
    return {
        "auth_thread_id": auth_thread_id,
        "api_thread_id": api_thread_id,
        "auth_context_entries": len(auth_context),
        "api_context_entries": len(api_context),
        "expired_threads": expired_count,
        "final_stats": await manager.get_learning_stats()
    }


if __name__ == "__main__":
    # Run simulations
    print("🧠 ZEN Memory Integration Simulation")
    print("=" * 40)
    
    # Original learning simulation
    print("\n1️⃣ Basic Learning Simulation:")
    stats = asyncio.run(simulate_learning_session())
    manager = get_zen_memory_manager()
    
    print(manager.get_pattern_summary())
    
    # New conversation threading simulation
    print("\n2️⃣ Conversation Threading Simulation:")
    threading_results = asyncio.run(simulate_conversation_threading())
    
    print("\n🧵 Threading Results:")
    print(f"• Auth Thread: {threading_results['auth_thread_id'][:8]}... ({threading_results['auth_context_entries']} entries)")
    if threading_results['expired_threads'] == 0:
        print(f"• API Thread: {threading_results['api_thread_id'][:8]}... ({threading_results['api_context_entries']} entries)")
    print(f"• Expired Threads: {threading_results['expired_threads']}")
    
    final_stats = threading_results['final_stats']
    print(f"\n📊 Final Threading Stats:")
    print(f"• Active Threads: {final_stats['threading']['active_threads']}")
    print(f"• Thread Entries: {final_stats['threading']['total_thread_entries']}")
    print(f"• Total Directives: {final_stats['total_directives']}")
    print(f"• Overall Success Rate: {final_stats['success_rate']:.1%}")
    
    # Export demonstration
    export_data = asyncio.run(manager.export_learning_data())
    print(f"\n💾 Export Data Summary:")
    print(f"• Namespace: {export_data['namespace']}")
    print(f"• Memory Entries: {export_data['memory_entries_count']}")
    print(f"• Conversation Threads: {len(export_data['conversation_threads'])}")
    print(f"• Export Size: ~{len(json.dumps(export_data))} bytes")