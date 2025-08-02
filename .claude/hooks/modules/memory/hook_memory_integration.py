"""Hook Memory Integration - Integrates memory operations into Claude hook lifecycle"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from .project_memory_manager import get_memory_manager

class HookMemoryIntegration:
    """Integrates memory operations throughout the hook lifecycle"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_memories = []
        
    async def capture_pre_tool_memory(self, tool_name: str, args: Dict[str, Any]) -> None:
        """Capture memory before tool execution"""
        # Analyze semantic significance
        analysis = self.memory_manager.analyze_semantic_context(
            {"tool": tool_name, "args": args},
            context_type="pre_tool"
        )
        
        if analysis["should_store"]:
            memory_key = f"pre_tool/{self.session_id}/{tool_name}_{datetime.now().timestamp()}"
            await self.memory_manager.store_memory(
                key=memory_key,
                value={
                    "tool": tool_name,
                    "args": args,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id
                },
                category="pre_tool",
                metadata=analysis
            )
            
            self.session_memories.append({
                "phase": "pre_tool",
                "tool": tool_name,
                "key": memory_key
            })
    
    async def capture_post_tool_memory(self, tool_name: str, result: Any, 
                                     execution_time: float) -> None:
        """Capture memory after tool execution"""
        # Determine if result is significant
        analysis = self.memory_manager.analyze_semantic_context(
            {"tool": tool_name, "result": str(result)[:500]},  # Limit result size
            context_type="post_tool"
        )
        
        # Always store errors or significant results
        if "error" in str(result).lower() or analysis["should_store"]:
            memory_key = f"post_tool/{self.session_id}/{tool_name}_{datetime.now().timestamp()}"
            await self.memory_manager.store_memory(
                key=memory_key,
                value={
                    "tool": tool_name,
                    "result_summary": self._summarize_result(result),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id,
                    "success": "error" not in str(result).lower()
                },
                category="post_tool",
                metadata=analysis
            )
            
            self.session_memories.append({
                "phase": "post_tool",
                "tool": tool_name,
                "key": memory_key
            })
    
    async def capture_user_prompt_memory(self, prompt: str) -> None:
        """Capture user prompt for context"""
        analysis = self.memory_manager.analyze_semantic_context(
            prompt,
            context_type="user_prompt"
        )
        
        if analysis["should_store"]:
            memory_key = f"prompts/{self.session_id}/prompt_{datetime.now().timestamp()}"
            await self.memory_manager.store_memory(
                key=memory_key,
                value={
                    "prompt": prompt,
                    "timestamp": datetime.now().isoformat(),
                    "session_id": self.session_id,
                    "categories": analysis["categories"]
                },
                category="prompts",
                metadata=analysis
            )
    
    async def capture_session_start_memory(self, context: Dict[str, Any]) -> None:
        """Capture session start context"""
        memory_key = f"sessions/{self.session_id}/start"
        await self.memory_manager.store_memory(
            key=memory_key,
            value={
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "context": context,
                "project_namespace": self.memory_manager.namespace
            },
            category="sessions",
            ttl=2592000  # 30 days
        )
        
        # Load relevant past memories
        await self._load_relevant_memories()
    
    async def capture_session_end_memory(self, summary: Dict[str, Any]) -> None:
        """Capture session end summary"""
        memory_key = f"sessions/{self.session_id}/end"
        await self.memory_manager.store_memory(
            key=memory_key,
            value={
                "session_id": self.session_id,
                "end_time": datetime.now().isoformat(),
                "summary": summary,
                "memories_created": len(self.session_memories),
                "session_memories": self.session_memories
            },
            category="sessions",
            ttl=2592000  # 30 days
        )
        
        # Sync with claude-flow
        await self.memory_manager.sync_with_claude_flow()
    
    async def _load_relevant_memories(self) -> List[Dict]:
        """Load relevant memories from previous sessions"""
        relevant_memories = []
        
        # Search for recent error patterns
        errors = await self.memory_manager.search_memories("error", category="errors")
        if errors:
            relevant_memories.extend(errors[-5:])  # Last 5 errors
        
        # Search for recent architectural decisions
        architecture = await self.memory_manager.search_memories("architecture", category="architecture")
        if architecture:
            relevant_memories.extend(architecture[-3:])  # Last 3 decisions
        
        # Search for optimization patterns
        optimizations = await self.memory_manager.search_memories("optimization", category="optimization")
        if optimizations:
            relevant_memories.extend(optimizations[-3:])  # Last 3 optimizations
        
        return relevant_memories
    
    def _summarize_result(self, result: Any) -> Dict[str, Any]:
        """Create a summary of the result for memory storage"""
        result_str = str(result)
        
        return {
            "type": type(result).__name__,
            "length": len(result_str),
            "preview": result_str[:200] + "..." if len(result_str) > 200 else result_str,
            "has_error": "error" in result_str.lower(),
            "has_warning": "warning" in result_str.lower()
        }
    
    async def get_context_memories(self, context: str) -> List[Dict]:
        """Get memories relevant to current context"""
        # Search across all categories
        memories = await self.memory_manager.search_memories(context)
        
        # Sort by relevance and recency
        sorted_memories = sorted(
            memories,
            key=lambda m: (
                m.get("metadata", {}).get("relevance_score", 0),
                m.get("timestamp", "")
            ),
            reverse=True
        )
        
        return sorted_memories[:10]  # Top 10 most relevant
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        return {
            "session_id": self.session_id,
            "memories_created": len(self.session_memories),
            "memory_categories": self._categorize_memories(),
            "namespace": self.memory_manager.namespace
        }
    
    def _categorize_memories(self) -> Dict[str, int]:
        """Categorize session memories by type"""
        categories = {}
        for memory in self.session_memories:
            phase = memory.get("phase", "unknown")
            categories[phase] = categories.get(phase, 0) + 1
        return categories

# Singleton instance
_hook_memory_integration = None

def get_hook_memory_integration() -> HookMemoryIntegration:
    """Get singleton hook memory integration instance"""
    global _hook_memory_integration
    if _hook_memory_integration is None:
        _hook_memory_integration = HookMemoryIntegration()
    return _hook_memory_integration