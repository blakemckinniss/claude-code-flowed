"""
Memory Retrieval Patterns - Intelligent memory retrieval for hooks
Provides context-aware memory retrieval based on current operations
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import re
from pathlib import Path

from .project_memory_manager import get_memory_manager

class MemoryRetrievalPatterns:
    """Provides intelligent memory retrieval patterns for hooks"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.retrieval_cache = {}
        
    async def get_contextual_memories(self, tool_name: str, tool_input: Dict[str, Any], 
                                    max_results: int = 10) -> List[Dict[str, Any]]:
        """Get memories relevant to current tool context"""
        memories = []
        
        # Extract context from tool input
        context_keywords = self._extract_context_keywords(tool_name, tool_input)
        
        # Search for relevant memories
        for keyword in context_keywords[:5]:  # Limit to top 5 keywords
            results = await self.memory_manager.search_memories(keyword)
            memories.extend(results)
        
        # Deduplicate and rank by relevance
        unique_memories = self._deduplicate_memories(memories)
        ranked_memories = self._rank_memories(unique_memories, context_keywords)
        
        return ranked_memories[:max_results]
    
    async def get_error_prevention_memories(self, tool_name: str, 
                                          tool_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get memories of past errors to prevent repeating them"""
        error_memories = []
        
        # Search for error patterns related to this tool
        error_results = await self.memory_manager.search_memories(
            f"error {tool_name}", category="errors"
        )
        
        # Filter for relevant errors
        for memory in error_results:
            if self._is_relevant_error(memory, tool_name, tool_input):
                error_memories.append(memory)
        
        return error_memories
    
    async def get_optimization_patterns(self, operation_type: str) -> List[Dict[str, Any]]:
        """Get optimization patterns for specific operations"""
        # Search for optimization patterns
        optimizations = await self.memory_manager.search_memories(
            f"optimization {operation_type}", category="optimization"
        )
        
        # Filter for successful optimizations
        successful_optimizations = [
            opt for opt in optimizations 
            if opt.get("value", {}).get("success", False)
        ]
        
        return successful_optimizations
    
    async def get_architectural_decisions(self, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get relevant architectural decisions"""
        if component:
            query = f"architecture {component}"
        else:
            query = "architecture"
        
        decisions = await self.memory_manager.search_memories(
            query, category="architecture"
        )
        
        # Sort by timestamp (newest first)
        sorted_decisions = sorted(
            decisions,
            key=lambda d: d.get("value", {}).get("timestamp", ""),
            reverse=True
        )
        
        return sorted_decisions
    
    async def get_similar_tasks(self, task_description: str) -> List[Dict[str, Any]]:
        """Find memories of similar tasks"""
        # Extract key terms from task description
        key_terms = self._extract_key_terms(task_description)
        
        similar_tasks = []
        for term in key_terms:
            results = await self.memory_manager.search_memories(
                term, category="tasks"
            )
            similar_tasks.extend(results)
        
        # Rank by similarity
        ranked_tasks = self._rank_by_similarity(similar_tasks, task_description)
        
        return ranked_tasks[:5]
    
    async def get_recent_session_context(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get context from recent sessions"""
        # Search for recent session memories
        sessions = await self.memory_manager.search_memories(
            "session", category="sessions"
        )
        
        # Filter by time
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_sessions = []
        
        for session in sessions:
            timestamp_str = session.get("value", {}).get("timestamp", "")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if timestamp > cutoff_time:
                        recent_sessions.append(session)
                except Exception:
                    pass
        
        return recent_sessions
    
    def _extract_context_keywords(self, tool_name: str, tool_input: Dict[str, Any]) -> List[str]:
        """Extract relevant keywords from tool context"""
        keywords = [tool_name.lower()]
        
        # Extract from common tool input patterns
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            file_path = tool_input.get("file_path", "")
            if file_path:
                path_parts = Path(file_path).parts
                keywords.extend([p.lower() for p in path_parts[-3:]])
        
        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            # Extract command name
            if command:
                cmd_parts = command.split()
                if cmd_parts:
                    keywords.append(cmd_parts[0].lower())
        
        elif tool_name == "Task":
            subagent = tool_input.get("subagent_type", "")
            if subagent:
                keywords.append(subagent.lower())
        
        # Extract from description if available
        description = tool_input.get("description", "")
        if description:
            key_terms = self._extract_key_terms(description)
            keywords.extend(key_terms[:3])
        
        return list(set(keywords))  # Remove duplicates
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction
        # Remove common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}
        
        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter stop words and short words
        key_terms = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return unique terms
        return list(set(key_terms))[:5]
    
    def _deduplicate_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate memories"""
        seen = set()
        unique = []
        
        for memory in memories:
            key = memory.get("key", "")
            if key not in seen:
                seen.add(key)
                unique.append(memory)
        
        return unique
    
    def _rank_memories(self, memories: List[Dict[str, Any]], 
                      context_keywords: List[str]) -> List[Dict[str, Any]]:
        """Rank memories by relevance to context"""
        scored_memories = []
        
        for memory in memories:
            score = 0
            memory_str = json.dumps(memory).lower()
            
            # Score based on keyword matches
            for keyword in context_keywords:
                if keyword in memory_str:
                    score += 1
            
            # Boost recent memories
            timestamp_str = memory.get("value", {}).get("timestamp", "")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age_hours = (datetime.now() - timestamp).total_seconds() / 3600
                    if age_hours < 24:
                        score += 2
                    elif age_hours < 168:  # 1 week
                        score += 1
                except Exception:
                    pass
            
            # Boost by semantic relevance score
            relevance = memory.get("metadata", {}).get("relevance_score", 0)
            score += relevance * 3
            
            scored_memories.append((score, memory))
        
        # Sort by score (descending)
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        return [memory for _, memory in scored_memories]
    
    def _is_relevant_error(self, error_memory: Dict[str, Any], 
                          tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Check if an error memory is relevant to current context"""
        error_data = error_memory.get("value", {})
        
        # Check if same tool
        if error_data.get("tool") != tool_name:
            return False
        
        # Check for similar inputs
        error_args = error_data.get("args", {})
        
        if tool_name in ["Write", "Edit", "MultiEdit"]:
            # Check if same file or similar path
            current_path = tool_input.get("file_path", "")
            error_path = error_args.get("file_path", "")
            
            if current_path and error_path:
                current_parts = Path(current_path).parts
                error_parts = Path(error_path).parts
                
                # Check for common path components
                common = set(current_parts) & set(error_parts)
                if len(common) >= 2:
                    return True
        
        elif tool_name == "Bash":
            # Check for similar commands
            current_cmd = tool_input.get("command", "").split()[0]
            error_cmd = error_args.get("command", "").split()[0]
            
            if current_cmd == error_cmd:
                return True
        
        return False
    
    def _rank_by_similarity(self, tasks: List[Dict[str, Any]], 
                           target_description: str) -> List[Dict[str, Any]]:
        """Rank tasks by similarity to target description"""
        target_terms = set(self._extract_key_terms(target_description))
        
        scored_tasks = []
        for task in tasks:
            task_desc = task.get("value", {}).get("prompt", "")
            task_terms = set(self._extract_key_terms(task_desc))
            
            # Calculate Jaccard similarity
            intersection = len(target_terms & task_terms)
            union = len(target_terms | task_terms)
            
            similarity = intersection / union if union > 0 else 0
            scored_tasks.append((similarity, task))
        
        # Sort by similarity (descending)
        scored_tasks.sort(key=lambda x: x[0], reverse=True)
        
        return [task for _, task in scored_tasks]

# Singleton instance
_retrieval_patterns = None

def get_retrieval_patterns() -> MemoryRetrievalPatterns:
    """Get singleton retrieval patterns instance"""
    global _retrieval_patterns
    if _retrieval_patterns is None:
        _retrieval_patterns = MemoryRetrievalPatterns()
    return _retrieval_patterns