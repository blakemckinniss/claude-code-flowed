"""
Project Memory Manager - Manages claude-flow memory with project namespaces
Integrates with hooks to provide semantic context storage and retrieval
"""

import json
import os
import subprocess
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import logging

# Import vector search service
try:
    from .vector_search_service import VectorSearchService
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False
    logging.warning("Vector search service not available")

class ProjectMemoryManager:
    """Manages project-specific memory namespaces and semantic context"""
    
    def __init__(self):
        self.config_path = Path.home() / "devcontainers" / "flowed" / ".claude" / "project_config.json"
        self.config = self._load_config()
        self.namespace = self._get_namespace()
        self.memory_cache = {}
        self.semantic_threshold = self.config.get("hooks", {}).get("memory", {}).get("captureThreshold", 0.7)
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load project configuration"""
        try:
            if self.config_path.exists():
                with open(self.config_path) as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading project config: {e}")
        
        # Return the default configuration for flowed project
        return {
            "project": {"id": "flowed", "name": "Claude Flow Project"},
            "memory": {"namespace": "flowed"},
            "vectorSearch": {
                "enabled": False,
                "provider": "local",
                "dimensions": 384,
                "similarityThreshold": 0.75
            },
            "contextInjection": {
                "enabled": True,
                "maxMemories": 5,
                "relevanceThreshold": 0.8
            }
        }
    
    def _get_namespace(self) -> str:
        """Get project namespace from config"""
        return self.config.get("memory", {}).get("namespace", "default")
    
    async def store_memory(self, key: str, value: Any, category: str = "general", 
                          ttl: Optional[int] = None, metadata: Optional[Dict] = None) -> bool:
        """Store memory with semantic context using claude-flow"""
        try:
            # Prepare memory data
            memory_data = {
                "key": key,
                "value": value,
                "category": category,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
                "semantic_hash": self._generate_semantic_hash(value)
            }
            
            # Use claude-flow MCP to store
            cmd = [
                "npx", "claude-flow", "memory", "store",
                key,
                json.dumps(memory_data),
                "--namespace", self.namespace
            ]
            
            result = await self._run_command(cmd)
            
            # Cache locally for performance
            self.memory_cache[key] = memory_data
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False
    
    async def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve memory from claude-flow namespace"""
        try:
            # Check cache first
            if key in self.memory_cache:
                return self.memory_cache[key]
            
            # Retrieve from claude-flow using query with exact key
            cmd = [
                "npx", "claude-flow", "memory", "query",
                key,
                "--namespace", self.namespace
            ]
            
            result = await self._run_command(cmd)
            
            if result.returncode == 0 and result.stdout and result.stdout.strip():
                output = result.stdout.strip()
                # Check for "No results found" message
                if "No results found" in output:
                    return None
                try:
                    data = json.loads(output)
                    # Find exact key match
                    for item in data:
                        if item.get("key") == key:
                            self.memory_cache[key] = item
                            return item
                except json.JSONDecodeError:
                    pass
                
        except Exception as e:
            print(f"Error retrieving memory: {e}")
        
        return None
    
    async def search_memories(self, pattern: str, category: Optional[str] = None) -> List[Dict]:
        """Search memories with semantic matching"""
        try:
            cmd = [
                "npx", "claude-flow", "memory", "query",
                pattern,
                "--namespace", self.namespace
            ]
            
            if category:
                cmd.extend(["--category", category])
            
            result = await self._run_command(cmd)
            
            if result.returncode == 0 and result.stdout and result.stdout.strip():
                output = result.stdout.strip()
                # Check for "No results found" message
                if "No results found" in output:
                    return []
                
                # Check if it's the text format output
                if "Found" in output and "results:" in output:
                    # Parse text format - for now just return empty list
                    # since we can't reliably parse the text format
                    return []
                
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    # Handle non-JSON output or empty results
                    return []
                
        except Exception as e:
            print(f"Error searching memories: {e}")
        
        return []
    
    def analyze_semantic_context(self, content: Any, context_type: str = "general") -> Dict[str, Any]:
        """Analyze content for semantic significance"""
        analysis = {
            "relevance_score": 0.0,
            "categories": [],
            "keywords": [],
            "patterns": [],
            "should_store": False
        }
        
        # Convert content to string for analysis
        content_str = str(content).lower()
        
        # Check for significant patterns
        significance_patterns = {
            "architecture": ["design", "pattern", "structure", "system", "component"],
            "error": ["error", "exception", "failed", "issue", "problem"],
            "optimization": ["performance", "optimize", "improve", "speed", "efficiency"],
            "task": ["todo", "task", "complete", "implement", "create"],
            "pattern": ["sparc", "tdd", "agent", "swarm", "coordination"]
        }
        
        # Calculate relevance score
        matched_categories = []
        total_matches = 0
        
        for category, keywords in significance_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in content_str)
            if matches > 0:
                matched_categories.append(category)
                total_matches += matches
        
        # Calculate score
        if total_matches > 0:
            analysis["relevance_score"] = min(1.0, total_matches / 10.0)
            analysis["categories"] = matched_categories
            analysis["should_store"] = analysis["relevance_score"] >= self.semantic_threshold
        
        return analysis
    
    def _generate_semantic_hash(self, content: Any) -> str:
        """Generate semantic hash for content deduplication"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]
    
    async def _run_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else ""
        )
    
    async def sync_with_claude_flow(self) -> bool:
        """Sync local cache with claude-flow backend"""
        try:
            # Export current namespace memory to sync
            cmd = ["npx", "claude-flow", "memory", "export", f"{self.namespace}_sync.json", "--namespace", self.namespace]
            result = await self._run_command(cmd)
            return result.returncode == 0
        except Exception as e:
            print(f"Error syncing with claude-flow: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {
            "namespace": self.namespace,
            "cached_items": len(self.memory_cache),
            "categories": list(self.config.get("memory", {}).get("categories", {}).keys()),
            "semantic_enabled": self.config.get("memory", {}).get("semantic", {}).get("enabled", False),
            "vector_search_available": VECTOR_SEARCH_AVAILABLE
        }
        return stats

class EnhancedProjectMemoryManager(ProjectMemoryManager):
    """Enhanced Project Memory Manager with vector search capabilities"""
    
    def __init__(self):
        super().__init__()
        self.vector_service = None
        self._initialize_vector_service()
    
    def _initialize_vector_service(self):
        """Initialize vector search service if available and enabled"""
        if not VECTOR_SEARCH_AVAILABLE:
            self.logger.info("Vector search service not available, using base memory functionality")
            return
        
        try:
            # Check if vector search is enabled in config
            vector_config = self.config.get("vectorSearch", {})
            if vector_config.get("enabled", False):
                self.vector_service = VectorSearchService(self.config)
                self.logger.info(f"Vector search initialized with provider: {vector_config.get('provider', 'local')}")
            else:
                self.logger.info("Vector search disabled in configuration")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector search service: {e}")
            self.vector_service = None
    
    async def vector_search_memories(self, query: str, category: Optional[str] = None, 
                                   top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """Search memories using vector similarity"""
        if not self.vector_service:
            self.logger.warning("Vector search not available, falling back to standard search")
            # Fallback to standard search
            memories = await self.search_memories(query, category)
            return [(memory, 1.0) for memory in memories[:top_k]]
        
        try:
            # Get all memories for vector search
            all_memories = []
            
            # Try to get memories from cache first
            if self.memory_cache:
                all_memories = list(self.memory_cache.values())
            else:
                # Retrieve from claude-flow backend
                cmd = [
                    "npx", "claude-flow", "memory", "query",
                    "*",  # Query all memories
                    "--namespace", self.namespace
                ]
                
                if category:
                    cmd.extend(["--category", category])
                
                result = await self._run_command(cmd)
                
                if result.returncode == 0 and result.stdout and result.stdout.strip():
                    output = result.stdout.strip()
                    if not ("No results found" in output):
                        try:
                            all_memories = json.loads(output)
                        except json.JSONDecodeError:
                            self.logger.warning("Failed to parse memory results for vector search")
                            all_memories = []
            
            # Perform vector search
            if all_memories:
                return await self.vector_service.find_similar_memories(query, all_memories, top_k)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            # Fallback to standard search
            memories = await self.search_memories(query, category)
            return [(memory, 0.5) for memory in memories[:top_k]]
    
    async def get_contextual_memories(self, context: str, max_memories: int = 5) -> List[Dict[str, Any]]:
        """Get contextually relevant memories for Claude Code visibility"""
        if not self.vector_service:
            self.logger.info("Vector search not available, using semantic analysis for context")
            # Fallback using semantic analysis
            analysis = self.analyze_semantic_context(context)
            if analysis["should_store"]:
                # Get memories from relevant categories
                relevant_memories = []
                for category in analysis["categories"]:
                    memories = await self.search_memories(context, category)
                    relevant_memories.extend(memories[:2])  # Limit per category
                return relevant_memories[:max_memories]
            return []
        
        try:
            # Get all cached memories
            all_memories = []
            
            if self.memory_cache:
                all_memories = list(self.memory_cache.values())
            else:
                # Try to get from backend
                cmd = ["npx", "claude-flow", "memory", "query", "*", "--namespace", self.namespace]
                result = await self._run_command(cmd)
                
                if result.returncode == 0 and result.stdout:
                    output = result.stdout.strip()
                    if not ("No results found" in output):
                        try:
                            all_memories = json.loads(output)
                        except json.JSONDecodeError:
                            pass
            
            if all_memories:
                return await self.vector_service.get_contextual_memories(context, all_memories)
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error getting contextual memories: {e}")
            return []
    
    async def inject_context_for_claude_code(self, user_prompt: str) -> Dict[str, Any]:
        """Inject relevant context for Claude Code based on user prompt"""
        context_data = {
            "relevant_memories": [],
            "context_summary": "",
            "injection_enabled": self.config.get("contextInjection", {}).get("enabled", True)
        }
        
        if not context_data["injection_enabled"]:
            return context_data
        
        try:
            # Get contextual memories
            max_memories = self.config.get("contextInjection", {}).get("maxMemories", 5)
            relevant_memories = await self.get_contextual_memories(user_prompt, max_memories)
            
            if relevant_memories:
                context_data["relevant_memories"] = relevant_memories
                
                # Generate context summary
                categories = set()
                for memory in relevant_memories:
                    if "category" in memory:
                        categories.add(memory["category"])
                
                if categories:
                    context_data["context_summary"] = f"Found {len(relevant_memories)} relevant memories from categories: {', '.join(categories)}"
                else:
                    context_data["context_summary"] = f"Found {len(relevant_memories)} relevant memories"
            
            return context_data
            
        except Exception as e:
            self.logger.error(f"Error injecting context: {e}")
            return context_data
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced memory statistics including vector search info"""
        stats = self.get_memory_stats()
        
        # Add vector search stats
        if self.vector_service:
            vector_stats = self.vector_service.get_stats()
            stats.update({
                "vector_search": vector_stats,
                "vector_service_initialized": True
            })
        else:
            stats.update({
                "vector_search": {"enabled": False},
                "vector_service_initialized": False
            })
        
        # Add context injection info
        context_config = self.config.get("contextInjection", {})
        stats["context_injection"] = {
            "enabled": context_config.get("enabled", True),
            "max_memories": context_config.get("maxMemories", 5),
            "relevance_threshold": context_config.get("relevanceThreshold", 0.8)
        }
        
        return stats


# Singleton instances
_memory_manager = None
_enhanced_memory_manager = None

def get_memory_manager() -> ProjectMemoryManager:
    """Get singleton memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ProjectMemoryManager()
    return _memory_manager

def get_enhanced_memory_manager() -> EnhancedProjectMemoryManager:
    """Get singleton enhanced memory manager instance"""
    global _enhanced_memory_manager
    if _enhanced_memory_manager is None:
        _enhanced_memory_manager = EnhancedProjectMemoryManager()
    return _enhanced_memory_manager