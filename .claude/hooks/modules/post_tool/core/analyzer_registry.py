"""Tool Analyzer Registry System.

Central registry for managing and coordinating tool analyzers across
the entire Claude Hook → ZEN → Claude Flow ecosystem.

This registry enables dynamic loading, prioritization, and coordination
of specialized analyzers for different tool categories.
"""

import asyncio
import importlib
import pkgutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Type, Union
import time
import threading

from .tool_analyzer_base import (
    ToolAnalyzer, BaseToolAnalyzer, ToolContext, FeedbackResult, 
    ToolCategory, FeedbackSeverity, AnalyzerConfiguration,
    AsyncAnalyzerPool
)


class AnalyzerRegistryError(Exception):
    """Exception raised by analyzer registry operations."""
    pass


class AnalyzerRegistry:
    """Central registry for tool analyzers with dynamic loading and coordination."""
    
    def __init__(self, max_concurrent_analyzers: int = 4):
        """Initialize analyzer registry.
        
        Args:
            max_concurrent_analyzers: Maximum concurrent analyzer executions
        """
        self.max_concurrent_analyzers = max_concurrent_analyzers
        self._analyzers: Dict[str, ToolAnalyzer] = {}
        self._configurations: Dict[str, AnalyzerConfiguration] = {}
        self._tool_mappings: Dict[str, List[str]] = defaultdict(list)
        self._category_mappings: Dict[ToolCategory, List[str]] = defaultdict(list)
        self._priority_sorted_analyzers: List[str] = []
        self._async_pool = AsyncAnalyzerPool(max_concurrent_analyzers)
        self._lock = threading.RLock()
        self._initialized = False
        
        # Performance tracking
        self.registry_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_analysis_time": 0.0,
            "cache_hit_rate": 0.0
        }
    
    def register_analyzer(
        self, 
        analyzer: Union[ToolAnalyzer, AnalyzerConfiguration],
        replace_existing: bool = False
    ) -> None:
        """Register a tool analyzer.
        
        Args:
            analyzer: Analyzer instance or configuration
            replace_existing: Whether to replace existing analyzer with same name
            
        Raises:
            AnalyzerRegistryError: If analyzer name conflicts and replace_existing=False
        """
        with self._lock:
            if isinstance(analyzer, AnalyzerConfiguration):
                config = analyzer
                analyzer_instance = config.create_analyzer()
                self._configurations[analyzer_instance.get_analyzer_name()] = config
            else:
                analyzer_instance = analyzer
            
            analyzer_name = analyzer_instance.get_analyzer_name()
            
            if analyzer_name in self._analyzers and not replace_existing:
                raise AnalyzerRegistryError(
                    f"Analyzer '{analyzer_name}' already registered. "
                    f"Use replace_existing=True to override."
                )
            
            # Register analyzer
            self._analyzers[analyzer_name] = analyzer_instance
            
            # Update tool mappings
            for tool_name in analyzer_instance.get_supported_tools():
                self._tool_mappings[tool_name].append(analyzer_name)
            
            # Update category mappings
            for category in analyzer_instance.get_tool_categories():
                self._category_mappings[category].append(analyzer_name)
            
            # Update priority ordering
            self._update_priority_ordering()
    
    def unregister_analyzer(self, analyzer_name: str) -> bool:
        """Unregister an analyzer.
        
        Args:
            analyzer_name: Name of analyzer to remove
            
        Returns:
            True if analyzer was removed, False if not found
        """
        with self._lock:
            if analyzer_name not in self._analyzers:
                return False
            
            analyzer = self._analyzers[analyzer_name]
            
            # Remove from tool mappings
            for tool_name in analyzer.get_supported_tools():
                if analyzer_name in self._tool_mappings[tool_name]:
                    self._tool_mappings[tool_name].remove(analyzer_name)
                    if not self._tool_mappings[tool_name]:
                        del self._tool_mappings[tool_name]
            
            # Remove from category mappings  
            for category in analyzer.get_tool_categories():
                if analyzer_name in self._category_mappings[category]:
                    self._category_mappings[category].remove(analyzer_name)
                    if not self._category_mappings[category]:
                        del self._category_mappings[category]
            
            # Remove analyzer
            del self._analyzers[analyzer_name]
            if analyzer_name in self._configurations:
                del self._configurations[analyzer_name]
            
            # Update priority ordering
            self._update_priority_ordering()
            
            return True
    
    def get_analyzers_for_tool(self, tool_name: str) -> List[ToolAnalyzer]:
        """Get all analyzers that support a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            List of analyzers in priority order
        """
        with self._lock:
            analyzer_names = []
            
            # Direct tool name matches
            if tool_name in self._tool_mappings:
                analyzer_names.extend(self._tool_mappings[tool_name])
            
            # Wildcard matches
            for pattern, names in self._tool_mappings.items():
                if pattern.endswith("*") and tool_name.startswith(pattern[:-1]):
                    analyzer_names.extend(names)
                elif pattern == "*":
                    analyzer_names.extend(names)
            
            # Remove duplicates and sort by priority
            unique_names = list(dict.fromkeys(analyzer_names))
            analyzers = [self._analyzers[name] for name in unique_names if name in self._analyzers]
            
            return sorted(analyzers, key=lambda a: a.get_priority(), reverse=True)
    
    def get_analyzers_for_category(self, category: ToolCategory) -> List[ToolAnalyzer]:
        """Get all analyzers for a tool category.
        
        Args:
            category: Tool category
            
        Returns:
            List of analyzers in priority order
        """
        with self._lock:
            if category not in self._category_mappings:
                return []
            
            analyzer_names = self._category_mappings[category]
            analyzers = [self._analyzers[name] for name in analyzer_names if name in self._analyzers]
            
            return sorted(analyzers, key=lambda a: a.get_priority(), reverse=True)
    
    async def analyze_tool(self, context: ToolContext) -> List[FeedbackResult]:
        """Analyze tool usage with all applicable analyzers.
        
        Args:
            context: Tool context to analyze
            
        Returns:
            List of feedback results from all analyzers
        """
        start_time = time.time()
        
        try:
            # Get applicable analyzers
            analyzers = self.get_analyzers_for_tool(context.tool_name)
            
            # Filter based on configuration and context
            active_analyzers = []
            for analyzer in analyzers:
                analyzer_name = analyzer.get_analyzer_name()
                
                # Check configuration filter
                if analyzer_name in self._configurations:
                    config = self._configurations[analyzer_name]
                    if not config.should_activate(context):
                        continue
                
                # Check analyzer's own filter
                if analyzer.should_analyze(context):
                    active_analyzers.append(analyzer)
            
            # Execute analyzers concurrently
            results = await self._async_pool.execute_analyzers(active_analyzers, context)
            
            # Update registry stats
            analysis_time = time.time() - start_time
            self._update_registry_stats(len(active_analyzers), analysis_time, results)
            
            return [r for r in results if r is not None]
        
        except Exception as e:
            self.registry_stats["failed_analyses"] += 1
            return [FeedbackResult(
                severity=FeedbackSeverity.ERROR,
                message=f"Registry analysis error: {e}",
                analyzer_name="registry_manager"
            )]
    
    def get_highest_priority_result(self, results: List[FeedbackResult]) -> Optional[FeedbackResult]:
        """Get highest priority result from analysis results.
        
        Args:
            results: List of feedback results
            
        Returns:
            Highest priority result or None
        """
        if not results:
            return None
        
        # Sort by severity (higher = more important)
        severity_order = {
            FeedbackSeverity.CRITICAL: 4,
            FeedbackSeverity.ERROR: 3,
            FeedbackSeverity.WARNING: 2,
            FeedbackSeverity.INFO: 1
        }
        
        return max(results, key=lambda r: severity_order.get(r.severity, 0))
    
    def auto_discover_analyzers(self, package_paths: List[str]) -> int:
        """Auto-discover and register analyzers from specified packages.
        
        Args:
            package_paths: List of package paths to search
            
        Returns:
            Number of analyzers discovered and registered
        """
        discovered_count = 0
        
        for package_path in package_paths:
            try:
                discovered_count += self._discover_analyzers_in_package(package_path)
            except Exception as e:
                print(f"Warning: Failed to discover analyzers in {package_path}: {e}", file=sys.stderr)
        
        return discovered_count
    
    def _discover_analyzers_in_package(self, package_path: str) -> int:
        """Discover analyzers in a specific package."""
        discovered_count = 0
        
        try:
            # Import the package
            package = importlib.import_module(package_path)
            
            # Walk through package modules
            if hasattr(package, "__path__"):
                for _, module_name, _ in pkgutil.iter_modules(package.__path__, package_path + "."):
                    try:
                        module = importlib.import_module(module_name)
                        discovered_count += self._register_analyzers_from_module(module)
                    except Exception as e:
                        print(f"Warning: Failed to import {module_name}: {e}", file=sys.stderr)
        
        except ImportError as e:
            print(f"Warning: Could not import package {package_path}: {e}", file=sys.stderr)
        
        return discovered_count
    
    def _register_analyzers_from_module(self, module) -> int:
        """Register analyzers found in a module."""
        registered_count = 0
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            # Check if it's an analyzer class
            if (isinstance(attr, type) and 
                issubclass(attr, BaseToolAnalyzer) and 
                attr != BaseToolAnalyzer):
                
                try:
                    # Instantiate and register
                    analyzer = attr()
                    self.register_analyzer(analyzer, replace_existing=True)
                    registered_count += 1
                except Exception as e:
                    print(f"Warning: Failed to register analyzer {attr_name}: {e}", file=sys.stderr)
        
        return registered_count
    
    def _update_priority_ordering(self):
        """Update priority-based ordering of analyzers."""
        analyzers_with_priority = [
            (name, analyzer.get_priority()) 
            for name, analyzer in self._analyzers.items()
        ]
        
        # Sort by priority (descending)
        self._priority_sorted_analyzers = [
            name for name, _ in sorted(analyzers_with_priority, key=lambda x: x[1], reverse=True)
        ]
    
    def _update_registry_stats(self, analyzer_count: int, analysis_time: float, results: List[FeedbackResult]):
        """Update registry performance statistics."""
        self.registry_stats["total_analyses"] += 1
        
        # Check if analysis was successful
        has_errors = any(r.severity in [FeedbackSeverity.ERROR, FeedbackSeverity.CRITICAL] for r in results)
        if not has_errors:
            self.registry_stats["successful_analyses"] += 1
        else:
            self.registry_stats["failed_analyses"] += 1
        
        # Update average analysis time
        prev_avg = self.registry_stats["average_analysis_time"]
        total_count = self.registry_stats["total_analyses"]
        self.registry_stats["average_analysis_time"] = (
            (prev_avg * (total_count - 1) + analysis_time) / total_count
        )
        
        # Update cache hit rate (aggregate from analyzers)
        cache_hits = sum(getattr(a, 'metrics', type('', (), {'cache_hits': 0})).cache_hits 
                        for a in self._analyzers.values())
        total_analyzer_calls = sum(getattr(a, 'metrics', type('', (), {'total_analyses': 0})).total_analyses 
                                  for a in self._analyzers.values())
        
        if total_analyzer_calls > 0:
            self.registry_stats["cache_hit_rate"] = (cache_hits / total_analyzer_calls) * 100
    
    def get_registry_info(self) -> Dict:
        """Get comprehensive registry information."""
        with self._lock:
            return {
                "total_analyzers": len(self._analyzers),
                "analyzer_names": list(self._analyzers.keys()),
                "tool_mappings": dict(self._tool_mappings),
                "category_mappings": {cat.value: names for cat, names in self._category_mappings.items()},
                "priority_order": self._priority_sorted_analyzers,
                "performance_stats": self.registry_stats.copy(),
                "async_pool_stats": self._async_pool.execution_stats.copy()
            }
    
    def clear_caches(self):
        """Clear all analyzer caches."""
        with self._lock:
            for analyzer in self._analyzers.values():
                if hasattr(analyzer, '_cache'):
                    analyzer._cache.clear()
                if hasattr(analyzer, '_cache_ttl'):
                    analyzer._cache_ttl.clear()
    
    def shutdown(self):
        """Shutdown registry and cleanup resources."""
        with self._lock:
            self.clear_caches()
            # Additional cleanup could be added here


# Global registry instance
_global_registry: Optional[AnalyzerRegistry] = None
_registry_lock = threading.Lock()


def get_global_registry() -> AnalyzerRegistry:
    """Get or create the global analyzer registry."""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = AnalyzerRegistry()
                
                # Auto-discover analyzers in standard locations
                discover_paths = [
                    "modules.post_tool.analyzers.specialized",
                    "modules.post_tool.analyzers.builtin"
                ]
                
                _global_registry.auto_discover_analyzers(discover_paths)
    
    return _global_registry


def register_analyzer_globally(analyzer: Union[ToolAnalyzer, AnalyzerConfiguration]) -> None:
    """Register analyzer in global registry."""
    registry = get_global_registry()
    registry.register_analyzer(analyzer)


def analyze_tool_globally(context: ToolContext) -> List[FeedbackResult]:
    """Analyze tool using global registry (sync wrapper)."""
    registry = get_global_registry()
    
    # Create new event loop if none exists
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(registry.analyze_tool(context))