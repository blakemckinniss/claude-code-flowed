"""Universal Tool Analyzer Base Framework.

This module provides the foundational interfaces and protocols for expanding
the non-blocking stderr exit(2) feedback system to all common tool matchers.

Architecture: PostToolUse Hook → ToolAnalyzerRegistry → Specialized Analyzers → StderrFeedback
Performance Target: <100ms per tool analysis with async execution pools.
"""

from abc import ABC, abstractmethod
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Protocol, Union, Callable, 
    TypeVar, Generic, runtime_checkable
)


class FeedbackSeverity(Enum):
    """Severity levels for tool feedback."""
    INFO = 0       # Informational only - exit(0)
    WARNING = 1    # Warning guidance - exit(2) 
    ERROR = 2      # Error blocking - exit(1)
    CRITICAL = 3   # Critical failure - exit(1)


class ToolCategory(Enum):
    """Categories of tools for specialized analysis."""
    FILE_OPERATIONS = "file_ops"      # Read, Write, Edit, MultiEdit
    CODE_QUALITY = "code_quality"     # Ruff, formatters, linters
    EXECUTION = "execution"           # Bash, subprocess commands
    SEARCH_NAVIGATION = "search_nav"  # Grep, Glob, LS
    MCP_COORDINATION = "mcp_coord"    # All mcp__* tools
    VERSION_CONTROL = "version_ctrl"  # Git operations
    PACKAGE_MANAGEMENT = "pkg_mgmt"   # npm, pip, cargo operations
    TESTING = "testing"               # Test runners, frameworks
    DOCUMENTATION = "documentation"   # Docs generation, formatting
    DEPLOYMENT = "deployment"         # CI/CD, deployment operations


@dataclass
class ToolContext:
    """Context information for tool analysis."""
    tool_name: str
    tool_input: Dict[str, Any]
    tool_response: Dict[str, Any]
    execution_time: float = 0.0
    success: bool = True
    session_context: Optional[Dict[str, Any]] = None
    workflow_history: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class FeedbackResult:
    """Result from tool analysis with actionable feedback."""
    severity: FeedbackSeverity
    message: str
    exit_code: int = 0
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    analyzer_name: str = ""
    performance_impact: Optional[float] = None
    
    def __post_init__(self):
        """Set exit code based on severity."""
        if self.exit_code == 0:  # Only set if not explicitly provided
            if self.severity == FeedbackSeverity.INFO:
                self.exit_code = 0
            elif self.severity == FeedbackSeverity.WARNING:
                self.exit_code = 2  # Non-blocking guidance
            else:  # ERROR or CRITICAL
                self.exit_code = 1  # Blocking error


@dataclass
class AnalyzerMetrics:
    """Performance metrics for analyzer instances."""
    total_analyses: int = 0
    total_duration: float = 0.0
    success_count: int = 0
    error_count: int = 0
    cache_hits: int = 0
    last_execution_time: float = 0.0
    
    @property
    def average_duration(self) -> float:
        """Get average analysis duration."""
        return self.total_duration / max(1, self.total_analyses)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        return (self.success_count / max(1, self.total_analyses)) * 100


@runtime_checkable
class ToolAnalyzer(Protocol):
    """Protocol interface for all tool analyzers."""
    
    def get_analyzer_name(self) -> str:
        """Get unique analyzer name."""
        ...
    
    def get_supported_tools(self) -> List[str]:
        """Get list of tool names this analyzer supports."""
        ...
    
    def get_tool_categories(self) -> List[ToolCategory]:
        """Get tool categories this analyzer handles."""
        ...
    
    def get_priority(self) -> int:
        """Get analyzer priority (higher = runs first)."""
        ...
    
    async def analyze_tool(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze tool usage and return feedback result."""
        ...
    
    def should_analyze(self, context: ToolContext) -> bool:
        """Check if this analyzer should process the given tool context."""
        ...


class BaseToolAnalyzer(ABC):
    """Abstract base class for tool analyzers with common functionality."""
    
    def __init__(self, priority: int = 500, cache_enabled: bool = True):
        """Initialize base analyzer.
        
        Args:
            priority: Analysis priority (higher = runs first)
            cache_enabled: Whether to enable result caching
        """
        self.priority = priority
        self.cache_enabled = cache_enabled
        self.metrics = AnalyzerMetrics()
        self._cache: Dict[str, FeedbackResult] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._cache_max_age = 300.0  # 5 minutes
    
    @abstractmethod
    def get_analyzer_name(self) -> str:
        """Get unique analyzer name."""
        pass
    
    @abstractmethod
    def get_supported_tools(self) -> List[str]:
        """Get list of tool names this analyzer supports."""
        pass
    
    @abstractmethod
    def get_tool_categories(self) -> List[ToolCategory]:
        """Get tool categories this analyzer handles."""
        pass
    
    def get_priority(self) -> int:
        """Get analyzer priority."""
        return self.priority
    
    @abstractmethod 
    async def _analyze_tool_impl(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Implementation-specific analysis logic."""
        pass
    
    async def analyze_tool(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Analyze tool with caching and metrics tracking."""
        start_time = time.time()
        
        try:
            # Check cache first
            if self.cache_enabled:
                cached_result = self._get_cached_result(context)
                if cached_result:
                    self.metrics.cache_hits += 1
                    return cached_result
            
            # Perform actual analysis
            result = await self._analyze_tool_impl(context)
            
            # Cache result if applicable
            if result and self.cache_enabled:
                self._cache_result(context, result)
            
            # Update metrics
            duration = time.time() - start_time
            self.metrics.total_analyses += 1
            self.metrics.total_duration += duration
            self.metrics.last_execution_time = duration
            
            if result and result.severity != FeedbackSeverity.ERROR:
                self.metrics.success_count += 1
            else:
                self.metrics.error_count += 1
            
            return result
        
        except Exception as e:
            # Handle analysis errors gracefully
            self.metrics.error_count += 1
            return FeedbackResult(
                severity=FeedbackSeverity.ERROR,
                message=f"Analyzer error in {self.get_analyzer_name()}: {e}",
                analyzer_name=self.get_analyzer_name(),
                performance_impact=time.time() - start_time
            )
    
    def should_analyze(self, context: ToolContext) -> bool:
        """Default implementation checks supported tools."""
        supported_tools = self.get_supported_tools()
        
        # Support wildcard matching
        for pattern in supported_tools:
            if pattern == "*":
                return True
            elif pattern.endswith("*"):
                if context.tool_name.startswith(pattern[:-1]):
                    return True
            elif pattern == context.tool_name:
                return True
        
        return False
    
    def _get_cache_key(self, context: ToolContext) -> str:
        """Generate cache key for context."""
        import hashlib
        key_data = f"{context.tool_name}:{hash(str(sorted(context.tool_input.items())))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, context: ToolContext) -> Optional[FeedbackResult]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(context)
        
        if cache_key in self._cache:
            cache_time = self._cache_ttl.get(cache_key, 0)
            if time.time() - cache_time < self._cache_max_age:
                return self._cache[cache_key]
            else:
                # Remove expired entry
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
        
        return None
    
    def _cache_result(self, context: ToolContext, result: FeedbackResult):
        """Cache analysis result."""
        cache_key = self._get_cache_key(context)
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = time.time()
        
        # Limit cache size
        if len(self._cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])[:100]
            for key in oldest_keys:
                del self._cache[key]
                del self._cache_ttl[key]


class AsyncAnalyzerPool:
    """Pool for executing analyzers concurrently with performance optimization."""
    
    def __init__(self, max_concurrent: int = 4, timeout: float = 5.0):
        """Initialize analyzer pool.
        
        Args:
            max_concurrent: Maximum concurrent analyzer executions
            timeout: Timeout for individual analyzer execution
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.execution_stats = {
            "total_executions": 0,
            "timeout_count": 0,
            "error_count": 0,
            "average_duration": 0.0
        }
    
    async def execute_analyzers(
        self, 
        analyzers: List[ToolAnalyzer], 
        context: ToolContext
    ) -> List[FeedbackResult]:
        """Execute analyzers concurrently with performance monitoring."""
        if not analyzers:
            return []
        
        start_time = time.time()
        
        # Create tasks for each analyzer
        tasks = [
            self._execute_analyzer_with_semaphore(analyzer, context)
            for analyzer in analyzers
        ]
        
        # Execute with timeout protection
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout * 2  # Global timeout
            )
        except asyncio.TimeoutError:
            self.execution_stats["timeout_count"] += 1
            return [FeedbackResult(
                severity=FeedbackSeverity.WARNING,
                message="Analyzer pool timeout - some analyses skipped",
                analyzer_name="pool_manager"
            )]
        
        # Process results
        feedback_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.execution_stats["error_count"] += 1
                feedback_results.append(FeedbackResult(
                    severity=FeedbackSeverity.ERROR,
                    message=f"Analyzer error: {result}",
                    analyzer_name=f"analyzer_{i}"
                ))
            elif isinstance(result, FeedbackResult):
                feedback_results.append(result)
        
        # Update stats
        duration = time.time() - start_time
        self.execution_stats["total_executions"] += 1
        prev_avg = self.execution_stats["average_duration"]
        exec_count = self.execution_stats["total_executions"]
        self.execution_stats["average_duration"] = (
            (prev_avg * (exec_count - 1) + duration) / exec_count
        )
        
        return [r for r in feedback_results if r is not None]
    
    async def _execute_analyzer_with_semaphore(
        self, 
        analyzer: ToolAnalyzer, 
        context: ToolContext
    ) -> Optional[FeedbackResult]:
        """Execute single analyzer with semaphore protection."""
        async with self.semaphore:
            try:
                return await asyncio.wait_for(
                    analyzer.analyze_tool(context),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                return FeedbackResult(
                    severity=FeedbackSeverity.WARNING,
                    message=f"Analyzer {analyzer.get_analyzer_name()} timed out",
                    analyzer_name=analyzer.get_analyzer_name()
                )
            except Exception as e:
                return FeedbackResult(
                    severity=FeedbackSeverity.ERROR,
                    message=f"Analyzer {analyzer.get_analyzer_name()} failed: {e}",
                    analyzer_name=analyzer.get_analyzer_name()
                )


# Type aliases for better readability
AnalyzerFactory = Callable[[], ToolAnalyzer]
AnalyzerFilter = Callable[[ToolContext], bool]

T = TypeVar('T', bound=ToolAnalyzer)


class AnalyzerConfiguration(Generic[T]):
    """Configuration for analyzer registration."""
    
    def __init__(
        self,
        analyzer_class: type[T],
        priority: int = 500,
        enabled: bool = True,
        config_params: Optional[Dict[str, Any]] = None,
        activation_filter: Optional[AnalyzerFilter] = None
    ):
        """Initialize analyzer configuration.
        
        Args:
            analyzer_class: Analyzer class to instantiate
            priority: Analysis priority
            enabled: Whether analyzer is enabled
            config_params: Configuration parameters for analyzer
            activation_filter: Optional filter for when to activate analyzer
        """
        self.analyzer_class = analyzer_class
        self.priority = priority
        self.enabled = enabled
        self.config_params = config_params or {}
        self.activation_filter = activation_filter
    
    def create_analyzer(self) -> T:
        """Create analyzer instance with configuration."""
        return self.analyzer_class(priority=self.priority, **self.config_params)
    
    def should_activate(self, context: ToolContext) -> bool:
        """Check if analyzer should be activated for context."""
        if not self.enabled:
            return False
        
        if self.activation_filter:
            return self.activation_filter(context)
        
        return True