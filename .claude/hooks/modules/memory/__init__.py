"""Memory management module for project-specific namespaces"""

from .project_memory_manager import (
    ProjectMemoryManager, 
    EnhancedProjectMemoryManager,
    get_memory_manager,
    get_enhanced_memory_manager
)

# Try to import vector search service (optional dependency)
try:
    from .vector_search_service import VectorSearchService, VectorSearchConfig, EmbeddingProvider
    VECTOR_SEARCH_AVAILABLE = True
except ImportError:
    VECTOR_SEARCH_AVAILABLE = False

__all__ = [
    'ProjectMemoryManager', 
    'EnhancedProjectMemoryManager',
    'get_memory_manager',
    'get_enhanced_memory_manager',
    'VECTOR_SEARCH_AVAILABLE'
]

# Add vector search exports if available
if VECTOR_SEARCH_AVAILABLE:
    __all__.extend(['VectorSearchService', 'VectorSearchConfig', 'EmbeddingProvider'])