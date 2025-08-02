"""Pre-tool analysis components - Single Responsibility Principle implementation.

This package contains the SRP-compliant components that replace the monolithic manager:

- ValidatorRegistry: Manages validator discovery, initialization, and registration
- ValidationCoordinator: Orchestrates validation execution with parallel/caching support  
- SlimmedPreToolAnalysisManager: Facade for configuration and result processing

These components follow the Single Responsibility Principle and provide better
maintainability, testability, and separation of concerns.
"""

from .validator_registry import ValidatorRegistry
from .validation_coordinator import ValidationCoordinator
from .slimmed_manager import SlimmedPreToolAnalysisManager, GuidanceOutputHandler, PreToolAnalysisConfig

__all__ = [
    'GuidanceOutputHandler',
    'PreToolAnalysisConfig',
    'SlimmedPreToolAnalysisManager',
    'ValidationCoordinator',
    'ValidatorRegistry'
]