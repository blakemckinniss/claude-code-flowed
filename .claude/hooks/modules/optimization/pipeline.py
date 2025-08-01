"""Hook execution pipeline with composable stages and parallel processing."""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Union
import threading
from queue import Queue, Empty
import logging


@dataclass
class PipelineStage:
    """Represents a single stage in the pipeline."""
    name: str
    handler: Callable
    parallel: bool = False
    timeout: float = 5.0
    fallback: Optional[Callable] = None
    retry_count: int = 0


@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    success: bool
    stages_completed: List[str]
    results: Dict[str, Any]
    errors: Dict[str, str]
    total_duration: float
    stage_durations: Dict[str, float]


class StageProcessor(ABC):
    """Abstract base for stage processors."""
    
    @abstractmethod
    def process(self, data: Any, context: Dict[str, Any]) -> Any:
        """Process data through this stage."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> bool:
        """Validate input data for this stage."""
        pass


class HookPipeline:
    """
    Manages multi-stage hook execution with parallel processing.
    
    Features:
    - Composable pipeline stages
    - Parallel stage execution
    - Error handling and fallbacks
    - Performance monitoring
    - Context propagation
    """
    
    def __init__(self, max_workers: int = 4):
        self.stages: List[PipelineStage] = []
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.context = {}
        self.logger = logging.getLogger(__name__)
    
    def add_stage(
        self,
        name: str,
        handler: Callable,
        parallel: bool = False,
        timeout: float = 5.0,
        fallback: Optional[Callable] = None,
        retry_count: int = 0
    ) -> 'HookPipeline':
        """Add a stage to the pipeline."""
        stage = PipelineStage(
            name=name,
            handler=handler,
            parallel=parallel,
            timeout=timeout,
            fallback=fallback,
            retry_count=retry_count
        )
        self.stages.append(stage)
        return self
    
    def execute(self, data: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Execute the pipeline with the given data."""
        start_time = time.time()
        
        # Initialize execution context
        exec_context = self.context.copy()
        if context:
            exec_context.update(context)
        
        results = {}
        errors = {}
        stage_durations = {}
        stages_completed = []
        
        # Group stages by execution order
        stage_groups = self._group_stages()
        
        try:
            current_data = data
            
            for group in stage_groups:
                if len(group) == 1 and not group[0].parallel:
                    # Sequential execution
                    stage = group[0]
                    stage_result = self._execute_stage(
                        stage, current_data, exec_context, stage_durations
                    )
                    
                    if stage_result['success']:
                        results[stage.name] = stage_result['result']
                        stages_completed.append(stage.name)
                        current_data = stage_result['result']
                    else:
                        errors[stage.name] = stage_result['error']
                        if not stage.fallback:
                            break
                else:
                    # Parallel execution
                    parallel_results = self._execute_parallel_stages(
                        group, current_data, exec_context, stage_durations
                    )
                    
                    for stage_name, stage_result in parallel_results.items():
                        if stage_result['success']:
                            results[stage_name] = stage_result['result']
                            stages_completed.append(stage_name)
                        else:
                            errors[stage_name] = stage_result['error']
                    
                    # For parallel stages, merge results
                    if parallel_results:
                        current_data = self._merge_parallel_results(
                            [r['result'] for r in parallel_results.values() if r['success']]
                        )
            
            success = len(errors) == 0
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            success = False
            errors['pipeline'] = str(e)
        
        total_duration = time.time() - start_time
        
        return PipelineResult(
            success=success,
            stages_completed=stages_completed,
            results=results,
            errors=errors,
            total_duration=total_duration,
            stage_durations=stage_durations
        )
    
    def _group_stages(self) -> List[List[PipelineStage]]:
        """Group stages by parallel execution capability."""
        groups = []
        current_group = []
        
        for stage in self.stages:
            if stage.parallel and current_group and current_group[-1].parallel:
                current_group.append(stage)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [stage]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _execute_stage(
        self,
        stage: PipelineStage,
        data: Any,
        context: Dict[str, Any],
        durations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Execute a single stage."""
        start_time = time.time()
        retry_count = 0
        
        while retry_count <= stage.retry_count:
            try:
                # Execute with timeout
                future = self.executor.submit(stage.handler, data, context)
                result = future.result(timeout=stage.timeout)
                
                durations[stage.name] = time.time() - start_time
                return {'success': True, 'result': result}
                
            except Exception as e:
                self.logger.warning(f"Stage {stage.name} failed (attempt {retry_count + 1}): {e}")
                retry_count += 1
                
                if retry_count > stage.retry_count:
                    # Try fallback if available
                    if stage.fallback:
                        try:
                            result = stage.fallback(data, context)
                            durations[stage.name] = time.time() - start_time
                            return {'success': True, 'result': result}
                        except Exception as fallback_e:
                            durations[stage.name] = time.time() - start_time
                            return {'success': False, 'error': str(fallback_e)}
                    
                    durations[stage.name] = time.time() - start_time
                    return {'success': False, 'error': str(e)}
        
        durations[stage.name] = time.time() - start_time
        return {'success': False, 'error': 'Max retries exceeded'}
    
    def _execute_parallel_stages(
        self,
        stages: List[PipelineStage],
        data: Any,
        context: Dict[str, Any],
        durations: Dict[str, float]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute multiple stages in parallel."""
        results = {}
        futures = {}
        
        # Submit all parallel stages
        for stage in stages:
            future = self.executor.submit(
                self._execute_stage, stage, data, context, durations
            )
            futures[future] = stage.name
        
        # Collect results
        for future in as_completed(futures):
            stage_name = futures[future]
            try:
                result = future.result()
                results[stage_name] = result
            except Exception as e:
                results[stage_name] = {'success': False, 'error': str(e)}
        
        return results
    
    def _merge_parallel_results(self, results: List[Any]) -> Any:
        """Merge results from parallel stages."""
        # Default implementation: merge dictionaries
        if not results:
            return {}
        
        if all(isinstance(r, dict) for r in results):
            merged = {}
            for result in results:
                merged.update(result)
            return merged
        
        # For other types, return list of results
        return results
    
    def set_context(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.context[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pipeline execution metrics."""
        return {
            'stages': len(self.stages),
            'max_workers': self.max_workers,
            'parallel_stages': sum(1 for s in self.stages if s.parallel)
        }
    
    def shutdown(self):
        """Shutdown the pipeline executor."""
        self.executor.shutdown(wait=True)


class AsyncHookPipeline:
    """Asynchronous version of the hook pipeline."""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.context = {}
    
    async def execute(self, data: Any, context: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Execute pipeline asynchronously."""
        start_time = time.time()
        exec_context = self.context.copy()
        if context:
            exec_context.update(context)
        
        results = {}
        errors = {}
        stage_durations = {}
        stages_completed = []
        
        current_data = data
        
        for stage in self.stages:
            stage_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(stage.handler):
                    result = await asyncio.wait_for(
                        stage.handler(current_data, exec_context),
                        timeout=stage.timeout
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        None, stage.handler, current_data, exec_context
                    )
                
                results[stage.name] = result
                stages_completed.append(stage.name)
                current_data = result
                
            except Exception as e:
                errors[stage.name] = str(e)
                if stage.fallback:
                    try:
                        if asyncio.iscoroutinefunction(stage.fallback):
                            result = await stage.fallback(current_data, exec_context)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, stage.fallback, current_data, exec_context
                            )
                        results[stage.name] = result
                        stages_completed.append(stage.name)
                        current_data = result
                    except Exception as fallback_e:
                        errors[f"{stage.name}_fallback"] = str(fallback_e)
                        break
                else:
                    break
            
            stage_durations[stage.name] = time.time() - stage_start
        
        return PipelineResult(
            success=len(errors) == 0,
            stages_completed=stages_completed,
            results=results,
            errors=errors,
            total_duration=time.time() - start_time,
            stage_durations=stage_durations
        )


# Example usage functions
def create_validation_pipeline() -> HookPipeline:
    """Create a standard validation pipeline."""
    pipeline = HookPipeline(max_workers=4)
    
    # Add stages
    pipeline.add_stage(
        "parse_input",
        lambda data, ctx: json.loads(data) if isinstance(data, str) else data,
        parallel=False,
        timeout=1.0
    )
    
    pipeline.add_stage(
        "validate_schema",
        lambda data, ctx: data if 'tool' in data else None,
        parallel=False,
        timeout=2.0
    )
    
    pipeline.add_stage(
        "check_permissions",
        lambda data, ctx: {'allowed': True, 'data': data},
        parallel=True,
        timeout=3.0
    )
    
    return pipeline


def create_processing_pipeline() -> HookPipeline:
    """Create a processing pipeline with parallel stages."""
    pipeline = HookPipeline(max_workers=6)
    
    # Sequential preprocessing
    pipeline.add_stage("preprocess", lambda d, c: d, timeout=2.0)
    
    # Parallel processing stages
    pipeline.add_stage("analyze", lambda d, c: {'analysis': 'complete'}, parallel=True)
    pipeline.add_stage("validate", lambda d, c: {'validation': 'passed'}, parallel=True)
    pipeline.add_stage("optimize", lambda d, c: {'optimization': 'done'}, parallel=True)
    
    # Sequential postprocessing
    pipeline.add_stage("postprocess", lambda d, c: d, timeout=2.0)
    
    return pipeline