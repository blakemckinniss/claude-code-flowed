"""Intelligent Batching and Queue Management for Hook System.

Features:
- Dynamic batch sizing based on load
- Priority-aware batching
- Deadline-driven scheduling
- Affinity grouping for related operations
- Backpressure handling
"""

import asyncio
import time
import heapq
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque
import statistics
import math


class BatchingStrategy(Enum):
    """Batching strategies for different scenarios."""
    TIME_BASED = "time_based"          # Batch by time window
    SIZE_BASED = "size_based"          # Batch by item count
    ADAPTIVE = "adaptive"              # Dynamically adjust
    DEADLINE_AWARE = "deadline_aware"  # Consider deadlines
    AFFINITY = "affinity"              # Group by affinity


@dataclass
class BatchableItem:
    """Represents an item that can be batched."""
    id: str
    data: Any
    priority: int = 0
    deadline: Optional[float] = None
    affinity_group: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def __lt__(self, other):
        # For heap operations - higher priority first
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class Batch:
    """Represents a batch of items."""
    id: str
    items: List[BatchableItem]
    created_at: float
    priority: int
    total_size: int = 0
    affinity_group: Optional[str] = None
    
    def add_item(self, item: BatchableItem):
        """Add item to batch."""
        self.items.append(item)
        self.total_size += item.size_bytes
        
        # Update batch priority
        self.priority = max(self.priority, item.priority)


class AdaptiveBatchSizer:
    """Dynamically adjusts batch size based on system metrics."""
    
    def __init__(self,
                 min_batch_size: int = 1,
                 max_batch_size: int = 100,
                 target_latency_ms: float = 100):
        
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        
        # Current batch size
        self.current_size = min(10, max_batch_size)
        
        # Performance history
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        
        # Adjustment parameters
        self.increase_threshold = 0.8  # 80% of target
        self.decrease_threshold = 1.2  # 120% of target
        self.adjustment_factor = 1.2
    
    def record_batch_metrics(self, batch_size: int, latency_ms: float, items_processed: int):
        """Record batch execution metrics."""
        self.latency_history.append(latency_ms)
        
        if latency_ms > 0:
            throughput = items_processed / (latency_ms / 1000)
            self.throughput_history.append(throughput)
        
        # Adjust batch size
        self._adjust_batch_size()
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance."""
        if len(self.latency_history) < 10:
            return  # Not enough data
        
        avg_latency = statistics.mean(self.latency_history[-10:])
        
        if avg_latency < self.target_latency_ms * self.increase_threshold:
            # Can increase batch size
            new_size = int(self.current_size * self.adjustment_factor)
            self.current_size = min(new_size, self.max_batch_size)
        
        elif avg_latency > self.target_latency_ms * self.decrease_threshold:
            # Should decrease batch size
            new_size = int(self.current_size / self.adjustment_factor)
            self.current_size = max(new_size, self.min_batch_size)
    
    def get_optimal_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_size
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get sizing metrics."""
        return {
            'current_batch_size': self.current_size,
            'average_latency_ms': statistics.mean(self.latency_history) if self.latency_history else 0,
            'average_throughput': statistics.mean(self.throughput_history) if self.throughput_history else 0,
            'min_size': self.min_batch_size,
            'max_size': self.max_batch_size
        }


class PriorityQueue:
    """Priority queue with deadline awareness."""
    
    def __init__(self):
        self._heap = []
        self._item_map = {}  # id -> item
        self._lock = threading.Lock()
    
    def put(self, item: BatchableItem):
        """Add item to queue."""
        with self._lock:
            # Calculate effective priority
            effective_priority = self._calculate_effective_priority(item)
            
            # Add to heap (negate for min-heap behavior)
            heapq.heappush(self._heap, (-effective_priority, item.created_at, item))
            self._item_map[item.id] = item
    
    def get(self) -> Optional[BatchableItem]:
        """Get highest priority item."""
        with self._lock:
            while self._heap:
                _, _, item = heapq.heappop(self._heap)
                
                # Check if item still valid
                if item.id in self._item_map:
                    del self._item_map[item.id]
                    return item
            
            return None
    
    def get_batch(self, max_size: int, affinity_group: Optional[str] = None) -> List[BatchableItem]:
        """Get a batch of items."""
        batch = []
        
        with self._lock:
            # First pass: get items with matching affinity
            if affinity_group:
                temp_heap = []
                
                while self._heap and len(batch) < max_size:
                    priority, created_at, item = heapq.heappop(self._heap)
                    
                    if item.id in self._item_map:
                        if item.affinity_group == affinity_group:
                            batch.append(item)
                            del self._item_map[item.id]
                        else:
                            temp_heap.append((priority, created_at, item))
                
                # Restore non-matching items
                for entry in temp_heap:
                    heapq.heappush(self._heap, entry)
            
            # Second pass: fill remaining slots
            while self._heap and len(batch) < max_size:
                _, _, item = heapq.heappop(self._heap)
                
                if item.id in self._item_map:
                    batch.append(item)
                    del self._item_map[item.id]
        
        return batch
    
    def _calculate_effective_priority(self, item: BatchableItem) -> float:
        """Calculate effective priority considering deadlines."""
        base_priority = item.priority
        
        if item.deadline:
            time_until_deadline = item.deadline - time.time()
            if time_until_deadline > 0:
                # Boost priority as deadline approaches
                urgency_factor = 1 / (1 + time_until_deadline / 60)  # 60 second normalization
                base_priority += urgency_factor * 10
            else:
                # Past deadline - highest priority
                base_priority += 1000
        
        return base_priority
    
    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._item_map)
    
    def remove(self, item_id: str) -> bool:
        """Remove item from queue."""
        with self._lock:
            if item_id in self._item_map:
                del self._item_map[item_id]
                return True
            return False


class AffinityGroupManager:
    """Manages affinity groups for intelligent batching."""
    
    def __init__(self):
        self._groups: Dict[str, Set[str]] = defaultdict(set)
        self._item_groups: Dict[str, str] = {}
        self._group_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'count': 0, 'last_batch': 0}
        )
    
    def add_item(self, item_id: str, affinity_group: str):
        """Add item to affinity group."""
        self._groups[affinity_group].add(item_id)
        self._item_groups[item_id] = affinity_group
        self._group_stats[affinity_group]['count'] += 1
    
    def remove_item(self, item_id: str):
        """Remove item from its affinity group."""
        if item_id in self._item_groups:
            group = self._item_groups[item_id]
            self._groups[group].discard(item_id)
            del self._item_groups[item_id]
    
    def get_ready_groups(self, min_size: int = 5) -> List[str]:
        """Get affinity groups ready for batching."""
        ready = []
        current_time = time.time()
        
        for group, items in self._groups.items():
            stats = self._group_stats[group]
            
            # Check if group has enough items
            if len(items) >= min_size:
                ready.append(group)
            # Or if it's been waiting too long
            elif items and current_time - stats['last_batch'] > 1.0:
                ready.append(group)
        
        return ready
    
    def mark_group_batched(self, group: str):
        """Mark group as recently batched."""
        self._group_stats[group]['last_batch'] = time.time()


class IntelligentBatcher:
    """Main intelligent batching system."""
    
    def __init__(self,
                 strategy: BatchingStrategy = BatchingStrategy.ADAPTIVE,
                 min_batch_size: int = 1,
                 max_batch_size: int = 100,
                 batch_timeout_ms: float = 50):
        
        self.strategy = strategy
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Components
        self.queue = PriorityQueue()
        self.affinity_manager = AffinityGroupManager()
        self.adaptive_sizer = AdaptiveBatchSizer(
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            target_latency_ms=batch_timeout_ms
        )
        
        # Batch processing
        self._batch_handler: Optional[Callable] = None
        self._processing = False
        self._process_task = None
        
        # Metrics
        self._stats = {
            'total_items': 0,
            'total_batches': 0,
            'items_per_batch': deque(maxlen=100),
            'batch_latencies': deque(maxlen=100)
        }
    
    def set_batch_handler(self, handler: Callable[[List[BatchableItem]], Any]):
        """Set the batch processing handler."""
        self._batch_handler = handler
    
    async def add_item(self, 
                      data: Any,
                      priority: int = 0,
                      deadline: Optional[float] = None,
                      affinity_group: Optional[str] = None) -> str:
        """Add item for batching."""
        
        item = BatchableItem(
            id=f"item_{self._stats['total_items']}",
            data=data,
            priority=priority,
            deadline=deadline,
            affinity_group=affinity_group,
            size_bytes=len(str(data))  # Simplified size calculation
        )
        
        self._stats['total_items'] += 1
        
        # Add to queue
        self.queue.put(item)
        
        # Add to affinity group if specified
        if affinity_group:
            self.affinity_manager.add_item(item.id, affinity_group)
        
        # Start processing if not already running
        if not self._processing:
            self._processing = True
            self._process_task = asyncio.create_task(self._process_loop())
        
        return item.id
    
    async def _process_loop(self):
        """Main processing loop."""
        while self._processing:
            try:
                # Determine batch size based on strategy
                batch_size = self._get_batch_size()
                
                # Check for ready affinity groups
                ready_groups = []
                if self.strategy == BatchingStrategy.AFFINITY:
                    ready_groups = self.affinity_manager.get_ready_groups(
                        min_size=batch_size // 2
                    )
                
                # Collect batch
                batch_items = []
                
                # First, process affinity groups
                for group in ready_groups[:3]:  # Limit groups per batch
                    group_items = self.queue.get_batch(
                        max_size=batch_size - len(batch_items),
                        affinity_group=group
                    )
                    batch_items.extend(group_items)
                    
                    if group_items:
                        self.affinity_manager.mark_group_batched(group)
                
                # Fill remaining with priority items
                if len(batch_items) < batch_size:
                    remaining = self.queue.get_batch(
                        max_size=batch_size - len(batch_items)
                    )
                    batch_items.extend(remaining)
                
                # Process batch if we have items
                if batch_items:
                    await self._process_batch(batch_items)
                elif self.queue.size() == 0:
                    # No items, stop processing
                    self._processing = False
                    break
                else:
                    # Wait for timeout or more items
                    await asyncio.sleep(self.batch_timeout_ms / 1000)
                    
            except Exception as e:
                print(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
    
    def _get_batch_size(self) -> int:
        """Get batch size based on strategy."""
        if self.strategy == BatchingStrategy.ADAPTIVE:
            return self.adaptive_sizer.get_optimal_batch_size()
        elif self.strategy == BatchingStrategy.DEADLINE_AWARE:
            # Smaller batches when deadlines are tight
            return max(self.min_batch_size, self.max_batch_size // 2)
        else:
            return self.max_batch_size
    
    async def _process_batch(self, items: List[BatchableItem]):
        """Process a batch of items."""
        if not self._batch_handler:
            return
        
        start_time = time.time()
        batch_id = f"batch_{self._stats['total_batches']}"
        
        # Create batch
        Batch(
            id=batch_id,
            items=items,
            created_at=start_time,
            priority=max(item.priority for item in items) if items else 0
        )
        
        # Update stats
        self._stats['total_batches'] += 1
        self._stats['items_per_batch'].append(len(items))
        
        try:
            # Process batch
            if asyncio.iscoroutinefunction(self._batch_handler):
                await self._batch_handler(items)
            else:
                # Run in executor for sync handlers
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self._batch_handler, items)
            
            # Record metrics
            latency_ms = (time.time() - start_time) * 1000
            self._stats['batch_latencies'].append(latency_ms)
            
            # Update adaptive sizer
            self.adaptive_sizer.record_batch_metrics(
                batch_size=len(items),
                latency_ms=latency_ms,
                items_processed=len(items)
            )
            
        except Exception as e:
            print(f"Batch handler error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            'total_items': self._stats['total_items'],
            'total_batches': self._stats['total_batches'],
            'queue_size': self.queue.size(),
            'average_batch_size': (
                statistics.mean(self._stats['items_per_batch']) 
                if self._stats['items_per_batch'] else 0
            ),
            'average_latency_ms': (
                statistics.mean(self._stats['batch_latencies'])
                if self._stats['batch_latencies'] else 0
            ),
            'adaptive_metrics': self.adaptive_sizer.get_metrics(),
            'strategy': self.strategy.value
        }
    
    async def flush(self, timeout: float = 5.0):
        """Flush all pending items."""
        start_time = time.time()
        
        while self.queue.size() > 0 and time.time() - start_time < timeout:
            # Force immediate batch processing
            items = self.queue.get_batch(max_size=self.max_batch_size)
            if items:
                await self._process_batch(items)
            else:
                await asyncio.sleep(0.01)
    
    def stop(self):
        """Stop the batcher."""
        self._processing = False
        if self._process_task:
            self._process_task.cancel()


class BackpressureController:
    """Controls backpressure for the batching system."""
    
    def __init__(self,
                 max_queue_size: int = 10000,
                 high_watermark: float = 0.8,
                 low_watermark: float = 0.5):
        
        self.max_queue_size = max_queue_size
        self.high_watermark = high_watermark
        self.low_watermark = low_watermark
        
        self._current_pressure = 0.0
        self._is_throttling = False
        self._throttle_callbacks = []
    
    def update_pressure(self, queue_size: int):
        """Update backpressure based on queue size."""
        self._current_pressure = queue_size / self.max_queue_size
        
        # Check thresholds
        if self._current_pressure > self.high_watermark and not self._is_throttling:
            self._is_throttling = True
            self._notify_throttle(True)
        elif self._current_pressure < self.low_watermark and self._is_throttling:
            self._is_throttling = False
            self._notify_throttle(False)
    
    def should_accept_item(self) -> bool:
        """Check if new items should be accepted."""
        return self._current_pressure < 1.0
    
    def get_delay_ms(self) -> float:
        """Get suggested delay in milliseconds based on pressure."""
        if not self._is_throttling:
            return 0.0
        
        # Exponential backoff based on pressure
        base_delay = 10  # 10ms base
        pressure_factor = (self._current_pressure - self.high_watermark) / (1.0 - self.high_watermark)
        return base_delay * math.exp(pressure_factor * 3)  # Max ~200ms at full pressure
    
    def register_throttle_callback(self, callback: Callable[[bool], None]):
        """Register callback for throttle state changes."""
        self._throttle_callbacks.append(callback)
    
    def _notify_throttle(self, is_throttling: bool):
        """Notify callbacks of throttle state change."""
        for callback in self._throttle_callbacks:
            try:
                callback(is_throttling)
            except Exception:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get backpressure status."""
        return {
            'current_pressure': self._current_pressure,
            'is_throttling': self._is_throttling,
            'suggested_delay_ms': self.get_delay_ms()
        }


# Example integration
async def create_hook_batcher() -> IntelligentBatcher:
    """Create a batcher optimized for hook execution."""
    
    batcher = IntelligentBatcher(
        strategy=BatchingStrategy.ADAPTIVE,
        min_batch_size=5,
        max_batch_size=50,
        batch_timeout_ms=25
    )
    
    # Set up batch handler
    async def handle_hook_batch(items: List[BatchableItem]):
        # Extract hook data
        [item.data for item in items]
        
        # Process hooks in parallel
        # ... implementation ...
        
    batcher.set_batch_handler(handle_hook_batch)
    
    return batcher