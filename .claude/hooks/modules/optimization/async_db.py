"""Asynchronous database operations for performance optimization.

This module provides async database functionality to eliminate synchronous I/O
bottlenecks in hook execution.
"""

import asyncio
try:
    import aiosqlite
    AIOSQLITE_AVAILABLE = True
except ImportError:
    AIOSQLITE_AVAILABLE = False
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime, timezone
import queue
import threading
import sys

# Path setup handled by centralized resolver when importing this module


class AsyncDatabaseManager:
    """Manages asynchronous database operations with batching."""
    
    def __init__(self, db_path: Path, batch_size: int = 50, batch_timeout: float = 1.0):
        """Initialize async database manager.
        
        Args:
            db_path: Path to SQLite database
            batch_size: Maximum batch size before forcing write
            batch_timeout: Maximum seconds to wait before writing partial batch
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.write_queue = queue.Queue()
        self._running = False
        self._worker_thread = None
        self._loop = None
        self._start_worker()
    
    def _start_worker(self):
        """Start the async worker thread."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._run_async_worker,
            daemon=True
        )
        self._worker_thread.start()
    
    def _run_async_worker(self):
        """Run the async event loop in worker thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._batch_write_worker())
        except Exception as e:
            print(f"Async worker error: {e}", file=sys.stderr)
        finally:
            self._loop.close()
    
    async def _batch_write_worker(self):
        """Process database writes in batches."""
        while self._running:
            batch = []
            
            # Collect batch
            deadline = time.time() + self.batch_timeout
            
            while len(batch) < self.batch_size and time.time() < deadline:
                timeout = deadline - time.time()
                
                try:
                    # Non-blocking get with timeout
                    item = self.write_queue.get(timeout=max(0.1, timeout))
                    batch.append(item)
                except queue.Empty:
                    break
            
            # Execute batch if we have items
            if batch:
                await self._execute_batch(batch)
            else:
                # Sleep briefly to avoid busy loop
                await asyncio.sleep(0.1)
    
    async def _execute_batch(self, batch: List[Dict[str, Any]]):
        """Execute a batch of database operations."""
        if not AIOSQLITE_AVAILABLE:
            return
        try:
            async with aiosqlite.connect(self.db_path) as db:
                for operation in batch:
                    op_type = operation.get("type")
                    
                    if op_type == "execute":
                        await db.execute(operation["sql"], operation.get("params", ()))
                    elif op_type == "executemany":
                        await db.executemany(operation["sql"], operation["params"])
                    elif op_type == "execute_script":
                        await db.executescript(operation["sql"])
                
                await db.commit()
                
            # Log success
            if len(batch) > 1:
                print(f"Batch executed: {len(batch)} operations", file=sys.stderr)
                
        except Exception as e:
            print(f"Batch execution error: {e}", file=sys.stderr)
    
    def queue_write(self, sql: str, params: Optional[tuple] = None):
        """Queue a single write operation."""
        operation = {
            "type": "execute",
            "sql": sql,
            "params": params or ()
        }
        self.write_queue.put(operation)
    
    def queue_many(self, sql: str, params_list: List[tuple]):
        """Queue a batch write operation."""
        operation = {
            "type": "executemany",
            "sql": sql,
            "params": params_list
        }
        self.write_queue.put(operation)
    
    def queue_script(self, sql_script: str):
        """Queue a SQL script execution."""
        operation = {
            "type": "execute_script",
            "sql": sql_script
        }
        self.write_queue.put(operation)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self.write_queue.qsize()
    
    def shutdown(self):
        """Shutdown the async worker."""
        self._running = False
        
        if self._worker_thread:
            self._worker_thread.join(timeout=5)


class PerformanceMetricsDB(AsyncDatabaseManager):
    """Specialized async database for performance metrics."""
    
    def __init__(self, db_path: Path):
        """Initialize performance metrics database."""
        super().__init__(db_path, batch_size=100, batch_timeout=2.0)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema."""
        schema = """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            operation_type TEXT NOT NULL,
            duration_seconds REAL NOT NULL,
            memory_usage_mb REAL,
            cpu_usage_percent REAL,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            optimization_score REAL DEFAULT 0.5,
            bottleneck_detected BOOLEAN DEFAULT FALSE,
            session_id TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_metrics(timestamp);
        CREATE INDEX IF NOT EXISTS idx_operation ON performance_metrics(operation_type);
        CREATE INDEX IF NOT EXISTS idx_session ON performance_metrics(session_id);
        """
        
        self.queue_script(schema)
    
    def record_metric(self, metric_data: Dict[str, Any]):
        """Record a performance metric asynchronously."""
        sql = """
        INSERT INTO performance_metrics (
            timestamp, operation_type, duration_seconds, memory_usage_mb,
            cpu_usage_percent, success, error_message, optimization_score,
            bottleneck_detected, session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            metric_data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            metric_data.get("operation_type", "unknown"),
            metric_data.get("duration", 0.0),
            metric_data.get("memory_usage_mb", 0),
            metric_data.get("cpu_usage_percent", 0),
            metric_data.get("success", True),
            metric_data.get("error_message", ""),
            metric_data.get("optimization_score", 0.5),
            metric_data.get("bottleneck_detected", False),
            metric_data.get("session_id", "")
        )
        
        self.queue_write(sql, params)
    
    def record_metrics_batch(self, metrics: List[Dict[str, Any]]):
        """Record multiple metrics in a single batch."""
        sql = """
        INSERT INTO performance_metrics (
            timestamp, operation_type, duration_seconds, memory_usage_mb,
            cpu_usage_percent, success, error_message, optimization_score,
            bottleneck_detected, session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params_list = []
        for metric in metrics:
            params = (
                metric.get("timestamp", datetime.now(timezone.utc).isoformat()),
                metric.get("operation_type", "unknown"),
                metric.get("duration", 0.0),
                metric.get("memory_usage_mb", 0),
                metric.get("cpu_usage_percent", 0),
                metric.get("success", True),
                metric.get("error_message", ""),
                metric.get("optimization_score", 0.5),
                metric.get("bottleneck_detected", False),
                metric.get("session_id", "")
            )
            params_list.append(params)
        
        self.queue_many(sql, params_list)