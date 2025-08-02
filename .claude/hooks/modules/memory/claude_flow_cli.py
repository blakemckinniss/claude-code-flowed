#!/usr/bin/env python3
"""
Claude Flow Memory CLI Integration
Provides command-line interface for memory operations with project namespaces
"""

import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import memory manager
from project_memory_manager import get_memory_manager

class ClaudeFlowMemoryCLI:
    """CLI interface for claude-flow memory operations"""
    
    def __init__(self):
        self.memory_manager = get_memory_manager()
    
    async def store_command(self, key: str, value: str, category: str = "general", 
                          ttl: Optional[int] = None) -> None:
        """Store memory via CLI"""
        try:
            # Parse value as JSON if possible
            try:
                parsed_value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                parsed_value = value
            
            success = await self.memory_manager.store_memory(
                key=key,
                value=parsed_value,
                category=category,
                ttl=ttl
            )
            
            if success:
                print(json.dumps({
                    "status": "success",
                    "namespace": self.memory_manager.namespace,
                    "key": key,
                    "message": "Memory stored successfully"
                }))
            else:
                print(json.dumps({
                    "status": "error",
                    "message": "Failed to store memory"
                }), file=sys.stderr)
                sys.exit(1)
                
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error storing memory: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)
    
    async def retrieve_command(self, key: str) -> None:
        """Retrieve memory via CLI"""
        try:
            data = await self.memory_manager.retrieve_memory(key)
            
            if data:
                print(json.dumps({
                    "status": "success",
                    "namespace": self.memory_manager.namespace,
                    "key": key,
                    "data": data
                }, indent=2))
            else:
                print(json.dumps({
                    "status": "not_found",
                    "namespace": self.memory_manager.namespace,
                    "key": key,
                    "message": "Memory not found"
                }))
                
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error retrieving memory: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)
    
    async def search_command(self, pattern: str, category: Optional[str] = None) -> None:
        """Search memories via CLI"""
        try:
            results = await self.memory_manager.search_memories(pattern, category)
            
            print(json.dumps({
                "status": "success",
                "namespace": self.memory_manager.namespace,
                "pattern": pattern,
                "category": category,
                "count": len(results),
                "results": results
            }, indent=2))
            
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error searching memories: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)
    
    async def sync_command(self) -> None:
        """Sync with claude-flow backend"""
        try:
            success = await self.memory_manager.sync_with_claude_flow()
            
            if success:
                print(json.dumps({
                    "status": "success",
                    "namespace": self.memory_manager.namespace,
                    "message": "Memory synchronized successfully"
                }))
            else:
                print(json.dumps({
                    "status": "error",
                    "message": "Failed to sync memory"
                }), file=sys.stderr)
                sys.exit(1)
                
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error syncing memory: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)
    
    async def stats_command(self) -> None:
        """Get memory statistics"""
        try:
            stats = self.memory_manager.get_memory_stats()
            
            print(json.dumps({
                "status": "success",
                "stats": stats
            }, indent=2))
            
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error getting stats: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)
    
    async def namespace_command(self) -> None:
        """Get current namespace information"""
        try:
            config = self.memory_manager.config
            
            print(json.dumps({
                "status": "success",
                "namespace": self.memory_manager.namespace,
                "project": config.get("project", {}),
                "memory_config": config.get("memory", {})
            }, indent=2))
            
        except Exception as e:
            print(json.dumps({
                "status": "error",
                "message": f"Error getting namespace: {e!s}"
            }), file=sys.stderr)
            sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Claude Flow Memory CLI - Project namespace memory management"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Memory commands")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store memory")
    store_parser.add_argument("--key", required=True, help="Memory key")
    store_parser.add_argument("--value", required=True, help="Memory value (JSON or string)")
    store_parser.add_argument("--category", default="general", help="Memory category")
    store_parser.add_argument("--ttl", type=int, help="Time to live in seconds")
    
    # Retrieve command
    get_parser = subparsers.add_parser("get", help="Retrieve memory")
    get_parser.add_argument("--key", required=True, help="Memory key")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("--pattern", required=True, help="Search pattern")
    search_parser.add_argument("--category", help="Filter by category")
    
    # Sync command
    subparsers.add_parser("sync", help="Sync with claude-flow backend")
    
    # Stats command
    subparsers.add_parser("stats", help="Get memory statistics")
    
    # Namespace command
    subparsers.add_parser("namespace", help="Get namespace information")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Create CLI instance
    cli = ClaudeFlowMemoryCLI()
    
    # Run async command
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        if args.command == "store":
            loop.run_until_complete(cli.store_command(
                args.key, args.value, args.category, args.ttl
            ))
        elif args.command == "get":
            loop.run_until_complete(cli.retrieve_command(args.key))
        elif args.command == "search":
            loop.run_until_complete(cli.search_command(args.pattern, args.category))
        elif args.command == "sync":
            loop.run_until_complete(cli.sync_command())
        elif args.command == "stats":
            loop.run_until_complete(cli.stats_command())
        elif args.command == "namespace":
            loop.run_until_complete(cli.namespace_command())
        else:
            parser.print_help()
            sys.exit(1)
    finally:
        loop.close()

if __name__ == "__main__":
    main()