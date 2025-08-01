"""Configuration management for the hook system."""

import json
import os
from typing import Dict, Any, Optional, List


class Config:
    """Manages configuration for the hook system."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.config: Dict[str, Any] = {}
        self.load()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration path."""
        # Go up two levels from modules/core to get to hooks directory
        hooks_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(hooks_dir, "hook_config.json")
    
    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "enabled_analyzers": ["development", "github", "testing", "performance", "swarm"],
            "quick_tips": {
                "enabled": True,
                "message": "\n💡 Quick Tips:\n• Check agent count: CLI args or auto-decide (3-12)\n• Batch everything: TodoWrite, Task, File ops, Bash\n• Coordinate through memory and hooks\n• MCP plans, Claude executes"
            },
            "deduplication": True,
            "custom_patterns": []
        }
    
    def save(self) -> None:
        """Save configuration to file."""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        self.config[key] = value
    
    def get_enabled_analyzers(self) -> List[str]:
        """Get list of enabled analyzers."""
        return self.get("enabled_analyzers", [])