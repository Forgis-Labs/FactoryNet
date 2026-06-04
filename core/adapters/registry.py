"""Adapter registry for dataset plugins.

Provides a registry pattern for discovering and instantiating dataset adapters
using a decorator-based registration system.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

from .base_adapter import BaseDatasetAdapter, DatasetMetadata

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for dataset adapters.

    Provides a central registry for discovering and instantiating dataset
    adapters. Adapters register themselves using the @register decorator.

    Example:
        @AdapterRegistry.register("cwru_bearing")
        class CWRUAdapter(BaseDatasetAdapter):
            ...

        # Later:
        adapter = AdapterRegistry.get("cwru_bearing", data_dir="/path/to/data")
    """

    _adapters: Dict[str, Type[BaseDatasetAdapter]] = {}
    _metadata: Dict[str, DatasetMetadata] = {}

    @classmethod
    def register(
        cls,
        name: str,
        aliases: Optional[List[str]] = None,
    ) -> Callable[[Type[BaseDatasetAdapter]], Type[BaseDatasetAdapter]]:
        """Decorator to register an adapter class.

        Args:
            name: Unique identifier for the adapter (e.g., "cwru_bearing")
            aliases: Optional alternative names for the adapter

        Returns:
            Decorator function

        Example:
            @AdapterRegistry.register("cwru_bearing", aliases=["cwru", "case_western"])
            class CWRUAdapter(BaseDatasetAdapter):
                ...
        """

        def decorator(adapter_cls: Type[BaseDatasetAdapter]) -> Type[BaseDatasetAdapter]:
            if name in cls._adapters:
                logger.warning(f"Overwriting existing adapter: {name}")

            cls._adapters[name] = adapter_cls

            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in cls._adapters:
                        logger.warning(f"Overwriting existing adapter alias: {alias}")
                    cls._adapters[alias] = adapter_cls

            logger.debug(f"Registered adapter: {name} ({adapter_cls.__name__})")
            return adapter_cls

        return decorator

    @classmethod
    def get(
        cls,
        name: str,
        data_dir: str | Path,
        **kwargs: Any,
    ) -> BaseDatasetAdapter:
        """Get an adapter instance by name.

        Args:
            name: Registered adapter name or alias
            data_dir: Directory containing dataset files
            **kwargs: Additional arguments passed to adapter constructor

        Returns:
            Instantiated adapter

        Raises:
            KeyError: If adapter name is not registered
        """
        if name not in cls._adapters:
            available = ", ".join(cls.list_adapters())
            raise KeyError(
                f"Unknown adapter: {name}. Available adapters: {available}"
            )

        adapter_cls = cls._adapters[name]
        return adapter_cls(data_dir=data_dir, **kwargs)

    @classmethod
    def list_adapters(cls) -> List[str]:
        """List all registered adapter names (excluding aliases).

        Returns:
            List of unique adapter names
        """
        # Deduplicate by class to exclude aliases
        seen_classes = set()
        names = []
        for name, adapter_cls in cls._adapters.items():
            if adapter_cls not in seen_classes:
                seen_classes.add(adapter_cls)
                names.append(name)
        return sorted(names)

    @classmethod
    def list_all(cls) -> Dict[str, Type[BaseDatasetAdapter]]:
        """Get all registered adapters including aliases.

        Returns:
            Dictionary mapping names/aliases to adapter classes
        """
        return dict(cls._adapters)

    @classmethod
    def get_metadata(cls, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a registered adapter.

        Args:
            name: Adapter name or alias

        Returns:
            DatasetMetadata if available, None otherwise
        """
        if name not in cls._adapters:
            return None

        adapter_cls = cls._adapters[name]
        # Try to get metadata from class attribute or instantiate
        if hasattr(adapter_cls, "METADATA"):
            return adapter_cls.METADATA
        return None

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters. Useful for testing."""
        cls._adapters.clear()
        cls._metadata.clear()


def register_adapter(
    name: str,
    aliases: Optional[List[str]] = None,
) -> Callable[[Type[BaseDatasetAdapter]], Type[BaseDatasetAdapter]]:
    """Convenience function for registering adapters.

    This is an alias for AdapterRegistry.register() for cleaner imports.

    Args:
        name: Unique identifier for the adapter
        aliases: Optional alternative names

    Returns:
        Decorator function

    Example:
        from core.adapters.registry import register_adapter

        @register_adapter("cwru_bearing")
        class CWRUAdapter(BaseDatasetAdapter):
            ...
    """
    return AdapterRegistry.register(name, aliases=aliases)
