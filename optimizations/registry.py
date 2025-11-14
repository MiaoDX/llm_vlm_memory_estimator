"""
Optimization registry - definition storage only (stateless).

The registry stores optimization class definitions and provides discovery mechanisms.
It does NOT track which optimizations are "active" - that's the estimator's responsibility.
This design allows multiple estimator instances to coexist safely without global state conflicts.
"""

import pkgutil
import importlib
from typing import Dict, List, Type, Optional, Set
import inspect

from .base import Optimization


class OptimizationRegistry:
    """
    Stateless registry of optimization definitions.

    Responsibilities:
    - Store optimization class definitions (not instances)
    - Provide name-based lookup
    - Auto-discover third-party plugins
    - Validate optimization implementations

    NOT responsible for:
    - Tracking which optimizations are active (that's MemoryEstimator's job)
    - Instantiating optimizations (that's MemoryEstimator's job)
    - Storing per-config state
    """

    def __init__(self):
        # Map: optimization_name -> Optimization class (not instance!)
        self._definitions: Dict[str, Type[Optimization]] = {}
        self._discovered: Set[str] = set()

    def register(self, optimization_class: Type[Optimization]) -> None:
        """
        Register an optimization definition.

        Args:
            optimization_class: Class implementing Optimization protocol

        Raises:
            TypeError: If class doesn't implement Optimization protocol
            ValueError: If optimization name already registered
        """
        # Validate it's a class, not an instance
        if not inspect.isclass(optimization_class):
            raise TypeError(
                f"register() expects a class, got instance: {optimization_class}. "
                "Did you mean to pass the class instead of an instance?"
            )

        # Instantiate temporarily to get name and validate protocol
        try:
            instance = optimization_class()
            name = instance.name
        except Exception as e:
            raise TypeError(
                f"Failed to instantiate {optimization_class.__name__} to validate protocol: {e}"
            )

        # Validate name format
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(
                f"Optimization name must be alphanumeric + underscores, got: '{name}'"
            )

        if name in self._definitions:
            existing_class = self._definitions[name]
            if existing_class is not optimization_class:
                raise ValueError(
                    f"Optimization '{name}' already registered by {existing_class.__name__}. "
                    f"Cannot register {optimization_class.__name__}."
                )
            # Already registered with same class, silently ignore
            return

        # Store the class (not the instance!)
        self._definitions[name] = optimization_class

    def get(self, name: str) -> Type[Optimization]:
        """
        Get optimization class by name.

        Args:
            name: Optimization name

        Returns:
            Optimization class (NOT an instance)

        Raises:
            ValueError: If optimization not found
        """
        if name not in self._definitions:
            available = ", ".join(sorted(self._definitions.keys()))
            raise ValueError(
                f"Unknown optimization '{name}'. Available optimizations: {available}\n"
                f"Hint: Check spelling or ensure the optimization plugin is installed."
            )
        return self._definitions[name]

    def list_available(self) -> List[str]:
        """
        List all registered optimization names.

        Returns:
            Sorted list of optimization names
        """
        return sorted(self._definitions.keys())

    def is_registered(self, name: str) -> bool:
        """Check if an optimization is registered"""
        return name in self._definitions

    def get_documentation(self, name: str) -> str:
        """
        Get documentation for an optimization.

        Args:
            name: Optimization name

        Returns:
            Documentation string

        Raises:
            ValueError: If optimization not found
        """
        OptClass = self.get(name)
        instance = OptClass()
        return instance.get_documentation()

    def auto_discover_plugins(self) -> None:
        """
        Auto-discover third-party optimization plugins.

        Searches the third_party/ directory for modules containing
        Optimization implementations and registers them automatically.

        This is called once at module import time.
        """
        if "third_party" in self._discovered:
            return  # Already discovered

        try:
            # Import third_party package
            from . import third_party

            # Iterate over modules in third_party/
            for finder, name, ispkg in pkgutil.iter_modules(
                third_party.__path__,
                prefix=third_party.__name__ + "."
            ):
                try:
                    # Import the module
                    module = importlib.import_module(name)

                    # Look for Optimization implementations
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)

                        # Check if it's a class implementing Optimization
                        if (
                            inspect.isclass(attr) and
                            hasattr(attr, 'name') and
                            hasattr(attr, 'get_effects') and
                            hasattr(attr, 'conflicts_with') and
                            hasattr(attr, 'get_probe_modifier') and
                            hasattr(attr, 'get_documentation') and
                            attr_name not in ('Optimization', 'Protocol')  # Exclude base types
                        ):
                            try:
                                self.register(attr)
                            except (TypeError, ValueError) as e:
                                # Skip invalid implementations with warning
                                print(f"[warn] Skipping {name}.{attr_name}: {e}")

                except Exception as e:
                    # Don't fail on plugin errors, just warn
                    print(f"[warn] Failed to load plugin {name}: {e}")

            self._discovered.add("third_party")

        except ImportError:
            # third_party package doesn't exist or can't be imported
            pass

    def suggest_similar(self, name: str, max_suggestions: int = 3) -> List[str]:
        """
        Suggest similar optimization names (fuzzy matching for better UX).

        Args:
            name: User-provided optimization name
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested names
        """
        available = self.list_available()
        if not available:
            return []

        # Simple Levenshtein-like distance for suggestions
        def distance(s1: str, s2: str) -> int:
            """Simple Levenshtein distance"""
            if len(s1) < len(s2):
                return distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # Calculate distances and sort
        suggestions = sorted(
            available,
            key=lambda opt: distance(name.lower(), opt.lower())
        )

        return suggestions[:max_suggestions]


# Global registry - definition storage only, no active state
OPTIMIZATION_REGISTRY = OptimizationRegistry()
