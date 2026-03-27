"""Isolated project importer — avoids module name collisions across projects.

All three projects have modules named 'models', 'graph', 'nodes', etc.
When imported via sys.path, Python caches the first one in sys.modules and
subsequent projects get the wrong module. This helper clears the conflicting
cached names before each project's imports, so each adapter gets its own
project's modules.
"""

import sys

# Module names that exist in multiple projects and would collide
_CONFLICTING = {
    "models", "graph", "nodes", "prompts", "chains",
    "conversation", "intake", "ingestion", "tools", "agents",
    "evaluation", "risk", "departments",
}


def clear_project_modules() -> None:
    """Remove conflicting module names from sys.modules.

    Call this BEFORE importing from a project directory. Each adapter
    saves direct references to the imported objects, so clearing the
    cache doesn't break previously loaded adapters.
    """
    for name in list(sys.modules):
        if name in _CONFLICTING or name == "data" or name.startswith("data."):
            del sys.modules[name]
