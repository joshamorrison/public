"""
Optimization Module

Budget optimization and resource allocation for Media Mix Modeling.
Provides lightweight wrappers and interfaces to optimization algorithms.
"""

# Import from the main budget optimizer
try:
    import sys
    import os
    # Add the project root to sys.path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    from models.mmm.budget_optimizer import BudgetOptimizer
    __all__ = ["BudgetOptimizer"]
except ImportError:
    # If the import fails, create a placeholder
    __all__ = []