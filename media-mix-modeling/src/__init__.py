"""
Core application logic and business components.
MLflow integration, attribution, dbt integration, and reporting.
"""

# Import available components with graceful failure
__all__ = []

# MLflow integration for experiment tracking
try:
    from .mlflow_integration import MMMMLflowTracker, track_mmm_experiment
    __all__.extend(["MMMMLflowTracker", "track_mmm_experiment"])
except ImportError:
    pass

# dbt Integration for data transformation
try:
    from .dbt_integration.dbt_runner import DBTRunner
    from .dbt_integration.dbt_integration import DBTIntegration
    __all__.extend(["DBTRunner", "DBTIntegration"])
except ImportError:
    pass

# Executive reporting and business intelligence
try:
    from .reports.executive_reporter import ExecutiveReporter  
    __all__.append("ExecutiveReporter")
except ImportError:
    pass

# Attribution analysis components
try:
    from .attribution.attribution_analyzer import AttributionAnalyzer
    from .attribution.attribution_engine import AttributionEngine
    __all__.extend(["AttributionAnalyzer", "AttributionEngine"])
except ImportError:
    pass

# Budget optimization (lightweight wrapper)
try:
    from .optimization import BudgetOptimizer
    __all__.append("BudgetOptimizer")
except ImportError:
    pass

# Note: Core BudgetOptimizer implementation is in models.mmm.budget_optimizer
# Note: AttributionModeler is in models.mmm.attribution_models