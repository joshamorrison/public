"""
Router Agent for AutoML Platform

Intelligent task analysis and routing agent that:
1. Analyzes natural language input to understand ML requirements
2. Examines dataset characteristics 
3. Determines the appropriate sequence of agents to execute
4. Routes tasks to specialized agents in the correct order

This is the orchestration brain of the AutoML system.
"""

import re
import json
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class MLProblemType(Enum):
    """Machine learning problem types."""
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    MULTILABEL_CLASSIFICATION = "multilabel_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES_FORECASTING = "time_series_forecasting"
    TIME_SERIES_CLASSIFICATION = "time_series_classification"
    TEXT_CLASSIFICATION = "text_classification"
    TEXT_GENERATION = "text_generation"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NAMED_ENTITY_RECOGNITION = "ner"
    IMAGE_CLASSIFICATION = "image_classification"
    OBJECT_DETECTION = "object_detection"
    RECOMMENDATION = "recommendation"
    ANOMALY_DETECTION = "anomaly_detection"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    UNKNOWN = "unknown"


class DataType(Enum):
    """Data type categories."""
    TABULAR = "tabular"
    TEXT = "text"  
    IMAGE = "image"
    TIME_SERIES = "time_series"
    AUDIO = "audio"
    MIXED = "mixed"


@dataclass
class MLTaskSpecification:
    """Complete ML task specification."""
    problem_type: MLProblemType
    data_type: DataType
    target_variable: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_shape: Optional[Tuple[int, int]] = None
    feature_types: Optional[Dict[str, str]] = None
    complexity_score: float = 0.0
    confidence_score: float = 0.0
    requirements: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


@dataclass  
class AgentWorkflow:
    """Workflow specification for agent execution."""
    sequence: List[str]  # Agent names in execution order
    parallel_groups: Optional[List[List[str]]] = None  # Groups that can run in parallel
    dependencies: Optional[Dict[str, List[str]]] = None  # Agent dependencies
    estimated_duration: float = 0.0


class RouterAgent(BaseAgent):
    """
    Router Agent for intelligent task analysis and routing.
    
    Responsibilities:
    1. Natural language processing for task understanding
    2. Dataset analysis and characterization
    3. Problem type identification
    4. Agent workflow planning
    5. Task routing and coordination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Router Agent."""
        super().__init__(
            name="Router Agent",
            description="Intelligent task analysis and agent routing coordinator",
            specialization="Task Analysis & Agent Orchestration",
            config=config,
            communication_hub=communication_hub
        )
        
        # Keywords for problem type detection
        self.problem_keywords = {
            MLProblemType.BINARY_CLASSIFICATION: [
                "binary classification", "classify", "predict", "binary",
                "yes/no", "true/false", "churn", "fraud", "spam"
            ],
            MLProblemType.MULTICLASS_CLASSIFICATION: [
                "multiclass", "classify", "categorize", "category", "class",
                "species", "type", "group", "label"
            ],
            MLProblemType.REGRESSION: [
                "predict", "forecast", "estimate", "price", "value", "amount",
                "regression", "continuous", "numeric"
            ],
            MLProblemType.CLUSTERING: [
                "cluster", "group", "segment", "unsupervised", "similarity",
                "pattern", "discover"
            ],
            MLProblemType.TIME_SERIES_FORECASTING: [
                "time series", "forecast", "predict", "temporal", "trend",
                "seasonal", "daily", "monthly", "future"
            ],
            MLProblemType.TEXT_CLASSIFICATION: [
                "text", "document", "sentiment", "topic", "classify text",
                "nlp", "natural language"
            ],
            MLProblemType.IMAGE_CLASSIFICATION: [
                "image", "picture", "photo", "visual", "computer vision",
                "classify image", "recognize"
            ],
            MLProblemType.RECOMMENDATION: [
                "recommend", "suggestion", "collaborative filtering",
                "content based", "similar", "like"
            ],
            MLProblemType.ANOMALY_DETECTION: [
                "anomaly", "outlier", "unusual", "abnormal", "fraud",
                "detect", "exception"
            ]
        }
        
        # Available agent types
        self.available_agents = [
            "EDA Agent",
            "Data Hygiene Agent", 
            "Feature Engineering Agent",
            "Classification Agent",
            "Regression Agent",
            "Clustering Agent",
            "NLP Agent",
            "Computer Vision Agent",
            "Time Series Agent", 
            "Recommendation Agent",
            "Hyperparameter Tuning Agent",
            "Ensemble Agent",
            "Validation Agent",
            "Quality Assurance Agent"
        ]
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute routing task: analyze input and create workflow plan.
        
        Args:
            context: Task context with user input and dataset info
            
        Returns:
            AgentResult with ML task specification and workflow plan
        """
        try:
            # Step 1: Analyze natural language input
            self.logger.info("Analyzing natural language input...")
            nl_analysis = self._analyze_natural_language(context.user_input)
            
            # Step 2: Analyze dataset if provided
            dataset_analysis = None
            if context.dataset_info:
                self.logger.info("Analyzing dataset characteristics...")
                dataset_analysis = self._analyze_dataset(context.dataset_info)
            
            # Step 3: Determine ML task specification
            self.logger.info("Determining ML task specification...")
            task_spec = self._create_task_specification(nl_analysis, dataset_analysis)
            
            # Step 4: Create agent workflow
            self.logger.info("Creating agent workflow plan...")
            workflow = self._create_workflow(task_spec)
            
            # Step 5: Validate and optimize workflow
            self.logger.info("Validating workflow plan...")
            optimized_workflow = self._optimize_workflow(workflow, task_spec)
            
            result_data = {
                "task_specification": {
                    "problem_type": task_spec.problem_type.value,
                    "data_type": task_spec.data_type.value,
                    "target_variable": task_spec.target_variable,
                    "complexity_score": task_spec.complexity_score,
                    "confidence_score": task_spec.confidence_score,
                    "requirements": task_spec.requirements,
                    "constraints": task_spec.constraints
                },
                "workflow": {
                    "sequence": optimized_workflow.sequence,
                    "parallel_groups": optimized_workflow.parallel_groups,
                    "dependencies": optimized_workflow.dependencies,
                    "estimated_duration": optimized_workflow.estimated_duration
                },
                "natural_language_analysis": nl_analysis,
                "dataset_analysis": dataset_analysis
            }
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Task analysis complete: {task_spec.problem_type.value} workflow planned",
                recommendations=[
                    f"Identified as {task_spec.problem_type.value} problem",
                    f"Workflow requires {len(optimized_workflow.sequence)} agents",
                    f"Estimated completion time: {optimized_workflow.estimated_duration:.1f} minutes"
                ]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Router analysis failed: {str(e)}"
            )
    
    def _analyze_natural_language(self, user_input: str) -> Dict[str, Any]:
        """Analyze natural language input to understand task requirements."""
        user_input_lower = user_input.lower()
        
        # Detect problem type based on keywords
        problem_scores = {}
        for problem_type, keywords in self.problem_keywords.items():
            score = sum(1 for keyword in keywords if keyword in user_input_lower)
            if score > 0:
                problem_scores[problem_type] = score
        
        # Find most likely problem type
        most_likely_problem = max(problem_scores, key=problem_scores.get) if problem_scores else MLProblemType.UNKNOWN
        confidence = problem_scores.get(most_likely_problem, 0) / len(self.problem_keywords.get(most_likely_problem, []))
        
        # Extract target variable if mentioned
        target_variable = self._extract_target_variable(user_input)
        
        # Extract constraints and requirements
        constraints = self._extract_constraints(user_input)
        requirements = self._extract_requirements(user_input)
        
        return {
            "problem_type": most_likely_problem,
            "confidence": confidence,
            "target_variable": target_variable,
            "constraints": constraints,
            "requirements": requirements,
            "problem_scores": {k.value: v for k, v in problem_scores.items()}
        }
    
    def _extract_target_variable(self, user_input: str) -> Optional[str]:
        """Extract target variable from natural language input."""
        # Look for patterns like "predict X", "classify Y", "target is Z"
        patterns = [
            r"predict (\w+)",
            r"classify (\w+)", 
            r"target (?:is|variable) (\w+)",
            r"outcome (?:is|variable) (\w+)",
            r"label (?:is|column) (\w+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1)
        
        return None
    
    def _extract_constraints(self, user_input: str) -> Dict[str, Any]:
        """Extract constraints from user input."""
        constraints = {}
        
        # Time constraints
        if "fast" in user_input.lower() or "quick" in user_input.lower():
            constraints["max_training_time"] = 10  # minutes
        elif "slow" in user_input.lower() or "thorough" in user_input.lower():
            constraints["max_training_time"] = 60  # minutes
        
        # Accuracy constraints
        accuracy_match = re.search(r"(\d+)%?\s*accuracy", user_input.lower())
        if accuracy_match:
            constraints["min_accuracy"] = float(accuracy_match.group(1)) / 100
        
        # Model interpretability
        if "interpret" in user_input.lower() or "explain" in user_input.lower():
            constraints["require_interpretability"] = True
        
        return constraints
    
    def _extract_requirements(self, user_input: str) -> Dict[str, Any]:
        """Extract specific requirements from user input."""
        requirements = {}
        
        # Performance requirements
        if "production" in user_input.lower():
            requirements["deployment_ready"] = True
        
        # Visualization requirements
        if "plot" in user_input.lower() or "chart" in user_input.lower():
            requirements["generate_plots"] = True
        
        # Report requirements
        if "report" in user_input.lower() or "summary" in user_input.lower():
            requirements["generate_report"] = True
        
        return requirements
    
    def _analyze_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        analysis = {
            "shape": dataset_info.get("shape"),
            "columns": dataset_info.get("columns", []),
            "dtypes": dataset_info.get("dtypes", {}),
            "missing_values": dataset_info.get("missing_values", {}),
            "data_type": DataType.TABULAR  # Default assumption
        }
        
        # Determine data type
        if "text" in str(dataset_info.get("columns", [])).lower():
            analysis["data_type"] = DataType.TEXT
        elif "image" in str(dataset_info.get("columns", [])).lower():
            analysis["data_type"] = DataType.IMAGE
        elif dataset_info.get("is_time_series", False):
            analysis["data_type"] = DataType.TIME_SERIES
        
        # Analyze complexity
        if analysis["shape"]:
            rows, cols = analysis["shape"]
            analysis["complexity_factors"] = {
                "num_rows": rows,
                "num_cols": cols,
                "size_category": "small" if rows < 1000 else "medium" if rows < 100000 else "large"
            }
        
        return analysis
    
    def _create_task_specification(
        self, 
        nl_analysis: Dict[str, Any], 
        dataset_analysis: Optional[Dict[str, Any]]
    ) -> MLTaskSpecification:
        """Create complete ML task specification."""
        
        problem_type = nl_analysis["problem_type"]
        confidence = nl_analysis["confidence"]
        
        # Determine data type
        data_type = DataType.TABULAR  # Default
        if dataset_analysis:
            data_type = dataset_analysis.get("data_type", DataType.TABULAR)
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(nl_analysis, dataset_analysis)
        
        # Get dataset shape
        dataset_shape = None
        if dataset_analysis and dataset_analysis.get("shape"):
            dataset_shape = tuple(dataset_analysis["shape"])
        
        return MLTaskSpecification(
            problem_type=problem_type,
            data_type=data_type,
            target_variable=nl_analysis["target_variable"],
            dataset_shape=dataset_shape,
            complexity_score=complexity_score,
            confidence_score=confidence,
            requirements=nl_analysis["requirements"],
            constraints=nl_analysis["constraints"]
        )
    
    def _calculate_complexity_score(
        self, 
        nl_analysis: Dict[str, Any], 
        dataset_analysis: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate task complexity score (0-1)."""
        complexity = 0.0
        
        # Base complexity by problem type
        problem_complexity = {
            MLProblemType.BINARY_CLASSIFICATION: 0.3,
            MLProblemType.MULTICLASS_CLASSIFICATION: 0.5,
            MLProblemType.REGRESSION: 0.4,
            MLProblemType.CLUSTERING: 0.6,
            MLProblemType.TIME_SERIES_FORECASTING: 0.7,
            MLProblemType.TEXT_CLASSIFICATION: 0.6,
            MLProblemType.IMAGE_CLASSIFICATION: 0.8,
            MLProblemType.RECOMMENDATION: 0.7,
        }
        
        complexity += problem_complexity.get(nl_analysis["problem_type"], 0.5)
        
        # Dataset size complexity
        if dataset_analysis and dataset_analysis.get("shape"):
            rows, cols = dataset_analysis["shape"]
            if rows > 100000:
                complexity += 0.2
            if cols > 100:
                complexity += 0.1
        
        # Missing data complexity
        if dataset_analysis and dataset_analysis.get("missing_values"):
            missing_ratio = sum(dataset_analysis["missing_values"].values()) / len(dataset_analysis["missing_values"])
            complexity += missing_ratio * 0.2
        
        return min(complexity, 1.0)
    
    def _create_workflow(self, task_spec: MLTaskSpecification) -> AgentWorkflow:
        """Create agent workflow based on task specification."""
        sequence = []
        
        # Always start with data preparation
        sequence.extend([
            "EDA Agent",
            "Data Hygiene Agent", 
            "Feature Engineering Agent"
        ])
        
        # Add problem-specific agent
        problem_agent_map = {
            MLProblemType.BINARY_CLASSIFICATION: "Classification Agent",
            MLProblemType.MULTICLASS_CLASSIFICATION: "Classification Agent",
            MLProblemType.MULTILABEL_CLASSIFICATION: "Classification Agent", 
            MLProblemType.REGRESSION: "Regression Agent",
            MLProblemType.CLUSTERING: "Clustering Agent",
            MLProblemType.TIME_SERIES_FORECASTING: "Time Series Agent",
            MLProblemType.TEXT_CLASSIFICATION: "NLP Agent",
            MLProblemType.IMAGE_CLASSIFICATION: "Computer Vision Agent",
            MLProblemType.RECOMMENDATION: "Recommendation Agent"
        }
        
        ml_agent = problem_agent_map.get(task_spec.problem_type)
        if ml_agent:
            sequence.append(ml_agent)
        
        # Add optimization agents for supervised learning
        if task_spec.problem_type in [
            MLProblemType.BINARY_CLASSIFICATION,
            MLProblemType.MULTICLASS_CLASSIFICATION,
            MLProblemType.REGRESSION
        ]:
            sequence.extend([
                "Hyperparameter Tuning Agent",
                "Ensemble Agent",
                "Validation Agent"
            ])
        else:
            # Just validation for unsupervised/other tasks
            sequence.append("Validation Agent")
        
        # Always end with quality assurance
        sequence.append("Quality Assurance Agent")
        
        # Estimate duration (minutes)
        base_duration = len(sequence) * 2  # 2 minutes per agent base
        complexity_multiplier = 1 + task_spec.complexity_score
        estimated_duration = base_duration * complexity_multiplier
        
        return AgentWorkflow(
            sequence=sequence,
            estimated_duration=estimated_duration
        )
    
    def _optimize_workflow(self, workflow: AgentWorkflow, task_spec: MLTaskSpecification) -> AgentWorkflow:
        """Optimize workflow based on constraints and requirements."""
        
        # If fast execution required, skip ensemble
        if task_spec.constraints and task_spec.constraints.get("max_training_time", 60) < 30:
            if "Ensemble Agent" in workflow.sequence:
                workflow.sequence.remove("Ensemble Agent")
                workflow.estimated_duration *= 0.8
        
        # If interpretability required, ensure we have validation
        if task_spec.requirements and task_spec.requirements.get("require_interpretability"):
            if "Validation Agent" not in workflow.sequence:
                workflow.sequence.insert(-1, "Validation Agent")  # Before QA
        
        return workflow
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Router can handle any task - it's the entry point."""
        return True
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate router task complexity."""
        # Router itself is always simple - complexity is in the routed tasks
        return TaskComplexity.SIMPLE