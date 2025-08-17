"""
Automated Sensitivity Testing Framework
LLM-based automated sensitivity analysis for econometric models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor
import itertools

# LangChain imports for LLM-based analysis
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from ..causal_inference.causal_models import CausalInferenceEngine
from ..scenario_analysis.scenario_engine import HighPerformanceScenarioEngine

logger = logging.getLogger(__name__)


class SensitivityTestResult(BaseModel):
    """Structure for sensitivity test results."""
    parameter: str = Field(description="Parameter being tested")
    baseline_value: float = Field(description="Original parameter value")
    test_value: float = Field(description="Tested parameter value")
    impact_magnitude: float = Field(description="Magnitude of impact on forecasts")
    affected_variables: List[str] = Field(description="Variables most affected by parameter change")
    risk_assessment: str = Field(description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    business_interpretation: str = Field(description="Business interpretation of the sensitivity")


class AutomatedSensitivityTester:
    """
    Automated sensitivity testing using LLM-based analysis and interpretation.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.3):
        """Initialize the automated sensitivity tester."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.parser = JsonOutputParser(pydantic_object=SensitivityTestResult)
        self.causal_engine = CausalInferenceEngine()
        self.scenario_engine = HighPerformanceScenarioEngine()
        
        # Sensitivity testing configuration
        self.default_perturbations = [-0.5, -0.25, -0.1, 0.1, 0.25, 0.5]  # ±50%, ±25%, ±10%
        self.critical_parameters = [
            'gdp_growth', 'inflation_rate', 'unemployment_rate', 'interest_rate',
            'consumer_confidence', 'housing_starts', 'retail_sales'
        ]
        
        # LLM prompt templates
        self.sensitivity_analysis_prompt = PromptTemplate(
            input_variables=["parameter", "baseline_value", "test_value", "impact_data", "forecast_changes"],
            template="""
            You are an expert econometrician analyzing model sensitivity. 

            SENSITIVITY TEST ANALYSIS:
            Parameter: {parameter}
            Baseline Value: {baseline_value}
            Test Value: {test_value}
            
            IMPACT DATA:
            {impact_data}
            
            FORECAST CHANGES:
            {forecast_changes}
            
            Analyze this sensitivity test and provide:
            1. Impact magnitude (0-10 scale where 10 is maximum sensitivity)
            2. Most affected variables (list)
            3. Risk assessment (LOW/MEDIUM/HIGH/CRITICAL)
            4. Business interpretation of this sensitivity
            
            Focus on practical business implications and model reliability concerns.
            
            Respond in the following JSON format:
            {{
                "parameter": "{parameter}",
                "baseline_value": {baseline_value},
                "test_value": {test_value},
                "impact_magnitude": [score 0-10],
                "affected_variables": ["variable1", "variable2", ...],
                "risk_assessment": "LOW/MEDIUM/HIGH/CRITICAL",
                "business_interpretation": "Clear explanation of business implications"
            }}
            """
        )
        
        logger.info("Initialized AutomatedSensitivityTester with LLM-based analysis")
    
    def run_comprehensive_sensitivity_analysis(self,
                                             model_parameters: Dict[str, float],
                                             historical_data: pd.DataFrame,
                                             forecast_function: callable,
                                             target_variables: List[str],
                                             test_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run comprehensive automated sensitivity analysis.
        
        Args:
            model_parameters: Dictionary of model parameters to test
            historical_data: Historical data for context
            forecast_function: Function that generates forecasts given parameters
            target_variables: Variables to monitor for sensitivity
            test_scenarios: Specific scenarios to test (optional)
            
        Returns:
            Comprehensive sensitivity analysis results
        """
        logger.info(f"Starting comprehensive sensitivity analysis for {len(model_parameters)} parameters...")
        
        start_time = datetime.utcnow()
        
        # Generate baseline forecast
        baseline_forecast = forecast_function(model_parameters)
        
        # Run parameter sensitivity tests
        parameter_sensitivity = self._test_parameter_sensitivity(
            model_parameters, forecast_function, baseline_forecast, target_variables
        )
        
        # Run scenario sensitivity tests
        scenario_sensitivity = self._test_scenario_sensitivity(
            historical_data, target_variables, test_scenarios
        )
        
        # Run interaction effects analysis
        interaction_effects = self._test_parameter_interactions(
            model_parameters, forecast_function, baseline_forecast, target_variables
        )
        
        # Generate LLM-based interpretations
        llm_interpretations = self._generate_llm_interpretations(
            parameter_sensitivity, scenario_sensitivity, interaction_effects
        )
        
        # Calculate overall model stability metrics
        stability_metrics = self._calculate_stability_metrics(
            parameter_sensitivity, scenario_sensitivity
        )
        
        # Generate actionable recommendations
        recommendations = self._generate_sensitivity_recommendations(
            parameter_sensitivity, stability_metrics, llm_interpretations
        )
        
        analysis_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            "sensitivity_analysis": {
                "parameter_sensitivity": parameter_sensitivity,
                "scenario_sensitivity": scenario_sensitivity,
                "interaction_effects": interaction_effects,
                "stability_metrics": stability_metrics
            },
            "llm_interpretations": llm_interpretations,
            "recommendations": recommendations,
            "metadata": {
                "analysis_timestamp": start_time.isoformat(),
                "analysis_duration_seconds": analysis_time,
                "parameters_tested": len(model_parameters),
                "perturbations_per_parameter": len(self.default_perturbations),
                "total_tests_run": len(model_parameters) * len(self.default_perturbations),
                "target_variables": target_variables
            }
        }
    
    def _test_parameter_sensitivity(self,
                                  model_parameters: Dict[str, float],
                                  forecast_function: callable,
                                  baseline_forecast: Dict[str, Any],
                                  target_variables: List[str]) -> Dict[str, Any]:
        """Test sensitivity to individual parameter changes."""
        
        logger.info("Testing parameter sensitivity...")
        parameter_results = {}
        
        for param_name, baseline_value in model_parameters.items():
            logger.info(f"Testing sensitivity for parameter: {param_name}")
            
            param_sensitivity = {
                "baseline_value": baseline_value,
                "perturbation_results": {},
                "sensitivity_score": 0.0,
                "max_impact_variable": None,
                "max_impact_magnitude": 0.0
            }
            
            impact_magnitudes = []
            
            for perturbation in self.default_perturbations:
                # Create perturbed parameters
                perturbed_params = model_parameters.copy()
                
                # Apply perturbation
                if baseline_value != 0:
                    perturbed_params[param_name] = baseline_value * (1 + perturbation)
                else:
                    perturbed_params[param_name] = perturbation
                
                try:
                    # Generate forecast with perturbed parameter
                    perturbed_forecast = forecast_function(perturbed_params)
                    
                    # Calculate impact on target variables
                    impact_analysis = self._calculate_forecast_impact(
                        baseline_forecast, perturbed_forecast, target_variables
                    )
                    
                    param_sensitivity["perturbation_results"][f"{perturbation:+.1%}"] = {
                        "perturbed_value": perturbed_params[param_name],
                        "impact_analysis": impact_analysis,
                        "max_change": max(impact_analysis["percentage_changes"].values()) if impact_analysis["percentage_changes"] else 0
                    }
                    
                    # Track maximum impact
                    max_change = max(impact_analysis["percentage_changes"].values()) if impact_analysis["percentage_changes"] else 0
                    impact_magnitudes.append(abs(max_change))
                    
                    if abs(max_change) > param_sensitivity["max_impact_magnitude"]:
                        param_sensitivity["max_impact_magnitude"] = abs(max_change)
                        param_sensitivity["max_impact_variable"] = max(
                            impact_analysis["percentage_changes"],
                            key=lambda k: abs(impact_analysis["percentage_changes"][k])
                        )
                
                except Exception as e:
                    logger.error(f"Sensitivity test failed for {param_name} with perturbation {perturbation}: {e}")
                    param_sensitivity["perturbation_results"][f"{perturbation:+.1%}"] = {"error": str(e)}
            
            # Calculate overall sensitivity score
            param_sensitivity["sensitivity_score"] = np.mean(impact_magnitudes) if impact_magnitudes else 0.0
            
            parameter_results[param_name] = param_sensitivity
        
        return parameter_results
    
    def _test_scenario_sensitivity(self,
                                 historical_data: pd.DataFrame,
                                 target_variables: List[str],
                                 test_scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        """Test sensitivity to different economic scenarios."""
        
        logger.info("Testing scenario sensitivity...")
        
        # Get scenario templates
        scenario_templates = self.scenario_engine.create_scenario_templates()
        
        # Filter to requested scenarios or use all
        if test_scenarios:
            scenarios_to_test = {k: v for k, v in scenario_templates.items() if k in test_scenarios}
        else:
            scenarios_to_test = scenario_templates
        
        try:
            # Run scenario analysis
            scenario_results = self.scenario_engine.generate_scenario_forecasts(
                historical_data=historical_data,
                scenarios=scenarios_to_test,
                forecast_horizon=12,
                n_simulations=100  # Reduced for speed
            )
            
            # Analyze scenario sensitivity
            scenario_sensitivity = {
                "scenario_impacts": {},
                "cross_scenario_correlation": {},
                "extreme_scenario_analysis": {},
                "scenario_stability_score": 0.0
            }
            
            baseline_scenario = "baseline"
            if baseline_scenario in scenario_results["scenario_forecasts"]:
                baseline_forecasts = scenario_results["scenario_forecasts"][baseline_scenario]["forecasts"]
                
                stability_scores = []
                
                for scenario_name, scenario_data in scenario_results["scenario_forecasts"].items():
                    if scenario_name != baseline_scenario and "forecasts" in scenario_data:
                        
                        # Calculate impact vs baseline
                        scenario_forecasts = scenario_data["forecasts"]
                        impact_analysis = self._calculate_forecast_impact(
                            baseline_forecasts, scenario_forecasts, target_variables
                        )
                        
                        scenario_sensitivity["scenario_impacts"][scenario_name] = {
                            "probability": scenario_data["config"]["probability"],
                            "impact_analysis": impact_analysis,
                            "risk_adjusted_impact": impact_analysis.get("average_change", 0) * scenario_data["config"]["probability"]
                        }
                        
                        # Track stability
                        avg_change = abs(impact_analysis.get("average_change", 0))
                        stability_scores.append(avg_change)
                
                scenario_sensitivity["scenario_stability_score"] = 1 / (1 + np.mean(stability_scores)) if stability_scores else 1.0
            
            return scenario_sensitivity
            
        except Exception as e:
            logger.error(f"Scenario sensitivity testing failed: {e}")
            return {"error": str(e)}
    
    def _test_parameter_interactions(self,
                                   model_parameters: Dict[str, float],
                                   forecast_function: callable,
                                   baseline_forecast: Dict[str, Any],
                                   target_variables: List[str]) -> Dict[str, Any]:
        """Test for parameter interaction effects."""
        
        logger.info("Testing parameter interactions...")
        
        # Select most important parameters for interaction testing
        important_params = list(model_parameters.keys())[:4]  # Limit for computational efficiency
        
        interaction_results = {
            "pairwise_interactions": {},
            "significant_interactions": [],
            "interaction_strength_matrix": {}
        }
        
        # Test pairwise interactions
        for param1, param2 in itertools.combinations(important_params, 2):
            try:
                interaction_effect = self._test_pairwise_interaction(
                    param1, param2, model_parameters, forecast_function, baseline_forecast, target_variables
                )
                
                interaction_key = f"{param1}_x_{param2}"
                interaction_results["pairwise_interactions"][interaction_key] = interaction_effect
                
                # Check if interaction is significant
                if interaction_effect.get("interaction_magnitude", 0) > 0.1:  # 10% threshold
                    interaction_results["significant_interactions"].append({
                        "parameters": [param1, param2],
                        "magnitude": interaction_effect["interaction_magnitude"],
                        "type": interaction_effect.get("interaction_type", "unknown")
                    })
                
            except Exception as e:
                logger.error(f"Interaction test failed for {param1} x {param2}: {e}")
        
        return interaction_results
    
    def _test_pairwise_interaction(self,
                                 param1: str, param2: str,
                                 model_parameters: Dict[str, float],
                                 forecast_function: callable,
                                 baseline_forecast: Dict[str, Any],
                                 target_variables: List[str]) -> Dict[str, Any]:
        """Test interaction between two parameters."""
        
        perturbation = 0.2  # 20% change
        
        # Get baseline values
        base_val1 = model_parameters[param1]
        base_val2 = model_parameters[param2]
        
        # Test scenarios: individual changes and combined change
        test_scenarios = {
            "param1_only": {param1: base_val1 * (1 + perturbation), param2: base_val2},
            "param2_only": {param1: base_val1, param2: base_val2 * (1 + perturbation)},
            "both_params": {param1: base_val1 * (1 + perturbation), param2: base_val2 * (1 + perturbation)}
        }
        
        scenario_impacts = {}
        
        for scenario_name, param_changes in test_scenarios.items():
            test_params = model_parameters.copy()
            test_params.update(param_changes)
            
            test_forecast = forecast_function(test_params)
            impact = self._calculate_forecast_impact(baseline_forecast, test_forecast, target_variables)
            
            scenario_impacts[scenario_name] = impact["average_change"]
        
        # Calculate interaction effect
        expected_combined = scenario_impacts["param1_only"] + scenario_impacts["param2_only"]
        actual_combined = scenario_impacts["both_params"]
        interaction_magnitude = abs(actual_combined - expected_combined)
        
        # Determine interaction type
        if actual_combined > expected_combined:
            interaction_type = "synergistic"  # Effects amplify each other
        elif actual_combined < expected_combined:
            interaction_type = "antagonistic"  # Effects cancel each other
        else:
            interaction_type = "additive"  # No interaction
        
        return {
            "parameter_1": param1,
            "parameter_2": param2,
            "individual_effects": {
                param1: scenario_impacts["param1_only"],
                param2: scenario_impacts["param2_only"]
            },
            "combined_effect": actual_combined,
            "expected_additive": expected_combined,
            "interaction_magnitude": interaction_magnitude,
            "interaction_type": interaction_type
        }
    
    def _calculate_forecast_impact(self,
                                 baseline_forecast: Dict[str, Any],
                                 test_forecast: Dict[str, Any],
                                 target_variables: List[str]) -> Dict[str, Any]:
        """Calculate the impact of parameter changes on forecasts."""
        
        impact_analysis = {
            "absolute_changes": {},
            "percentage_changes": {},
            "average_change": 0.0,
            "max_change": 0.0,
            "most_affected_variable": None
        }
        
        changes = []
        
        for var in target_variables:
            if var in baseline_forecast and var in test_forecast:
                try:
                    # Handle different forecast formats
                    if isinstance(baseline_forecast[var], list):
                        baseline_val = np.mean(baseline_forecast[var])
                        test_val = np.mean(test_forecast[var])
                    else:
                        baseline_val = baseline_forecast[var]
                        test_val = test_forecast[var]
                    
                    # Calculate changes
                    abs_change = test_val - baseline_val
                    pct_change = (abs_change / baseline_val * 100) if baseline_val != 0 else 0
                    
                    impact_analysis["absolute_changes"][var] = abs_change
                    impact_analysis["percentage_changes"][var] = pct_change
                    
                    changes.append(abs(pct_change))
                    
                    # Track maximum change
                    if abs(pct_change) > impact_analysis["max_change"]:
                        impact_analysis["max_change"] = abs(pct_change)
                        impact_analysis["most_affected_variable"] = var
                
                except Exception as e:
                    logger.warning(f"Failed to calculate impact for {var}: {e}")
        
        impact_analysis["average_change"] = np.mean(changes) if changes else 0.0
        
        return impact_analysis
    
    def _generate_llm_interpretations(self,
                                    parameter_sensitivity: Dict[str, Any],
                                    scenario_sensitivity: Dict[str, Any],
                                    interaction_effects: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LLM-based interpretations of sensitivity results."""
        
        logger.info("Generating LLM-based sensitivity interpretations...")
        
        interpretations = {
            "parameter_interpretations": {},
            "overall_model_assessment": {},
            "risk_recommendations": []
        }
        
        # Interpret individual parameters
        for param_name, sensitivity_data in parameter_sensitivity.items():
            try:
                # Prepare data for LLM analysis
                impact_data = {
                    "sensitivity_score": sensitivity_data["sensitivity_score"],
                    "max_impact_magnitude": sensitivity_data["max_impact_magnitude"],
                    "max_impact_variable": sensitivity_data["max_impact_variable"],
                    "perturbation_count": len(sensitivity_data["perturbation_results"])
                }
                
                forecast_changes = sensitivity_data["perturbation_results"]
                
                # Generate LLM interpretation
                interpretation_response = self.llm.invoke(
                    self.sensitivity_analysis_prompt.format(
                        parameter=param_name,
                        baseline_value=sensitivity_data["baseline_value"],
                        test_value=f"Multiple perturbations ({len(self.default_perturbations)} tests)",
                        impact_data=json.dumps(impact_data, indent=2),
                        forecast_changes=json.dumps(forecast_changes, indent=2, default=str)
                    )
                )
                
                # Parse LLM response
                try:
                    parsed_interpretation = self.parser.parse(interpretation_response.content)
                    interpretations["parameter_interpretations"][param_name] = parsed_interpretation.dict()
                except Exception as parse_error:
                    logger.warning(f"Failed to parse LLM response for {param_name}: {parse_error}")
                    interpretations["parameter_interpretations"][param_name] = {
                        "error": "Failed to parse LLM response",
                        "raw_response": interpretation_response.content
                    }
                
            except Exception as e:
                logger.error(f"LLM interpretation failed for {param_name}: {e}")
                interpretations["parameter_interpretations"][param_name] = {"error": str(e)}
        
        # Generate overall model assessment
        try:
            overall_stability = self._calculate_overall_stability(parameter_sensitivity, scenario_sensitivity)
            interpretations["overall_model_assessment"] = {
                "stability_score": overall_stability,
                "model_reliability": "HIGH" if overall_stability > 0.8 else "MEDIUM" if overall_stability > 0.6 else "LOW",
                "key_vulnerabilities": self._identify_key_vulnerabilities(parameter_sensitivity),
                "recommended_monitoring": self._recommend_monitoring_parameters(parameter_sensitivity)
            }
        except Exception as e:
            logger.error(f"Overall assessment generation failed: {e}")
            interpretations["overall_model_assessment"] = {"error": str(e)}
        
        return interpretations
    
    def _calculate_stability_metrics(self,
                                   parameter_sensitivity: Dict[str, Any],
                                   scenario_sensitivity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall model stability metrics."""
        
        # Parameter stability
        param_scores = [data["sensitivity_score"] for data in parameter_sensitivity.values()]
        param_stability = 1 / (1 + np.mean(param_scores)) if param_scores else 1.0
        
        # Scenario stability
        scenario_stability = scenario_sensitivity.get("scenario_stability_score", 1.0)
        
        # Overall stability (weighted average)
        overall_stability = 0.6 * param_stability + 0.4 * scenario_stability
        
        return {
            "parameter_stability": param_stability,
            "scenario_stability": scenario_stability,
            "overall_stability": overall_stability,
            "stability_grade": self._grade_stability(overall_stability),
            "critical_parameters": self._identify_critical_parameters(parameter_sensitivity),
            "model_robustness_score": min(param_stability, scenario_stability)  # Conservative measure
        }
    
    def _grade_stability(self, stability_score: float) -> str:
        """Convert stability score to letter grade."""
        if stability_score >= 0.9:
            return "A+"
        elif stability_score >= 0.8:
            return "A"
        elif stability_score >= 0.7:
            return "B"
        elif stability_score >= 0.6:
            return "C"
        elif stability_score >= 0.5:
            return "D"
        else:
            return "F"
    
    def _identify_critical_parameters(self, parameter_sensitivity: Dict[str, Any]) -> List[str]:
        """Identify parameters with highest sensitivity."""
        
        param_rankings = [
            (param, data["sensitivity_score"])
            for param, data in parameter_sensitivity.items()
        ]
        
        # Sort by sensitivity score (descending)
        param_rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 most sensitive parameters
        return [param for param, score in param_rankings[:3]]
    
    def _calculate_overall_stability(self,
                                   parameter_sensitivity: Dict[str, Any],
                                   scenario_sensitivity: Dict[str, Any]) -> float:
        """Calculate overall model stability score."""
        
        param_scores = [data["sensitivity_score"] for data in parameter_sensitivity.values()]
        avg_param_sensitivity = np.mean(param_scores) if param_scores else 0
        
        scenario_stability = scenario_sensitivity.get("scenario_stability_score", 1.0)
        
        # Combine scores (lower sensitivity = higher stability)
        param_stability = max(0, 1 - avg_param_sensitivity / 10)  # Normalize to 0-1
        
        return (param_stability + scenario_stability) / 2
    
    def _identify_key_vulnerabilities(self, parameter_sensitivity: Dict[str, Any]) -> List[str]:
        """Identify key model vulnerabilities."""
        
        vulnerabilities = []
        
        for param, data in parameter_sensitivity.items():
            if data["sensitivity_score"] > 5.0:  # High sensitivity threshold
                vulnerabilities.append(f"High sensitivity to {param} (score: {data['sensitivity_score']:.1f})")
        
        return vulnerabilities
    
    def _recommend_monitoring_parameters(self, parameter_sensitivity: Dict[str, Any]) -> List[str]:
        """Recommend parameters for ongoing monitoring."""
        
        # Sort parameters by sensitivity
        sorted_params = sorted(
            parameter_sensitivity.items(),
            key=lambda x: x[1]["sensitivity_score"],
            reverse=True
        )
        
        # Recommend top 5 most sensitive parameters for monitoring
        return [param for param, data in sorted_params[:5]]
    
    def _generate_sensitivity_recommendations(self,
                                            parameter_sensitivity: Dict[str, Any],
                                            stability_metrics: Dict[str, Any],
                                            llm_interpretations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable recommendations based on sensitivity analysis."""
        
        recommendations = {
            "immediate_actions": [],
            "monitoring_recommendations": [],
            "model_improvements": [],
            "risk_management": [],
            "confidence_intervals": {}
        }
        
        # Immediate actions for high-sensitivity parameters
        critical_params = stability_metrics.get("critical_parameters", [])
        for param in critical_params:
            recommendations["immediate_actions"].append({
                "action": f"Enhance {param} data quality and update frequency",
                "priority": "HIGH",
                "rationale": f"Model is highly sensitive to {param} changes"
            })
        
        # Monitoring recommendations
        monitoring_params = self._recommend_monitoring_parameters(parameter_sensitivity)
        for param in monitoring_params:
            sensitivity_score = parameter_sensitivity[param]["sensitivity_score"]
            recommendations["monitoring_recommendations"].append({
                "parameter": param,
                "frequency": "Daily" if sensitivity_score > 7 else "Weekly" if sensitivity_score > 4 else "Monthly",
                "alert_threshold": f"±{5 if sensitivity_score > 7 else 10 if sensitivity_score > 4 else 20}%"
            })
        
        # Model improvement suggestions
        if stability_metrics["overall_stability"] < 0.7:
            recommendations["model_improvements"].extend([
                {
                    "improvement": "Consider ensemble methods to reduce parameter sensitivity",
                    "expected_benefit": "Increased model robustness"
                },
                {
                    "improvement": "Implement regularization techniques",
                    "expected_benefit": "Reduced overfitting and improved stability"
                }
            ])
        
        return recommendations


# Convenience function for external use
def run_automated_sensitivity_testing(model_parameters: Dict[str, float],
                                     historical_data: pd.DataFrame,
                                     forecast_function: callable,
                                     target_variables: List[str]) -> Dict[str, Any]:
    """Run automated sensitivity testing with LLM-based interpretation."""
    
    tester = AutomatedSensitivityTester()
    
    return tester.run_comprehensive_sensitivity_analysis(
        model_parameters=model_parameters,
        historical_data=historical_data,
        forecast_function=forecast_function,
        target_variables=target_variables
    )