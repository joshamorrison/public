"""
GenAI-powered demand planning and scenario simulation capabilities.
Extends the econometric forecasting platform with business-specific demand planning features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class DemandScenario(BaseModel):
    """Structure for demand planning scenarios."""
    scenario_name: str = Field(description="Name of the scenario")
    probability: float = Field(description="Probability of occurrence (0-1)")
    demand_impact: str = Field(description="Expected impact on demand")
    key_drivers: List[str] = Field(description="Main drivers of this scenario")
    business_implications: List[str] = Field(description="Business implications")
    recommended_actions: List[str] = Field(description="Recommended actions")
    early_warning_indicators: List[str] = Field(description="Indicators to monitor")


class CustomerSegment(BaseModel):
    """Structure for customer segment analysis."""
    segment_name: str = Field(description="Name of the customer segment")
    demand_sensitivity: str = Field(description="Sensitivity to economic changes")
    key_characteristics: List[str] = Field(description="Key characteristics")
    purchasing_patterns: List[str] = Field(description="Typical purchasing patterns")
    risk_factors: List[str] = Field(description="Risk factors for this segment")


class GenAIDemandPlanner:
    """GenAI-powered demand planning and scenario simulation."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """Initialize the demand planner."""
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.scenario_parser = JsonOutputParser(pydantic_object=DemandScenario)
        self.segment_parser = JsonOutputParser(pydantic_object=CustomerSegment)
        
        # Template for scenario generation
        self.scenario_template = PromptTemplate(
            input_variables=["economic_indicators", "industry_context", "time_horizon"],
            template="""
            You are a demand planning expert analyzing potential future scenarios.
            
            ECONOMIC CONTEXT:
            {economic_indicators}
            
            INDUSTRY CONTEXT:
            {industry_context}
            
            TIME HORIZON: {time_horizon} months
            
            Generate a comprehensive demand scenario that includes:
            - scenario_name: A descriptive name for this scenario
            - probability: Estimated probability (0.0 to 1.0)
            - demand_impact: Expected impact on demand (increase/decrease/stable with %)
            - key_drivers: Main economic/business factors driving this scenario
            - business_implications: How this affects business operations
            - recommended_actions: Specific actions businesses should take
            - early_warning_indicators: Key metrics to monitor for early detection
            
            Focus on actionable insights for supply chain and inventory management.
            
            {format_instructions}
            """
        )
        
        # Template for customer segmentation
        self.segmentation_template = PromptTemplate(
            input_variables=["economic_forecast", "customer_data_summary"],
            template="""
            Analyze customer segments based on economic forecasts and behavior patterns.
            
            ECONOMIC FORECAST:
            {economic_forecast}
            
            CUSTOMER DATA CONTEXT:
            {customer_data_summary}
            
            Create a customer segment analysis with:
            - segment_name: Descriptive name for this customer segment
            - demand_sensitivity: How sensitive to economic changes (High/Medium/Low)
            - key_characteristics: Defining characteristics of this segment
            - purchasing_patterns: Typical buying behavior patterns
            - risk_factors: Potential risks affecting this segment
            
            Focus on how economic changes will affect different customer types.
            
            {format_instructions}
            """
        )
    
    def generate_demand_scenarios(self, economic_forecasts: Dict[str, Any],
                                industry: str = "retail",
                                time_horizon: int = 12,
                                scenario_count: int = 3) -> List[DemandScenario]:
        """
        Generate multiple demand planning scenarios based on economic forecasts.
        
        Args:
            economic_forecasts: Dictionary of economic indicator forecasts
            industry: Industry context for demand planning
            time_horizon: Planning horizon in months
            scenario_count: Number of scenarios to generate
        
        Returns:
            List of demand scenarios
        """
        scenarios = []
        
        # Prepare economic context
        economic_summary = self._summarize_economic_forecasts(economic_forecasts)
        
        # Define scenario types
        scenario_types = [
            "Base Case - Current trends continue",
            "Optimistic - Economic acceleration and growth",
            "Pessimistic - Economic slowdown or recession",
            "Black Swan - Unexpected external shock",
            "Structural Change - Fundamental market shift"
        ]
        
        for i, scenario_type in enumerate(scenario_types[:scenario_count]):
            try:
                # Customize industry context
                industry_context = f"""
                Industry: {industry}
                Scenario Type: {scenario_type}
                Market Conditions: Dynamic and competitive
                Supply Chain: Global with multiple dependencies
                Customer Base: Diverse segments with varying sensitivities
                """
                
                # Format template
                formatted_template = self.scenario_template.partial(
                    format_instructions=self.scenario_parser.get_format_instructions()
                )
                
                prompt = formatted_template.format(
                    economic_indicators=economic_summary,
                    industry_context=industry_context,
                    time_horizon=time_horizon
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                scenario = self.scenario_parser.parse(response.content)
                scenarios.append(scenario)
                
                logger.info(f"Generated scenario: {scenario.scenario_name}")
                
            except Exception as e:
                logger.error(f"Error generating scenario {i+1}: {e}")
                continue
        
        return scenarios
    
    def analyze_customer_segments(self, economic_forecasts: Dict[str, Any],
                                customer_context: str = "B2C retail customers") -> List[CustomerSegment]:
        """
        Analyze customer segments based on economic forecast impacts.
        
        Args:
            economic_forecasts: Economic indicator forecasts
            customer_context: Description of customer base
        
        Returns:
            List of customer segment analyses
        """
        segments = []
        
        # Prepare forecast summary
        forecast_summary = self._summarize_economic_forecasts(economic_forecasts)
        
        # Define typical customer segments
        segment_types = [
            "Price-Sensitive Consumers",
            "Premium/Luxury Customers", 
            "Business/B2B Customers",
            "Essential Goods Purchasers"
        ]
        
        for segment_type in segment_types:
            try:
                customer_data = f"""
                Customer Context: {customer_context}
                Segment Focus: {segment_type}
                Typical Characteristics: Varies by economic sensitivity
                Purchase Drivers: Price, quality, convenience, necessity
                """
                
                formatted_template = self.segmentation_template.partial(
                    format_instructions=self.segment_parser.get_format_instructions()
                )
                
                prompt = formatted_template.format(
                    economic_forecast=forecast_summary,
                    customer_data_summary=customer_data
                )
                
                response = self.llm.invoke([HumanMessage(content=prompt)])
                segment = self.segment_parser.parse(response.content)
                segments.append(segment)
                
                logger.info(f"Analyzed segment: {segment.segment_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing segment {segment_type}: {e}")
                continue
        
        return segments
    
    def generate_demand_planning_report(self, economic_forecasts: Dict[str, Any],
                                      scenarios: List[DemandScenario],
                                      segments: List[CustomerSegment]) -> str:
        """
        Generate comprehensive demand planning report.
        
        Args:
            economic_forecasts: Economic forecasts
            scenarios: Generated demand scenarios
            segments: Customer segment analyses
        
        Returns:
            Formatted demand planning report
        """
        report_template = PromptTemplate(
            input_variables=["economic_summary", "scenarios_summary", "segments_summary"],
            template="""
            Generate a comprehensive DEMAND PLANNING EXECUTIVE REPORT:
            
            ECONOMIC OUTLOOK:
            {economic_summary}
            
            DEMAND SCENARIOS:
            {scenarios_summary}
            
            CUSTOMER SEGMENT ANALYSIS:
            {segments_summary}
            
            Structure the report with:
            1. EXECUTIVE SUMMARY
            2. ECONOMIC ENVIRONMENT IMPACT
            3. DEMAND SCENARIO ANALYSIS
            4. CUSTOMER SEGMENT INSIGHTS
            5. STRATEGIC RECOMMENDATIONS
            6. RISK MITIGATION STRATEGIES
            7. KEY PERFORMANCE INDICATORS TO MONITOR
            
            Focus on actionable insights for inventory management, supply chain optimization,
            and customer targeting strategies. Keep it executive-ready and business-focused.
            """
        )
        
        try:
            # Prepare summaries
            economic_summary = self._summarize_economic_forecasts(economic_forecasts)
            
            scenarios_summary = "\n".join([
                f"• {s.scenario_name} (Probability: {s.probability:.0%}): {s.demand_impact}"
                for s in scenarios
            ])
            
            segments_summary = "\n".join([
                f"• {s.segment_name}: {s.demand_sensitivity} sensitivity to economic changes"
                for s in segments
            ])
            
            prompt = report_template.format(
                economic_summary=economic_summary,
                scenarios_summary=scenarios_summary,
                segments_summary=segments_summary
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating demand planning report: {e}")
            return "Unable to generate demand planning report due to processing error."
    
    def simulate_inventory_impact(self, scenarios: List[DemandScenario],
                                current_inventory: float = 100.0) -> Dict[str, Dict[str, float]]:
        """
        Simulate inventory impact under different demand scenarios.
        
        Args:
            scenarios: Demand scenarios to simulate
            current_inventory: Current inventory level (baseline)
        
        Returns:
            Dictionary with inventory projections for each scenario
        """
        inventory_simulations = {}
        
        for scenario in scenarios:
            # Extract demand impact percentage
            demand_impact_str = scenario.demand_impact.lower()
            
            # Simple parsing of demand impact
            if "increase" in demand_impact_str or "growth" in demand_impact_str:
                if "20%" in demand_impact_str or "high" in demand_impact_str:
                    demand_multiplier = 1.20
                elif "10%" in demand_impact_str or "moderate" in demand_impact_str:
                    demand_multiplier = 1.10
                else:
                    demand_multiplier = 1.05
            elif "decrease" in demand_impact_str or "decline" in demand_impact_str:
                if "20%" in demand_impact_str or "severe" in demand_impact_str:
                    demand_multiplier = 0.80
                elif "10%" in demand_impact_str or "moderate" in demand_impact_str:
                    demand_multiplier = 0.90
                else:
                    demand_multiplier = 0.95
            else:
                demand_multiplier = 1.00  # Stable
            
            # Calculate inventory requirements
            required_inventory = current_inventory * demand_multiplier
            inventory_variance = required_inventory - current_inventory
            
            inventory_simulations[scenario.scenario_name] = {
                'probability': scenario.probability,
                'demand_multiplier': demand_multiplier,
                'required_inventory': required_inventory,
                'inventory_variance': inventory_variance,
                'variance_percentage': (inventory_variance / current_inventory) * 100
            }
        
        return inventory_simulations
    
    def _summarize_economic_forecasts(self, forecasts: Dict[str, Any]) -> str:
        """Create a text summary of economic forecasts."""
        summary_parts = []
        
        for indicator, forecast_data in forecasts.items():
            if isinstance(forecast_data, dict):
                forecast_values = forecast_data.get('forecast', [])
                if len(forecast_values) > 0:
                    start_value = forecast_values[0]
                    end_value = forecast_values[-1]
                    change_pct = ((end_value / start_value - 1) * 100) if start_value != 0 else 0
                    direction = "↑" if change_pct > 1 else "↓" if change_pct < -1 else "→"
                    
                    summary_parts.append(
                        f"{indicator.replace('_', ' ').title()}: {direction} {abs(change_pct):.1f}% change predicted"
                    )
        
        return "\n".join(summary_parts) if summary_parts else "Economic forecasts not available"


def generate_comprehensive_demand_analysis(economic_forecasts: Dict[str, Any],
                                         industry: str = "retail",
                                         customer_context: str = "B2C retail customers") -> Dict[str, Any]:
    """
    Generate comprehensive demand planning analysis using GenAI.
    
    Args:
        economic_forecasts: Economic indicator forecasts
        industry: Industry context
        customer_context: Customer base description
    
    Returns:
        Complete demand planning analysis
    """
    planner = GenAIDemandPlanner()
    
    # Generate demand scenarios
    scenarios = planner.generate_demand_scenarios(economic_forecasts, industry)
    
    # Analyze customer segments
    segments = planner.analyze_customer_segments(economic_forecasts, customer_context)
    
    # Generate comprehensive report
    report = planner.generate_demand_planning_report(economic_forecasts, scenarios, segments)
    
    # Simulate inventory impacts
    inventory_simulations = planner.simulate_inventory_impact(scenarios)
    
    return {
        'demand_scenarios': [s.dict() for s in scenarios],
        'customer_segments': [s.dict() for s in segments],
        'executive_report': report,
        'inventory_simulations': inventory_simulations,
        'analysis_timestamp': datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Example usage
    sample_forecasts = {
        'gdp': {'forecast': [21000, 21100, 21200, 21300], 'model_type': 'ARIMA'},
        'unemployment': {'forecast': [5.0, 4.8, 4.6, 4.4], 'model_type': 'ARIMA'},
        'inflation': {'forecast': [3.0, 2.8, 2.6, 2.4], 'model_type': 'ARIMA'}
    }
    
    analysis = generate_comprehensive_demand_analysis(sample_forecasts)
    print("Demand Planning Analysis Complete:")
    print(f"- Generated {len(analysis['demand_scenarios'])} scenarios")
    print(f"- Analyzed {len(analysis['customer_segments'])} customer segments")
    print(f"- Created executive report and inventory simulations")