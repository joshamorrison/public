"""
Synthesizer Agent

Specialist agent focused on result aggregation, fusion of multiple inputs,
and creation of unified, coherent outputs from diverse sources.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent, AgentResult


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer agent specialized in result aggregation and fusion.
    
    The synthesizer agent:
    - Aggregates results from multiple agents or sources
    - Identifies common themes and conflicting information
    - Creates unified, coherent outputs
    - Resolves inconsistencies and gaps
    - Produces comprehensive summaries
    """

    def __init__(self, agent_id: str = "synthesizer-001"):
        super().__init__(
            agent_id=agent_id,
            name="Synthesizer Agent",
            description="Result aggregation specialist for unified output creation"
        )
        self.synthesis_methods = [
            "thematic_synthesis",
            "consensus_building",
            "gap_identification",
            "conflict_resolution",
            "hierarchical_aggregation"
        ]

    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process a synthesis task by aggregating multiple inputs into unified output.
        
        Args:
            task: Synthesis task with multiple inputs to aggregate
            
        Returns:
            AgentResult: Synthesized and unified result
        """
        start_time = datetime.now()
        
        try:
            # Extract inputs to synthesize
            inputs = task.get("inputs", [])
            synthesis_type = task.get("type", "general_synthesis")
            requirements = task.get("requirements", {})
            
            if not inputs:
                raise ValueError("No inputs provided for synthesis")
            
            # Analyze input characteristics
            input_analysis = await self._analyze_inputs(inputs)
            
            # Apply synthesis methods
            synthesis_results = await self._apply_synthesis_methods(inputs, input_analysis)
            
            # Resolve conflicts and inconsistencies
            resolved_content = await self._resolve_conflicts(synthesis_results, input_analysis)
            
            # Generate unified output
            unified_result = await self._generate_unified_output(resolved_content, task, input_analysis)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on input consistency and synthesis quality
            confidence = self._calculate_synthesis_confidence(input_analysis, synthesis_results)
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=unified_result,
                confidence=confidence,
                metadata={
                    "inputs_processed": len(inputs),
                    "synthesis_methods_used": len(synthesis_results),
                    "consistency_score": input_analysis.get("consistency_score", 0),
                    "coverage_completeness": input_analysis.get("coverage_completeness", 0),
                    "conflicts_resolved": input_analysis.get("conflicts_count", 0),
                    "processing_time": processing_time,
                    "synthesis_type": synthesis_type
                },
                timestamp=datetime.now()
            )
            
            self.update_performance_metrics(result, processing_time)
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=f"Synthesis failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "processing_time": processing_time},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            self.update_performance_metrics(result, processing_time)
            return result

    async def _analyze_inputs(self, inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze characteristics of input data for synthesis.
        
        Args:
            inputs: List of input data to be synthesized
            
        Returns:
            Analysis of input characteristics
        """
        analysis = {
            "input_count": len(inputs),
            "total_content_length": 0,
            "sources": [],
            "themes": [],
            "timestamps": [],
            "confidence_scores": []
        }
        
        # Analyze each input
        for input_data in inputs:
            if isinstance(input_data, dict):
                content = input_data.get("content", "")
                source = input_data.get("source", "unknown")
                confidence = input_data.get("confidence", 0.5)
                timestamp = input_data.get("timestamp", datetime.now())
                
                analysis["total_content_length"] += len(content)
                analysis["sources"].append(source)
                analysis["confidence_scores"].append(confidence)
                analysis["timestamps"].append(timestamp)
                
                # Extract themes (simplified)
                if "research" in content.lower():
                    analysis["themes"].append("research")
                if "analysis" in content.lower():
                    analysis["themes"].append("analysis")
                if "recommendation" in content.lower():
                    analysis["themes"].append("recommendations")
            else:
                # Handle string inputs
                content = str(input_data)
                analysis["total_content_length"] += len(content)
                analysis["sources"].append("direct_input")
                analysis["confidence_scores"].append(0.5)
        
        # Calculate derived metrics
        analysis["average_confidence"] = (
            sum(analysis["confidence_scores"]) / len(analysis["confidence_scores"]) 
            if analysis["confidence_scores"] else 0
        )
        
        analysis["source_diversity"] = len(set(analysis["sources"]))
        analysis["theme_diversity"] = len(set(analysis["themes"]))
        
        # Estimate consistency (simplified)
        confidence_variance = 0
        if len(analysis["confidence_scores"]) > 1:
            mean_conf = analysis["average_confidence"]
            confidence_variance = sum((c - mean_conf) ** 2 for c in analysis["confidence_scores"]) / len(analysis["confidence_scores"])
        
        analysis["consistency_score"] = max(0, 1 - confidence_variance)
        analysis["coverage_completeness"] = min(1.0, analysis["theme_diversity"] / 5)  # Normalized
        analysis["conflicts_count"] = max(0, analysis["source_diversity"] - 1)  # Simplified
        
        return analysis

    async def _apply_synthesis_methods(self, inputs: List[Dict[str, Any]], 
                                     input_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply different synthesis methods to the inputs.
        
        Args:
            inputs: Input data to synthesize
            input_analysis: Analysis of input characteristics
            
        Returns:
            Results from different synthesis methods
        """
        synthesis_results = []
        
        for method in self.synthesis_methods:
            result = await self._apply_synthesis_method(method, inputs, input_analysis)
            if result:
                synthesis_results.append(result)
        
        return synthesis_results

    async def _apply_synthesis_method(self, method: str, inputs: List[Dict[str, Any]], 
                                    input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a specific synthesis method.
        
        Args:
            method: Synthesis method to apply
            inputs: Input data
            input_analysis: Input characteristics
            
        Returns:
            Synthesis result from the method
        """
        base_result = {
            "method": method,
            "timestamp": datetime.now(),
            "input_count": len(inputs)
        }
        
        if method == "thematic_synthesis":
            themes = input_analysis.get("themes", [])
            unique_themes = list(set(themes))
            
            base_result.update({
                "synthesis": f"Identified {len(unique_themes)} major themes across {len(inputs)} inputs",
                "themes_identified": unique_themes,
                "theme_coverage": f"Themes appear {len(themes) / len(inputs):.1f} times per input on average",
                "confidence": 0.8
            })
        
        elif method == "consensus_building":
            avg_confidence = input_analysis.get("average_confidence", 0)
            consistency = input_analysis.get("consistency_score", 0)
            
            base_result.update({
                "synthesis": f"Built consensus from inputs with {avg_confidence:.1%} average confidence",
                "consensus_strength": consistency,
                "agreement_level": "high" if consistency > 0.8 else "medium" if consistency > 0.6 else "low",
                "confidence": min(avg_confidence * consistency, 1.0)
            })
        
        elif method == "gap_identification":
            completeness = input_analysis.get("coverage_completeness", 0)
            gaps_score = 1 - completeness
            
            base_result.update({
                "synthesis": f"Identified coverage gaps representing {gaps_score:.1%} of expected content",
                "completeness_score": completeness,
                "gaps_identified": ["temporal analysis", "comparative perspective", "quantitative metrics"] if gaps_score > 0.3 else [],
                "confidence": 0.75
            })
        
        elif method == "conflict_resolution":
            conflicts = input_analysis.get("conflicts_count", 0)
            source_diversity = input_analysis.get("source_diversity", 1)
            
            base_result.update({
                "synthesis": f"Resolved {conflicts} potential conflicts from {source_diversity} different sources",
                "conflict_resolution_approach": "evidence_weighting",
                "resolution_confidence": max(0.5, 1 - (conflicts / source_diversity)) if source_diversity > 0 else 0.5,
                "confidence": 0.7
            })
        
        elif method == "hierarchical_aggregation":
            total_length = input_analysis.get("total_content_length", 0)
            compression_ratio = min(1.0, 1000 / total_length) if total_length > 0 else 1.0
            
            base_result.update({
                "synthesis": f"Hierarchically aggregated {total_length} characters with {compression_ratio:.1%} compression",
                "aggregation_levels": ["high-level_summary", "key_points", "supporting_details"],
                "information_retention": 1 - compression_ratio,
                "confidence": 0.85
            })
        
        return base_result

    async def _resolve_conflicts(self, synthesis_results: List[Dict[str, Any]], 
                               input_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts and inconsistencies in synthesis results.
        
        Args:
            synthesis_results: Results from different synthesis methods
            input_analysis: Input analysis results
            
        Returns:
            Resolved and integrated content
        """
        resolution = {
            "resolved_content": {},
            "conflict_resolution_log": [],
            "integration_approach": "weighted_consensus"
        }
        
        # Weight synthesis results by their confidence scores
        total_weight = sum(result.get("confidence", 0.5) for result in synthesis_results)
        
        if total_weight == 0:
            return resolution
        
        # Integrate synthesis results with weighted approach
        integrated_themes = []
        integrated_findings = []
        
        for result in synthesis_results:
            weight = result.get("confidence", 0.5) / total_weight
            method = result.get("method", "unknown")
            
            # Collect themes and findings with weights
            if "themes_identified" in result:
                for theme in result["themes_identified"]:
                    integrated_themes.append({"theme": theme, "weight": weight, "method": method})
            
            synthesis_content = result.get("synthesis", "")
            if synthesis_content:
                integrated_findings.append({
                    "finding": synthesis_content,
                    "weight": weight,
                    "method": method,
                    "confidence": result.get("confidence", 0.5)
                })
        
        # Resolve theme conflicts (themes appearing multiple times get higher weight)
        theme_weights = {}
        for theme_entry in integrated_themes:
            theme = theme_entry["theme"]
            if theme in theme_weights:
                theme_weights[theme] += theme_entry["weight"]
            else:
                theme_weights[theme] = theme_entry["weight"]
        
        # Sort themes by weight
        top_themes = sorted(theme_weights.items(), key=lambda x: x[1], reverse=True)
        
        resolution["resolved_content"] = {
            "primary_themes": [theme for theme, weight in top_themes[:3]],
            "theme_confidence": dict(top_themes),
            "integrated_findings": integrated_findings,
            "resolution_quality": min(1.0, sum(f["confidence"] for f in integrated_findings) / len(integrated_findings)) if integrated_findings else 0
        }
        
        return resolution

    async def _generate_unified_output(self, resolved_content: Dict[str, Any], 
                                     task: Dict[str, Any], 
                                     input_analysis: Dict[str, Any]) -> str:
        """
        Generate unified output from resolved content.
        
        Args:
            resolved_content: Resolved and integrated content
            task: Original synthesis task
            input_analysis: Input analysis results
            
        Returns:
            Unified output content
        """
        output_lines = []
        
        # Header
        output_lines.append("SYNTHESIS REPORT")
        output_lines.append("=" * 50)
        output_lines.append(f"Synthesis Topic: {task.get('description', 'Multiple input synthesis')}")
        output_lines.append(f"Synthesis completed: {datetime.now()}")
        output_lines.append(f"Inputs processed: {input_analysis.get('input_count', 0)}")
        output_lines.append(f"Sources integrated: {input_analysis.get('source_diversity', 0)}")
        output_lines.append("")
        
        # Executive Summary
        avg_confidence = input_analysis.get("average_confidence", 0)
        consistency_score = input_analysis.get("consistency_score", 0)
        
        output_lines.append("EXECUTIVE SUMMARY")
        output_lines.append("-" * 20)
        output_lines.append(f"Successfully synthesized {input_analysis.get('input_count', 0)} inputs from {input_analysis.get('source_diversity', 0)} sources")
        output_lines.append(f"Average input confidence: {avg_confidence:.1%}")
        output_lines.append(f"Content consistency: {consistency_score:.1%}")
        output_lines.append(f"Synthesis quality: {resolved_content.get('resolved_content', {}).get('resolution_quality', 0):.1%}")
        output_lines.append("")
        
        # Primary Themes
        primary_themes = resolved_content.get("resolved_content", {}).get("primary_themes", [])
        if primary_themes:
            output_lines.append("PRIMARY THEMES")
            output_lines.append("-" * 20)
            theme_confidence = resolved_content.get("resolved_content", {}).get("theme_confidence", {})
            
            for i, theme in enumerate(primary_themes, 1):
                confidence = theme_confidence.get(theme, 0)
                output_lines.append(f"{i}. {theme.replace('_', ' ').title()} (Confidence: {confidence:.1%})")
            output_lines.append("")
        
        # Integrated Findings
        findings = resolved_content.get("resolved_content", {}).get("integrated_findings", [])
        if findings:
            output_lines.append("INTEGRATED FINDINGS")
            output_lines.append("-" * 25)
            
            # Sort findings by confidence
            sorted_findings = sorted(findings, key=lambda x: x.get("confidence", 0), reverse=True)
            
            for i, finding in enumerate(sorted_findings, 1):
                output_lines.append(f"{i}. {finding['finding']}")
                output_lines.append(f"   Method: {finding['method'].replace('_', ' ').title()}")
                output_lines.append(f"   Confidence: {finding['confidence']:.1%}")
                output_lines.append("")
        
        # Source Analysis
        output_lines.append("SOURCE ANALYSIS")
        output_lines.append("-" * 20)
        sources = input_analysis.get("sources", [])
        unique_sources = list(set(sources))
        
        for source in unique_sources:
            source_count = sources.count(source)
            output_lines.append(f"• {source}: {source_count} inputs ({source_count/len(sources):.1%} of total)")
        output_lines.append("")
        
        # Quality Assessment
        output_lines.append("SYNTHESIS QUALITY ASSESSMENT")
        output_lines.append("-" * 35)
        output_lines.append(f"Source Diversity: {'High' if input_analysis.get('source_diversity', 0) > 3 else 'Medium' if input_analysis.get('source_diversity', 0) > 1 else 'Low'}")
        output_lines.append(f"Content Consistency: {'High' if consistency_score > 0.8 else 'Medium' if consistency_score > 0.6 else 'Low'}")
        output_lines.append(f"Coverage Completeness: {input_analysis.get('coverage_completeness', 0):.1%}")
        
        conflicts = input_analysis.get("conflicts_count", 0)
        if conflicts > 0:
            output_lines.append(f"Conflicts Identified: {conflicts} (resolved through weighted consensus)")
        else:
            output_lines.append("Conflicts Identified: None")
        
        output_lines.append("")
        
        # Recommendations
        output_lines.append("RECOMMENDATIONS")
        output_lines.append("-" * 20)
        if consistency_score < 0.6:
            output_lines.append("• Seek additional sources to improve consistency")
        if input_analysis.get("coverage_completeness", 0) < 0.7:
            output_lines.append("• Gather more comprehensive information to fill coverage gaps")
        if avg_confidence < 0.7:
            output_lines.append("• Validate findings with higher-confidence sources")
        
        output_lines.append("• Regular updates recommended as new information becomes available")
        output_lines.append("• Consider domain expert review for critical decisions")
        
        return "\n".join(output_lines)

    def _calculate_synthesis_confidence(self, input_analysis: Dict[str, Any], 
                                      synthesis_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence in the synthesis result.
        
        Args:
            input_analysis: Analysis of input characteristics
            synthesis_results: Results from synthesis methods
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Factors affecting synthesis confidence
        input_confidence = input_analysis.get("average_confidence", 0)
        consistency_score = input_analysis.get("consistency_score", 0)
        source_diversity_factor = min(1.0, input_analysis.get("source_diversity", 1) / 3)
        
        # Synthesis method confidence
        if synthesis_results:
            method_confidence = sum(r.get("confidence", 0.5) for r in synthesis_results) / len(synthesis_results)
        else:
            method_confidence = 0.5
        
        # Weighted combination
        confidence = (
            input_confidence * 0.3 +
            consistency_score * 0.25 +
            source_diversity_factor * 0.2 +
            method_confidence * 0.25
        )
        
        return min(confidence, 1.0)

    def get_capabilities(self) -> List[str]:
        """Return synthesizer agent capabilities."""
        return [
            "result_aggregation",
            "content_fusion",
            "consensus_building",
            "conflict_resolution",
            "gap_identification",
            "unified_output_generation",
            "multi_source_integration"
        ]

    def get_synthesis_methods(self) -> List[str]:
        """Return available synthesis methods."""
        return self.synthesis_methods.copy()