"""
Critic Agent

Specialist agent focused on quality assessment, feedback generation,
and iterative improvement of agent outputs.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent, AgentResult


class CriticAgent(BaseAgent):
    """
    Critic agent specialized in quality assessment and feedback.
    
    The critic agent:
    - Evaluates quality of agent outputs
    - Provides constructive feedback and suggestions
    - Identifies areas for improvement
    - Supports iterative refinement processes
    """

    def __init__(self, agent_id: str = "critic-001"):
        super().__init__(
            agent_id=agent_id,
            name="Critic Agent",
            description="Quality assessment specialist for agent output evaluation and improvement"
        )
        self.evaluation_criteria = [
            "accuracy",
            "completeness", 
            "clarity",
            "relevance",
            "coherence",
            "actionability"
        ]

    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process a quality assessment task by evaluating content.
        
        Args:
            task: Quality assessment task with content to evaluate
            
        Returns:
            AgentResult: Quality assessment results and feedback
        """
        start_time = datetime.now()
        
        try:
            # Extract content to evaluate
            content_to_evaluate = task.get("content", "")
            evaluation_context = task.get("context", {})
            task_type = task.get("type", "quality_review")
            
            # Perform multi-dimensional quality evaluation
            quality_assessment = await self._evaluate_quality(content_to_evaluate, evaluation_context)
            
            # Generate specific feedback and suggestions
            feedback = await self._generate_feedback(quality_assessment, content_to_evaluate)
            
            # Create improvement recommendations
            recommendations = await self._generate_recommendations(quality_assessment, feedback)
            
            # Generate comprehensive quality report
            report = await self._generate_quality_report(quality_assessment, feedback, recommendations, task)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on evaluation consistency
            confidence = self._calculate_evaluation_confidence(quality_assessment)
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=report,
                confidence=confidence,
                metadata={
                    "overall_quality_score": quality_assessment.get("overall_score", 0),
                    "criteria_evaluated": len(self.evaluation_criteria),
                    "feedback_points": len(feedback),
                    "recommendations_count": len(recommendations),
                    "processing_time": processing_time,
                    "task_type": task_type
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
                content=f"Quality evaluation failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "processing_time": processing_time},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            self.update_performance_metrics(result, processing_time)
            return result

    async def _evaluate_quality(self, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate content quality across multiple criteria.
        
        Args:
            content: Content to evaluate
            context: Evaluation context and requirements
            
        Returns:
            Quality assessment results
        """
        evaluation = {
            "content_length": len(content),
            "evaluation_timestamp": datetime.now(),
            "criteria_scores": {},
            "detailed_assessment": {}
        }
        
        # Evaluate each criterion
        for criterion in self.evaluation_criteria:
            score, assessment = await self._evaluate_criterion(content, criterion, context)
            evaluation["criteria_scores"][criterion] = score
            evaluation["detailed_assessment"][criterion] = assessment
        
        # Calculate overall quality score
        overall_score = sum(evaluation["criteria_scores"].values()) / len(evaluation["criteria_scores"])
        evaluation["overall_score"] = overall_score
        
        # Determine quality level
        if overall_score >= 0.9:
            evaluation["quality_level"] = "excellent"
        elif overall_score >= 0.8:
            evaluation["quality_level"] = "good"
        elif overall_score >= 0.7:
            evaluation["quality_level"] = "satisfactory"
        elif overall_score >= 0.6:
            evaluation["quality_level"] = "needs_improvement"
        else:
            evaluation["quality_level"] = "poor"
        
        return evaluation

    async def _evaluate_criterion(self, content: str, criterion: str, 
                                context: Dict[str, Any]) -> tuple[float, str]:
        """
        Evaluate content against a specific quality criterion.
        
        Args:
            content: Content to evaluate
            criterion: Quality criterion to assess
            context: Evaluation context
            
        Returns:
            Tuple of (score, detailed_assessment)
        """
        # Simulate criterion-specific evaluation
        if criterion == "accuracy":
            # Check for factual consistency and correctness
            if "data" in content.lower() and "analysis" in content.lower():
                return 0.85, "Content appears to be data-driven with analytical backing"
            elif "research" in content.lower():
                return 0.8, "Content shows research foundation"
            else:
                return 0.7, "Limited evidence of accuracy verification"
        
        elif criterion == "completeness":
            # Assess coverage and thoroughness
            word_count = len(content.split())
            if word_count > 500:
                return 0.9, f"Comprehensive coverage with {word_count} words"
            elif word_count > 200:
                return 0.8, f"Good coverage with {word_count} words"
            else:
                return 0.6, f"Brief content with {word_count} words - may lack completeness"
        
        elif criterion == "clarity":
            # Evaluate readability and structure
            if content.count('\n') > 5:  # Good structure
                return 0.85, "Well-structured with clear sections"
            elif ":" in content or "-" in content:  # Some organization
                return 0.75, "Some organizational structure present"
            else:
                return 0.65, "Could benefit from better organization"
        
        elif criterion == "relevance":
            # Check alignment with requirements
            context_keywords = context.get("keywords", [])
            relevance_score = 0.8  # Base score
            if context_keywords:
                keyword_matches = sum(1 for kw in context_keywords if kw.lower() in content.lower())
                relevance_score = min(0.9, 0.6 + (keyword_matches * 0.1))
            return relevance_score, f"Content relevance score based on keyword alignment: {relevance_score:.1%}"
        
        elif criterion == "coherence":
            # Assess logical flow and consistency
            sentences = content.split('.')
            if len(sentences) > 10:
                return 0.8, "Good logical flow with multiple connected ideas"
            elif len(sentences) > 5:
                return 0.75, "Adequate coherence between ideas"
            else:
                return 0.7, "Limited scope for coherence assessment"
        
        elif criterion == "actionability":
            # Check for practical value and next steps
            action_words = ["recommend", "suggest", "should", "can", "implement", "consider"]
            action_count = sum(1 for word in action_words if word in content.lower())
            if action_count >= 3:
                return 0.85, f"High actionability with {action_count} action-oriented elements"
            elif action_count >= 1:
                return 0.7, f"Some actionable elements present ({action_count})"
            else:
                return 0.5, "Limited actionable insights"
        
        return 0.7, "Standard evaluation applied"

    async def _generate_feedback(self, quality_assessment: Dict[str, Any], 
                               content: str) -> List[Dict[str, Any]]:
        """
        Generate specific feedback based on quality assessment.
        
        Args:
            quality_assessment: Quality evaluation results
            content: Original content
            
        Returns:
            List of feedback items
        """
        feedback = []
        
        # Generate feedback for each criterion
        for criterion, score in quality_assessment["criteria_scores"].items():
            assessment = quality_assessment["detailed_assessment"][criterion]
            
            feedback_item = {
                "criterion": criterion,
                "score": score,
                "assessment": assessment,
                "priority": "high" if score < 0.7 else "medium" if score < 0.8 else "low"
            }
            
            # Add specific suggestions
            if criterion == "accuracy" and score < 0.8:
                feedback_item["suggestion"] = "Consider adding more data sources or citations to improve accuracy"
            elif criterion == "completeness" and score < 0.8:
                feedback_item["suggestion"] = "Expand content to cover additional relevant aspects"
            elif criterion == "clarity" and score < 0.8:
                feedback_item["suggestion"] = "Improve structure with clearer headings and bullet points"
            elif criterion == "relevance" and score < 0.8:
                feedback_item["suggestion"] = "Better align content with specific requirements and objectives"
            elif criterion == "coherence" and score < 0.8:
                feedback_item["suggestion"] = "Improve logical flow between ideas and sections"
            elif criterion == "actionability" and score < 0.8:
                feedback_item["suggestion"] = "Add more specific recommendations and actionable next steps"
            else:
                feedback_item["suggestion"] = "Maintain current quality level"
            
            feedback.append(feedback_item)
        
        return feedback

    async def _generate_recommendations(self, quality_assessment: Dict[str, Any], 
                                      feedback: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate improvement recommendations based on assessment and feedback.
        
        Args:
            quality_assessment: Quality evaluation results
            feedback: Generated feedback items
            
        Returns:
            List of improvement recommendations
        """
        recommendations = []
        
        overall_score = quality_assessment.get("overall_score", 0)
        quality_level = quality_assessment.get("quality_level", "unknown")
        
        # Priority recommendations based on quality level
        if quality_level in ["poor", "needs_improvement"]:
            recommendations.append({
                "priority": "high",
                "category": "overall_quality",
                "title": "Comprehensive Revision Required",
                "description": f"Content quality is {quality_level} (score: {overall_score:.1%}). Significant improvements needed across multiple criteria.",
                "specific_actions": [
                    "Review and enhance content structure",
                    "Add supporting evidence and data",
                    "Improve clarity and readability",
                    "Ensure complete coverage of topic"
                ]
            })
        
        # Criterion-specific recommendations
        high_priority_feedback = [f for f in feedback if f["priority"] == "high"]
        if high_priority_feedback:
            recommendations.append({
                "priority": "high",
                "category": "specific_improvements",
                "title": "Address High-Priority Issues",
                "description": f"{len(high_priority_feedback)} criteria need immediate attention",
                "specific_actions": [f["suggestion"] for f in high_priority_feedback]
            })
        
        # General improvement recommendations
        recommendations.append({
            "priority": "medium",
            "category": "continuous_improvement",
            "title": "Ongoing Enhancement",
            "description": "Regular quality monitoring and iterative improvement",
            "specific_actions": [
                "Schedule regular quality reviews",
                "Gather feedback from stakeholders", 
                "Monitor performance metrics",
                "Update content based on new information"
            ]
        })
        
        return recommendations

    async def _generate_quality_report(self, quality_assessment: Dict[str, Any],
                                     feedback: List[Dict[str, Any]],
                                     recommendations: List[Dict[str, Any]], 
                                     task: Dict[str, Any]) -> str:
        """
        Generate comprehensive quality assessment report.
        
        Args:
            quality_assessment: Quality evaluation results
            feedback: Generated feedback
            recommendations: Improvement recommendations
            task: Original task specification
            
        Returns:
            Formatted quality report
        """
        report_lines = []
        
        # Header
        report_lines.append("QUALITY ASSESSMENT REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Content evaluated: {task.get('description', 'Unknown content')}")
        report_lines.append(f"Assessment completed: {datetime.now()}")
        report_lines.append(f"Content length: {quality_assessment.get('content_length', 0)} characters")
        report_lines.append("")
        
        # Executive Summary
        overall_score = quality_assessment.get("overall_score", 0)
        quality_level = quality_assessment.get("quality_level", "unknown")
        
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(f"Overall Quality Score: {overall_score:.1%}")
        report_lines.append(f"Quality Level: {quality_level.replace('_', ' ').title()}")
        report_lines.append(f"Criteria Evaluated: {len(quality_assessment.get('criteria_scores', {}))}")
        report_lines.append(f"Improvement Areas Identified: {len([f for f in feedback if f['priority'] == 'high'])}")
        report_lines.append("")
        
        # Criteria Scores
        report_lines.append("DETAILED CRITERIA EVALUATION")
        report_lines.append("-" * 35)
        for criterion, score in quality_assessment.get("criteria_scores", {}).items():
            status = "✓" if score >= 0.8 else "⚠" if score >= 0.7 else "✗"
            report_lines.append(f"{status} {criterion.replace('_', ' ').title()}: {score:.1%}")
            
            # Include detailed assessment
            detailed = quality_assessment.get("detailed_assessment", {}).get(criterion, "")
            if detailed:
                report_lines.append(f"   {detailed}")
        report_lines.append("")
        
        # Feedback Summary
        report_lines.append("FEEDBACK SUMMARY")
        report_lines.append("-" * 20)
        for feedback_item in feedback:
            priority_symbol = "🔴" if feedback_item["priority"] == "high" else "🟡" if feedback_item["priority"] == "medium" else "🟢"
            report_lines.append(f"{priority_symbol} {feedback_item['criterion'].replace('_', ' ').title()}")
            report_lines.append(f"   Score: {feedback_item['score']:.1%}")
            report_lines.append(f"   Suggestion: {feedback_item['suggestion']}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("IMPROVEMENT RECOMMENDATIONS")
        report_lines.append("-" * 35)
        for i, rec in enumerate(recommendations, 1):
            priority_symbol = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🟢"
            report_lines.append(f"{i}. {priority_symbol} {rec['title']} ({rec['priority'].upper()} PRIORITY)")
            report_lines.append(f"   {rec['description']}")
            report_lines.append("   Specific Actions:")
            for action in rec.get("specific_actions", []):
                report_lines.append(f"   • {action}")
            report_lines.append("")
        
        # Next Steps
        report_lines.append("RECOMMENDED NEXT STEPS")
        report_lines.append("-" * 25)
        if quality_level in ["poor", "needs_improvement"]:
            report_lines.append("1. Address high-priority issues immediately")
            report_lines.append("2. Implement comprehensive content revision")
            report_lines.append("3. Re-evaluate after improvements")
        else:
            report_lines.append("1. Consider medium-priority improvements")
            report_lines.append("2. Monitor quality over time")
            report_lines.append("3. Maintain current quality standards")
        
        return "\n".join(report_lines)

    def _calculate_evaluation_confidence(self, quality_assessment: Dict[str, Any]) -> float:
        """
        Calculate confidence in the quality evaluation.
        
        Args:
            quality_assessment: Quality evaluation results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence on criteria coverage and score consistency
        criteria_coverage = len(quality_assessment.get("criteria_scores", {})) / len(self.evaluation_criteria)
        
        # Calculate score variance (lower variance = higher confidence)
        scores = list(quality_assessment.get("criteria_scores", {}).values())
        if scores:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            consistency_factor = max(0, 1 - variance)  # Lower variance = higher consistency
        else:
            consistency_factor = 0
        
        # Combine factors
        confidence = (criteria_coverage * 0.6) + (consistency_factor * 0.4)
        
        return min(confidence, 1.0)

    def get_capabilities(self) -> List[str]:
        """Return critic agent capabilities."""
        return [
            "quality_review",
            "quality_assessment",
            "feedback_generation",
            "improvement_recommendations",
            "iterative_refinement",
            "content_evaluation"
        ]

    def get_evaluation_criteria(self) -> List[str]:
        """Return available evaluation criteria."""
        return self.evaluation_criteria.copy()