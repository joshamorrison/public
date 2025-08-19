"""
Feedback Loop

System for implementing feedback loops and continuous improvement
in multi-agent workflows and pattern execution.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback in the system."""
    PERFORMANCE = "performance"
    QUALITY = "quality"
    EFFICIENCY = "efficiency"
    ERROR = "error"
    IMPROVEMENT = "improvement"
    USER = "user"


@dataclass
class FeedbackEntry:
    """Structured feedback entry."""
    feedback_id: str
    feedback_type: FeedbackType
    source: str
    target: str
    content: str
    metrics: Dict[str, float]
    recommendations: List[str]
    priority: str  # high, medium, low
    timestamp: datetime
    processed: bool = False


class FeedbackLoop:
    """
    Feedback loop system for continuous improvement of multi-agent patterns.
    
    The feedback loop:
    - Collects feedback from various sources
    - Analyzes performance and quality metrics
    - Generates improvement recommendations
    - Implements adaptive adjustments
    - Tracks improvement over time
    """

    def __init__(self, loop_id: str = "feedback-loop-001"):
        """
        Initialize the feedback loop system.
        
        Args:
            loop_id: Unique identifier for this feedback loop instance
        """
        self.loop_id = loop_id
        self.name = "Multi-Agent Feedback Loop System"
        
        # Feedback storage
        self.feedback_entries: List[FeedbackEntry] = []
        self.improvement_actions: List[Dict[str, Any]] = []
        self.feedback_handlers: Dict[FeedbackType, List[Callable]] = {
            feedback_type: [] for feedback_type in FeedbackType
        }
        
        # Analytics
        self.feedback_metrics = {
            "total_feedback": 0,
            "processed_feedback": 0,
            "improvement_actions_taken": 0,
            "average_response_time": 0.0,
            "feedback_by_type": {ft.value: 0 for ft in FeedbackType}
        }
        
        # Configuration
        self.auto_process = True
        self.feedback_threshold = 3  # Minimum feedback entries before processing
        self.improvement_cooldown = 300  # Seconds between improvement actions

    async def submit_feedback(self, feedback_type: FeedbackType, source: str, target: str,
                            content: str, metrics: Dict[str, float] = None,
                            recommendations: List[str] = None, priority: str = "medium") -> str:
        """
        Submit feedback to the feedback loop system.
        
        Args:
            feedback_type: Type of feedback
            source: Source of the feedback (agent_id, pattern_id, user, etc.)
            target: Target of the feedback (what it's about)
            content: Feedback content description
            metrics: Optional performance metrics
            recommendations: Optional improvement recommendations
            priority: Priority level (high, medium, low)
            
        Returns:
            Feedback entry ID
        """
        feedback_id = f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}"
        
        feedback_entry = FeedbackEntry(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            source=source,
            target=target,
            content=content,
            metrics=metrics or {},
            recommendations=recommendations or [],
            priority=priority,
            timestamp=datetime.now()
        )
        
        self.feedback_entries.append(feedback_entry)
        
        # Update metrics
        self.feedback_metrics["total_feedback"] += 1
        self.feedback_metrics["feedback_by_type"][feedback_type.value] += 1
        
        print(f"[FEEDBACK_LOOP] Received {feedback_type.value} feedback from {source} about {target}")
        
        # Auto-process if enabled and threshold reached
        if self.auto_process and len([f for f in self.feedback_entries if not f.processed]) >= self.feedback_threshold:
            await self.process_pending_feedback()
        
        return feedback_id

    async def submit_performance_feedback(self, source: str, target: str,
                                        execution_time: float, success_rate: float,
                                        confidence: float, error_count: int = 0) -> str:
        """
        Submit performance-specific feedback.
        
        Args:
            source: Source of the performance feedback
            target: Target being evaluated
            execution_time: Execution time in seconds
            success_rate: Success rate (0.0 to 1.0)
            confidence: Average confidence score (0.0 to 1.0)
            error_count: Number of errors encountered
            
        Returns:
            Feedback entry ID
        """
        metrics = {
            "execution_time": execution_time,
            "success_rate": success_rate,
            "confidence": confidence,
            "error_count": error_count
        }
        
        # Generate recommendations based on metrics
        recommendations = []
        if execution_time > 30:  # Slow execution
            recommendations.append("Consider optimizing agent processing time")
        if success_rate < 0.8:  # Low success rate
            recommendations.append("Review error handling and retry mechanisms")
        if confidence < 0.7:  # Low confidence
            recommendations.append("Improve agent training or add validation steps")
        if error_count > 0:
            recommendations.append("Investigate and fix recurring errors")
        
        # Determine priority based on metrics
        priority = "high" if success_rate < 0.6 or error_count > 3 else "medium" if success_rate < 0.8 else "low"
        
        content = f"Performance evaluation: {execution_time:.1f}s execution, {success_rate:.1%} success rate, {confidence:.1%} confidence"
        
        return await self.submit_feedback(
            FeedbackType.PERFORMANCE, source, target, content, metrics, recommendations, priority
        )

    async def submit_quality_feedback(self, source: str, target: str,
                                    quality_score: float, completeness: float,
                                    accuracy: float, relevance: float) -> str:
        """
        Submit quality-specific feedback.
        
        Args:
            source: Source of the quality feedback
            target: Target being evaluated
            quality_score: Overall quality score (0.0 to 1.0)
            completeness: Completeness score (0.0 to 1.0)
            accuracy: Accuracy score (0.0 to 1.0)
            relevance: Relevance score (0.0 to 1.0)
            
        Returns:
            Feedback entry ID
        """
        metrics = {
            "quality_score": quality_score,
            "completeness": completeness,
            "accuracy": accuracy,
            "relevance": relevance
        }
        
        # Generate recommendations
        recommendations = []
        if completeness < 0.8:
            recommendations.append("Improve content completeness and coverage")
        if accuracy < 0.8:
            recommendations.append("Enhance accuracy through better validation")
        if relevance < 0.8:
            recommendations.append("Improve relevance filtering and context awareness")
        
        priority = "high" if quality_score < 0.6 else "medium" if quality_score < 0.8 else "low"
        
        content = f"Quality evaluation: {quality_score:.1%} overall, {completeness:.1%} complete, {accuracy:.1%} accurate, {relevance:.1%} relevant"
        
        return await self.submit_feedback(
            FeedbackType.QUALITY, source, target, content, metrics, recommendations, priority
        )

    async def process_pending_feedback(self) -> Dict[str, Any]:
        """
        Process all pending feedback entries and generate improvement actions.
        
        Returns:
            Processing results and actions taken
        """
        pending_feedback = [f for f in self.feedback_entries if not f.processed]
        
        if not pending_feedback:
            return {"message": "No pending feedback to process"}
        
        print(f"[FEEDBACK_LOOP] Processing {len(pending_feedback)} pending feedback entries")
        
        processing_start = datetime.now()
        
        # Group feedback by target and type
        feedback_groups = {}
        for feedback in pending_feedback:
            key = (feedback.target, feedback.feedback_type)
            if key not in feedback_groups:
                feedback_groups[key] = []
            feedback_groups[key].append(feedback)
        
        # Process each group
        improvement_actions = []
        for (target, feedback_type), feedback_list in feedback_groups.items():
            action = await self._generate_improvement_action(target, feedback_type, feedback_list)
            if action:
                improvement_actions.append(action)
        
        # Mark feedback as processed
        for feedback in pending_feedback:
            feedback.processed = True
        
        # Update metrics
        self.feedback_metrics["processed_feedback"] += len(pending_feedback)
        self.feedback_metrics["improvement_actions_taken"] += len(improvement_actions)
        
        processing_time = (datetime.now() - processing_start).total_seconds()
        
        # Update average response time
        total_time = self.feedback_metrics["average_response_time"] * (self.feedback_metrics["processed_feedback"] - len(pending_feedback))
        self.feedback_metrics["average_response_time"] = (total_time + processing_time) / self.feedback_metrics["processed_feedback"]
        
        print(f"[FEEDBACK_LOOP] Generated {len(improvement_actions)} improvement actions")
        
        return {
            "processed_feedback_count": len(pending_feedback),
            "improvement_actions_generated": len(improvement_actions),
            "processing_time": processing_time,
            "actions": improvement_actions
        }

    async def _generate_improvement_action(self, target: str, feedback_type: FeedbackType,
                                         feedback_list: List[FeedbackEntry]) -> Optional[Dict[str, Any]]:
        """
        Generate improvement action based on feedback analysis.
        
        Args:
            target: Target of the feedback
            feedback_type: Type of feedback
            feedback_list: List of feedback entries for this target/type
            
        Returns:
            Improvement action or None
        """
        if not feedback_list:
            return None
        
        # Analyze feedback patterns
        high_priority_count = sum(1 for f in feedback_list if f.priority == "high")
        avg_metrics = {}
        all_recommendations = []
        
        # Aggregate metrics
        for feedback in feedback_list:
            for metric, value in feedback.metrics.items():
                if metric not in avg_metrics:
                    avg_metrics[metric] = []
                avg_metrics[metric].append(value)
            
            all_recommendations.extend(feedback.recommendations)
        
        # Calculate averages
        for metric in avg_metrics:
            avg_metrics[metric] = sum(avg_metrics[metric]) / len(avg_metrics[metric])
        
        # Get unique recommendations
        unique_recommendations = list(set(all_recommendations))
        
        # Determine action priority and type
        action_priority = "high" if high_priority_count > 0 else "medium"
        
        # Generate action based on feedback type
        if feedback_type == FeedbackType.PERFORMANCE:
            action_type = "performance_optimization"
            description = f"Optimize performance for {target} based on {len(feedback_list)} feedback entries"
        elif feedback_type == FeedbackType.QUALITY:
            action_type = "quality_improvement"
            description = f"Improve quality for {target} based on {len(feedback_list)} feedback entries"
        elif feedback_type == FeedbackType.ERROR:
            action_type = "error_resolution"
            description = f"Resolve errors for {target} based on {len(feedback_list)} error reports"
        else:
            action_type = "general_improvement"
            description = f"General improvements for {target}"
        
        improvement_action = {
            "action_id": f"action_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:17]}",
            "action_type": action_type,
            "target": target,
            "feedback_type": feedback_type.value,
            "priority": action_priority,
            "description": description,
            "recommendations": unique_recommendations,
            "metrics_analysis": avg_metrics,
            "feedback_count": len(feedback_list),
            "created_at": datetime.now(),
            "status": "pending"
        }
        
        self.improvement_actions.append(improvement_action)
        
        return improvement_action

    def register_feedback_handler(self, feedback_type: FeedbackType, handler: Callable):
        """
        Register a feedback handler for specific feedback types.
        
        Args:
            feedback_type: Type of feedback to handle
            handler: Handler function to call
        """
        self.feedback_handlers[feedback_type].append(handler)
        print(f"[FEEDBACK_LOOP] Registered handler for {feedback_type.value} feedback")

    async def get_feedback_summary(self, target: Optional[str] = None,
                                 feedback_type: Optional[FeedbackType] = None,
                                 days: int = 7) -> Dict[str, Any]:
        """
        Get feedback summary for analysis.
        
        Args:
            target: Optional target filter
            feedback_type: Optional feedback type filter
            days: Number of days to include in summary
            
        Returns:
            Feedback summary and analysis
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Filter feedback
        filtered_feedback = [
            f for f in self.feedback_entries
            if f.timestamp > cutoff_date
        ]
        
        if target:
            filtered_feedback = [f for f in filtered_feedback if f.target == target]
        
        if feedback_type:
            filtered_feedback = [f for f in filtered_feedback if f.feedback_type == feedback_type]
        
        # Analyze feedback
        if not filtered_feedback:
            return {"message": "No feedback found for the specified criteria"}
        
        # Group by various dimensions
        by_type = {}
        by_priority = {}
        by_target = {}
        
        for feedback in filtered_feedback:
            # By type
            fb_type = feedback.feedback_type.value
            if fb_type not in by_type:
                by_type[fb_type] = []
            by_type[fb_type].append(feedback)
            
            # By priority
            if feedback.priority not in by_priority:
                by_priority[feedback.priority] = []
            by_priority[feedback.priority].append(feedback)
            
            # By target
            if feedback.target not in by_target:
                by_target[feedback.target] = []
            by_target[feedback.target].append(feedback)
        
        # Calculate metrics averages
        metrics_summary = {}
        for feedback in filtered_feedback:
            for metric, value in feedback.metrics.items():
                if metric not in metrics_summary:
                    metrics_summary[metric] = []
                metrics_summary[metric].append(value)
        
        # Average the metrics
        for metric in metrics_summary:
            values = metrics_summary[metric]
            metrics_summary[metric] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return {
            "summary_period": f"Last {days} days",
            "total_feedback": len(filtered_feedback),
            "processed_feedback": len([f for f in filtered_feedback if f.processed]),
            "feedback_by_type": {k: len(v) for k, v in by_type.items()},
            "feedback_by_priority": {k: len(v) for k, v in by_priority.items()},
            "feedback_by_target": {k: len(v) for k, v in by_target.items()},
            "metrics_summary": metrics_summary,
            "recent_actions": [
                {
                    "action_id": action["action_id"],
                    "action_type": action["action_type"],
                    "target": action["target"],
                    "priority": action["priority"],
                    "created_at": action["created_at"].isoformat()
                }
                for action in self.improvement_actions[-10:]  # Last 10 actions
            ]
        }

    def get_improvement_recommendations(self, target: str) -> List[Dict[str, Any]]:
        """
        Get improvement recommendations for a specific target.
        
        Args:
            target: Target to get recommendations for
            
        Returns:
            List of recommendations
        """
        target_feedback = [f for f in self.feedback_entries if f.target == target and not f.processed]
        
        if not target_feedback:
            return []
        
        # Collect and prioritize recommendations
        all_recommendations = []
        for feedback in target_feedback:
            for rec in feedback.recommendations:
                all_recommendations.append({
                    "recommendation": rec,
                    "feedback_type": feedback.feedback_type.value,
                    "priority": feedback.priority,
                    "source": feedback.source,
                    "timestamp": feedback.timestamp.isoformat()
                })
        
        # Sort by priority (high first) and timestamp (recent first)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        all_recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 3), x["timestamp"]), 
            reverse=True
        )
        
        return all_recommendations

    def get_feedback_metrics(self) -> Dict[str, Any]:
        """Get comprehensive feedback loop metrics."""
        return {
            "loop_id": self.loop_id,
            "feedback_metrics": self.feedback_metrics.copy(),
            "current_state": {
                "total_feedback_entries": len(self.feedback_entries),
                "pending_feedback": len([f for f in self.feedback_entries if not f.processed]),
                "improvement_actions": len(self.improvement_actions),
                "pending_actions": len([a for a in self.improvement_actions if a["status"] == "pending"])
            },
            "configuration": {
                "auto_process": self.auto_process,
                "feedback_threshold": self.feedback_threshold,
                "improvement_cooldown": self.improvement_cooldown
            }
        }

    def clear_feedback_history(self):
        """Clear feedback history and metrics."""
        self.feedback_entries.clear()
        self.improvement_actions.clear()
        
        self.feedback_metrics = {
            "total_feedback": 0,
            "processed_feedback": 0,
            "improvement_actions_taken": 0,
            "average_response_time": 0.0,
            "feedback_by_type": {ft.value: 0 for ft in FeedbackType}
        }
        
        print("[FEEDBACK_LOOP] Feedback history cleared")