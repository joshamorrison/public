"""
Cost Monitoring

Tracks token usage, API costs, and resource consumption across multi-agent workflows.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class CostCategory(Enum):
    """Categories of costs to track."""
    LLM_TOKENS = "llm_tokens"
    API_CALLS = "api_calls" 
    COMPUTE_TIME = "compute_time"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class CostEvent:
    """Individual cost event."""
    category: CostCategory
    amount: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None


@dataclass
class TokenUsage:
    """Token usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens


class TokenUsageTracker:
    """
    Tracks token usage across different LLM providers and models.
    """
    
    def __init__(self):
        """Initialize token usage tracker."""
        self.usage_by_model: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self.usage_by_agent: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self.usage_history: List[Dict[str, Any]] = []
        
        # Pricing per 1K tokens (approximate, update with current rates)
        self.token_prices = {
            # OpenAI pricing (per 1K tokens)
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
            
            # Anthropic pricing (per 1K tokens)
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            
            # AWS Bedrock pricing (per 1K tokens)
            "anthropic.claude-3-opus": {"input": 0.015, "output": 0.075},
            "anthropic.claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "anthropic.claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "amazon.titan-text-express": {"input": 0.0008, "output": 0.0016},
        }
    
    def record_usage(self, model: str, prompt_tokens: int, completion_tokens: int,
                    agent_id: str = None, workflow_id: str = None):
        """Record token usage for a model."""
        usage = TokenUsage(prompt_tokens, completion_tokens)
        
        # Update model usage
        model_usage = self.usage_by_model[model]
        model_usage.prompt_tokens += prompt_tokens
        model_usage.completion_tokens += completion_tokens
        model_usage.total_tokens += usage.total_tokens
        
        # Update agent usage
        if agent_id:
            agent_usage = self.usage_by_agent[agent_id]
            agent_usage.prompt_tokens += prompt_tokens
            agent_usage.completion_tokens += completion_tokens
            agent_usage.total_tokens += usage.total_tokens
        
        # Record history
        self.usage_history.append({
            "timestamp": datetime.now(),
            "model": model,
            "agent_id": agent_id,
            "workflow_id": workflow_id,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.total_tokens,
            "estimated_cost": self.calculate_cost(model, prompt_tokens, completion_tokens)
        })
    
    def calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost for token usage."""
        if model not in self.token_prices:
            # Default pricing if model not found
            return (prompt_tokens + completion_tokens) * 0.002 / 1000
        
        pricing = self.token_prices[model]
        input_cost = (prompt_tokens / 1000) * pricing["input"]
        output_cost = (completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def get_usage_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get token usage summary for time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_usage = [
            u for u in self.usage_history
            if u["timestamp"] >= cutoff_time
        ]
        
        if not recent_usage:
            return {"error": "No recent usage data"}
        
        total_tokens = sum(u["total_tokens"] for u in recent_usage)
        total_cost = sum(u["estimated_cost"] for u in recent_usage)
        
        # Usage by model
        model_usage = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        for usage in recent_usage:
            model = usage["model"]
            model_usage[model]["tokens"] += usage["total_tokens"]
            model_usage[model]["cost"] += usage["estimated_cost"]
        
        # Usage by agent
        agent_usage = defaultdict(lambda: {"tokens": 0, "cost": 0.0})
        for usage in recent_usage:
            if usage["agent_id"]:
                agent_id = usage["agent_id"]
                agent_usage[agent_id]["tokens"] += usage["total_tokens"]
                agent_usage[agent_id]["cost"] += usage["estimated_cost"]
        
        return {
            "period_hours": hours,
            "total_tokens": total_tokens,
            "total_estimated_cost": round(total_cost, 4),
            "usage_events": len(recent_usage),
            "by_model": dict(model_usage),
            "by_agent": dict(agent_usage),
            "avg_cost_per_event": round(total_cost / len(recent_usage), 4) if recent_usage else 0
        }
    
    def get_cost_projection(self, hours: int = 24) -> Dict[str, Any]:
        """Project costs based on recent usage patterns."""
        summary = self.get_usage_summary(hours)
        if "error" in summary:
            return summary
        
        hourly_cost = summary["total_estimated_cost"] / hours
        
        return {
            "current_hourly_rate": round(hourly_cost, 4),
            "projected_daily": round(hourly_cost * 24, 2),
            "projected_weekly": round(hourly_cost * 24 * 7, 2),
            "projected_monthly": round(hourly_cost * 24 * 30, 2),
            "based_on_hours": hours
        }


class CostMonitor:
    """
    Comprehensive cost monitoring for multi-agent workflows.
    
    Tracks various types of costs including tokens, API calls, compute time, etc.
    """
    
    def __init__(self):
        """Initialize cost monitor."""
        self.cost_events: List[CostEvent] = []
        self.token_tracker = TokenUsageTracker()
        self.budgets: Dict[str, float] = {}
        self.alerts: List[Dict[str, Any]] = []
    
    def record_cost(self, category: CostCategory, amount: float, unit: str,
                   agent_id: str = None, workflow_id: str = None, **metadata):
        """Record a cost event."""
        event = CostEvent(
            category=category,
            amount=amount,
            unit=unit,
            timestamp=datetime.now(),
            agent_id=agent_id,
            workflow_id=workflow_id,
            metadata=metadata
        )
        
        self.cost_events.append(event)
        self._check_budgets()
    
    def record_llm_usage(self, model: str, prompt_tokens: int, completion_tokens: int,
                        agent_id: str = None, workflow_id: str = None):
        """Record LLM token usage and associated costs."""
        # Track tokens
        self.token_tracker.record_usage(
            model, prompt_tokens, completion_tokens, agent_id, workflow_id
        )
        
        # Record cost event
        cost = self.token_tracker.calculate_cost(model, prompt_tokens, completion_tokens)
        self.record_cost(
            category=CostCategory.LLM_TOKENS,
            amount=cost,
            unit="USD",
            agent_id=agent_id,
            workflow_id=workflow_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
    
    def record_api_call(self, api_name: str, cost: float, agent_id: str = None,
                       workflow_id: str = None, **metadata):
        """Record API call cost."""
        self.record_cost(
            category=CostCategory.API_CALLS,
            amount=cost,
            unit="USD",
            agent_id=agent_id,
            workflow_id=workflow_id,
            api_name=api_name,
            **metadata
        )
    
    def record_compute_time(self, duration_seconds: float, cost_per_second: float = 0.0001,
                           agent_id: str = None, workflow_id: str = None, **metadata):
        """Record compute time costs."""
        cost = duration_seconds * cost_per_second
        self.record_cost(
            category=CostCategory.COMPUTE_TIME,
            amount=cost,
            unit="USD",
            agent_id=agent_id,
            workflow_id=workflow_id,
            duration_seconds=duration_seconds,
            **metadata
        )
    
    def set_budget(self, category: str, amount: float, period: str = "daily"):
        """Set budget limit for a category."""
        self.budgets[f"{category}_{period}"] = amount
    
    def _check_budgets(self):
        """Check if any budgets are exceeded."""
        for budget_key, limit in self.budgets.items():
            category, period = budget_key.rsplit("_", 1)
            
            # Calculate time period
            if period == "daily":
                cutoff = datetime.now() - timedelta(days=1)
            elif period == "weekly":
                cutoff = datetime.now() - timedelta(weeks=1)
            elif period == "monthly":
                cutoff = datetime.now() - timedelta(days=30)
            else:
                continue
            
            # Calculate spending in period
            recent_events = [
                e for e in self.cost_events
                if e.timestamp >= cutoff and (category == "total" or e.category.value == category)
            ]
            
            total_spent = sum(e.amount for e in recent_events)
            
            # Check for budget exceeded
            if total_spent > limit:
                alert = {
                    "type": "budget_exceeded",
                    "category": category,
                    "period": period,
                    "limit": limit,
                    "spent": total_spent,
                    "timestamp": datetime.now()
                }
                self.alerts.append(alert)
    
    def get_cost_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive cost summary."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_events = [
            e for e in self.cost_events
            if e.timestamp >= cutoff_time
        ]
        
        if not recent_events:
            return {"error": "No recent cost data"}
        
        # Total costs
        total_cost = sum(e.amount for e in recent_events)
        
        # Costs by category
        category_costs = defaultdict(float)
        for event in recent_events:
            category_costs[event.category.value] += event.amount
        
        # Costs by agent
        agent_costs = defaultdict(float)
        for event in recent_events:
            if event.agent_id:
                agent_costs[event.agent_id] += event.amount
        
        # Costs by workflow
        workflow_costs = defaultdict(float)
        for event in recent_events:
            if event.workflow_id:
                workflow_costs[event.workflow_id] += event.amount
        
        return {
            "period_hours": hours,
            "total_cost": round(total_cost, 4),
            "cost_events": len(recent_events),
            "by_category": dict(category_costs),
            "by_agent": dict(agent_costs),
            "by_workflow": dict(workflow_costs),
            "token_usage": self.token_tracker.get_usage_summary(hours),
            "projections": self.token_tracker.get_cost_projection(hours),
            "active_alerts": len([a for a in self.alerts if 
                                (datetime.now() - a["timestamp"]).hours < 24])
        }
    
    def get_cost_breakdown(self, category: str = None, agent_id: str = None,
                          workflow_id: str = None, hours: int = 24) -> Dict[str, Any]:
        """Get detailed cost breakdown with filters."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Apply filters
        filtered_events = [
            e for e in self.cost_events
            if e.timestamp >= cutoff_time
        ]
        
        if category:
            filtered_events = [e for e in filtered_events if e.category.value == category]
        if agent_id:
            filtered_events = [e for e in filtered_events if e.agent_id == agent_id]
        if workflow_id:
            filtered_events = [e for e in filtered_events if e.workflow_id == workflow_id]
        
        if not filtered_events:
            return {"error": "No matching cost events"}
        
        # Detailed breakdown
        events_by_hour = defaultdict(float)
        for event in filtered_events:
            hour_key = event.timestamp.strftime("%Y-%m-%d %H:00")
            events_by_hour[hour_key] += event.amount
        
        return {
            "filters": {
                "category": category,
                "agent_id": agent_id,
                "workflow_id": workflow_id,
                "hours": hours
            },
            "total_cost": round(sum(e.amount for e in filtered_events), 4),
            "event_count": len(filtered_events),
            "hourly_breakdown": dict(events_by_hour),
            "average_per_event": round(
                sum(e.amount for e in filtered_events) / len(filtered_events), 6
            ) if filtered_events else 0
        }
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status."""
        status = {}
        
        for budget_key, limit in self.budgets.items():
            category, period = budget_key.rsplit("_", 1)
            
            # Calculate period
            if period == "daily":
                cutoff = datetime.now() - timedelta(days=1)
            elif period == "weekly":
                cutoff = datetime.now() - timedelta(weeks=1)
            elif period == "monthly":
                cutoff = datetime.now() - timedelta(days=30)
            else:
                continue
            
            # Calculate spending
            recent_events = [
                e for e in self.cost_events
                if e.timestamp >= cutoff and (category == "total" or e.category.value == category)
            ]
            
            spent = sum(e.amount for e in recent_events)
            remaining = limit - spent
            utilization = (spent / limit) * 100 if limit > 0 else 0
            
            status[budget_key] = {
                "limit": limit,
                "spent": round(spent, 4),
                "remaining": round(remaining, 4),
                "utilization_percent": round(utilization, 1),
                "exceeded": spent > limit
            }
        
        return status
    
    def export_cost_data(self, format: str = "json") -> str:
        """Export cost data."""
        if format == "json":
            return json.dumps({
                "summary": self.get_cost_summary(24),
                "budget_status": self.get_budget_status(),
                "recent_alerts": self.alerts[-10:],  # Last 10 alerts
                "token_details": self.token_tracker.get_usage_summary(24),
                "exported_at": datetime.now().isoformat()
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old cost data."""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Remove old events
        self.cost_events = [
            e for e in self.cost_events
            if e.timestamp >= cutoff_time
        ]
        
        # Remove old token usage history
        self.token_tracker.usage_history = [
            u for u in self.token_tracker.usage_history
            if u["timestamp"] >= cutoff_time
        ]
        
        # Remove old alerts
        self.alerts = [
            a for a in self.alerts
            if a["timestamp"] >= cutoff_time
        ]