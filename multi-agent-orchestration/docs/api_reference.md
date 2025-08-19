# API Reference

## 🌐 Overview

The Multi-Agent Orchestration Platform provides a comprehensive REST API for managing agents, executing workflows, and monitoring performance. The API is built with FastAPI and provides automatic OpenAPI documentation.

## 🚀 Quick Start

### **Base URL**
```
# Local Development
http://localhost:8000

# Production
https://your-domain.com/api/v1
```

### **Authentication**
```bash
# API Key Authentication
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     "http://localhost:8000/api/v1/agents"
```

### **Interactive Documentation**
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 📋 Core Endpoints

### **Health Check**
```bash
GET /health
```
**Response**:
```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "version": "1.0.0",
    "components": {
        "database": "healthy",
        "redis": "healthy",
        "langsmith": "healthy"
    }
}
```

## 🤖 Agent Management

### **List Agents**
```bash
GET /api/v1/agents
```
**Response**:
```json
{
    "agents": [
        {
            "id": "agent-123",
            "name": "research-agent",
            "type": "ResearchAgent",
            "status": "active",
            "capabilities": ["web_search", "document_analysis"],
            "created_at": "2024-01-15T09:00:00Z"
        }
    ],
    "total": 1
}
```

### **Create Agent**
```bash
POST /api/v1/agents
Content-Type: application/json

{
    "name": "custom-analyst",
    "type": "AnalysisAgent",
    "config": {
        "model": "gpt-4",
        "temperature": 0.7,
        "tools": ["calculator", "data_analyzer"]
    }
}
```
**Response**:
```json
{
    "id": "agent-456",
    "name": "custom-analyst",
    "type": "AnalysisAgent",
    "status": "created",
    "config": {
        "model": "gpt-4",
        "temperature": 0.7,
        "tools": ["calculator", "data_analyzer"]
    }
}
```

### **Get Agent Details**
```bash
GET /api/v1/agents/{agent_id}
```

### **Update Agent**
```bash
PUT /api/v1/agents/{agent_id}
Content-Type: application/json

{
    "config": {
        "temperature": 0.5
    }
}
```

### **Delete Agent**
```bash
DELETE /api/v1/agents/{agent_id}
```

## 🔄 Workflow Execution

### **Execute Pipeline Workflow**
```bash
POST /api/v1/workflows/pipeline
Content-Type: application/json

{
    "name": "content-creation-pipeline",
    "agents": [
        {
            "id": "research-agent",
            "task": "Research AI trends in healthcare"
        },
        {
            "id": "analysis-agent", 
            "task": "Analyze research findings"
        },
        {
            "id": "summary-agent",
            "task": "Create executive summary"
        }
    ],
    "config": {
        "timeout": 300,
        "quality_threshold": 0.8
    }
}
```

**Response**:
```json
{
    "workflow_id": "workflow-789",
    "status": "running",
    "pattern": "pipeline",
    "created_at": "2024-01-15T10:30:00Z",
    "estimated_completion": "2024-01-15T10:35:00Z"
}
```

### **Execute Supervisor Workflow**
```bash
POST /api/v1/workflows/supervisor
Content-Type: application/json

{
    "name": "market-research-coordination",
    "supervisor": "supervisor-agent",
    "specialists": [
        "market-researcher",
        "competitive-analyst", 
        "financial-analyst"
    ],
    "task": {
        "description": "Comprehensive analysis of fintech market",
        "requirements": [
            "Market size and growth trends",
            "Competitive landscape analysis", 
            "Investment opportunities"
        ]
    }
}
```

### **Execute Parallel Workflow**
```bash
POST /api/v1/workflows/parallel
Content-Type: application/json

{
    "name": "concurrent-analysis",
    "agents": [
        {
            "id": "agent-1",
            "task": "Analyze competitor A"
        },
        {
            "id": "agent-2", 
            "task": "Analyze competitor B"
        },
        {
            "id": "agent-3",
            "task": "Analyze market trends"
        }
    ],
    "fusion_strategy": "weighted_average",
    "weights": [0.4, 0.4, 0.2]
}
```

### **Execute Reflective Workflow**
```bash
POST /api/v1/workflows/reflective
Content-Type: application/json

{
    "name": "iterative-strategy-development",
    "agent": "strategy-agent",
    "task": "Develop go-to-market strategy",
    "reflection_config": {
        "max_iterations": 3,
        "improvement_threshold": 0.1,
        "critics": ["quality-critic", "feasibility-critic"]
    }
}
```

## 📊 Workflow Monitoring

### **Get Workflow Status**
```bash
GET /api/v1/workflows/{workflow_id}
```
**Response**:
```json
{
    "id": "workflow-789",
    "name": "content-creation-pipeline",
    "status": "completed",
    "pattern": "pipeline", 
    "progress": {
        "current_stage": 3,
        "total_stages": 3,
        "completion_percentage": 100
    },
    "results": {
        "confidence": 0.92,
        "output": "Executive summary of AI trends in healthcare...",
        "metadata": {
            "execution_time": 245,
            "tokens_used": 15420,
            "cost": 0.23
        }
    },
    "stages": [
        {
            "stage": 1,
            "agent": "research-agent",
            "status": "completed",
            "confidence": 0.89,
            "execution_time": 120
        }
    ]
}
```

### **List Workflows**
```bash
GET /api/v1/workflows?status=running&pattern=pipeline&limit=10
```

### **Cancel Workflow**
```bash
DELETE /api/v1/workflows/{workflow_id}
```

## 📈 Analytics & Monitoring

### **Get Performance Metrics**
```bash
GET /api/v1/analytics/performance?timeframe=24h
```
**Response**:
```json
{
    "timeframe": "24h",
    "metrics": {
        "total_workflows": 145,
        "successful_workflows": 142,
        "average_execution_time": 187.5,
        "average_confidence": 0.87,
        "total_cost": 45.67,
        "top_patterns": [
            {"pattern": "pipeline", "count": 67},
            {"pattern": "supervisor", "count": 34}
        ]
    }
}
```

### **Get Agent Performance**
```bash
GET /api/v1/analytics/agents/{agent_id}/performance
```

### **Get Cost Analytics**
```bash
GET /api/v1/analytics/costs?timeframe=7d&breakdown=agent
```

## 🛠️ Tools Management

### **List Available Tools**
```bash
GET /api/v1/tools
```
**Response**:
```json
{
    "tools": [
        {
            "id": "web-search",
            "name": "Web Search Tool",
            "description": "Search the internet for information",
            "parameters": {
                "query": "string",
                "max_results": "integer"
            }
        }
    ]
}
```

### **Execute Tool**
```bash
POST /api/v1/tools/{tool_id}/execute
Content-Type: application/json

{
    "parameters": {
        "query": "latest AI research papers",
        "max_results": 10
    }
}
```

## 📝 Data Models

### **Agent Model**
```json
{
    "id": "string",
    "name": "string", 
    "type": "string",
    "status": "active|inactive|error",
    "capabilities": ["string"],
    "config": {
        "model": "string",
        "temperature": "number",
        "tools": ["string"]
    },
    "created_at": "datetime",
    "updated_at": "datetime"
}
```

### **Workflow Model**
```json
{
    "id": "string",
    "name": "string",
    "pattern": "pipeline|supervisor|parallel|reflective",
    "status": "pending|running|completed|failed|cancelled",
    "progress": {
        "current_stage": "integer",
        "total_stages": "integer", 
        "completion_percentage": "number"
    },
    "results": {
        "confidence": "number",
        "output": "string",
        "metadata": "object"
    },
    "created_at": "datetime",
    "completed_at": "datetime"
}
```

### **Task Model**
```json
{
    "description": "string",
    "requirements": ["string"],
    "context": "object",
    "priority": "high|medium|low",
    "timeout": "integer"
}
```

## ❌ Error Handling

### **Error Response Format**
```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Invalid agent configuration",
        "details": {
            "field": "temperature",
            "reason": "must be between 0 and 1"
        },
        "request_id": "req-123456"
    }
}
```

### **Common Error Codes**
- `VALIDATION_ERROR` - Invalid request parameters
- `AGENT_NOT_FOUND` - Agent ID does not exist  
- `WORKFLOW_NOT_FOUND` - Workflow ID does not exist
- `INSUFFICIENT_CREDITS` - API usage limits exceeded
- `INTERNAL_ERROR` - Server-side error
- `RATE_LIMIT_EXCEEDED` - Too many requests

## 🔧 Configuration

### **Environment Variables**
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/agents
REDIS_URL=redis://localhost:6379

# LangSmith Integration
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=multi-agent-orchestration

# AWS Bedrock (optional)
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
```

### **Rate Limits**
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour  
- **Enterprise**: Custom limits

## 🔗 SDKs & Client Libraries

### **Python SDK**
```python
from multi_agent_client import MultiAgentClient

client = MultiAgentClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Execute pipeline workflow
workflow = client.workflows.create_pipeline(
    name="research-pipeline",
    agents=["research-agent", "analysis-agent"],
    task="Analyze market trends"
)

# Get results
result = client.workflows.get_result(workflow.id)
```

### **JavaScript SDK**
```javascript
import { MultiAgentClient } from 'multi-agent-js-sdk';

const client = new MultiAgentClient({
    baseUrl: 'http://localhost:8000',
    apiKey: 'your-api-key'
});

// Execute workflow
const workflow = await client.workflows.createSupervisor({
    name: 'market-analysis',
    supervisor: 'supervisor-agent',
    specialists: ['market-researcher', 'competitor-analyst']
});
```

## 🎯 Best Practices

### **API Usage**
- **Use appropriate timeouts** for long-running workflows
- **Implement retry logic** with exponential backoff
- **Monitor rate limits** and implement queuing if necessary
- **Use webhook notifications** for workflow completion

### **Performance Optimization**
- **Batch requests** when possible to reduce API calls
- **Cache agent configurations** to avoid repeated lookups
- **Use parallel workflows** for independent tasks
- **Set appropriate confidence thresholds** to balance quality and speed

### **Error Handling**
- **Always check response status codes**
- **Implement graceful degradation** for failed agents
- **Log request IDs** for debugging and support
- **Use health check endpoints** to monitor service availability

---

For additional support and advanced integration patterns, refer to the [deployment guide](DEPLOYMENT_GUIDE.md) and [troubleshooting documentation](TROUBLESHOOTING.md).