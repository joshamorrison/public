# Multi-Agent Orchestration Models

This directory contains trained models, configurations, and model artifacts for the multi-agent orchestration platform.

## Directory Structure

```
models/
├── __init__.py                 # Main model loading utilities
├── README.md                   # This file
├── llm_configs/               # LLM provider configurations
│   ├── __init__.py
│   ├── config_loader.py       # Configuration loading utilities
│   ├── provider_configs.py    # Programmatic config presets
│   ├── openai_configs.json    # OpenAI model configurations
│   ├── anthropic_configs.json # Anthropic model configurations
│   └── bedrock_configs.json   # AWS Bedrock configurations
├── routing_models/            # Task routing and agent assignment
│   ├── __init__.py
│   ├── model_loader.py        # Model persistence utilities
│   ├── task_router.py         # Main routing logic
│   └── capability_matcher.py  # Capability matching algorithms
└── agent_models/              # Agent performance models
    ├── __init__.py
    ├── performance_models.py   # Performance prediction models
    └── behavior_models.py      # Behavior modeling
```

## LLM Configurations

Pre-configured LLM settings optimized for different agent types and use cases:

### Providers Supported
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Anthropic**: Claude-3-haiku, Claude-3-sonnet, Claude-3-opus  
- **AWS Bedrock**: Claude models via Bedrock, Amazon Titan

### Configuration Types
- **Agent-specific**: Optimized for researcher, analyst, synthesizer, critic, supervisor
- **Use-case specific**: Fast completion, high quality, general purpose
- **Cost-optimized**: Balanced performance vs. cost configurations

### Usage Example
```python
from models import load_llm_config

# Load configuration for researcher agent using Anthropic
config = load_llm_config("anthropic", "researcher")

# Load fast completion config for quick responses  
fast_config = load_llm_config("openai", "fast_completion")
```

## Routing Models

Intelligent task routing and agent assignment models:

### Components
- **TaskRouter**: Main routing engine that assigns tasks to optimal agents
- **AgentPerformancePredictor**: Predicts agent performance for task types
- **CapabilityMatcher**: Matches task requirements to agent capabilities
- **TaskAgentScorer**: Comprehensive scoring for task-agent pairings

### Features
- Performance-based routing using historical data
- Capability matching with semantic similarity
- Workload balancing and availability tracking
- Multi-factor scoring (capability, experience, availability, performance)

### Usage Example
```python
from models.routing_models import TaskRouter

router = TaskRouter()

# Register agents with capabilities
router.register_agent("researcher_001", "researcher", 
                     ["web_search", "data_analysis", "research"])

# Route a task
task = {"type": "research", "description": "Market analysis"}
selected_agent = router.route_task(task)
```

## Agent Models

Trained models for agent behavior and performance optimization:

### Model Types
- **Performance Models**: Predict execution time, success probability, resource usage
- **Behavior Models**: Model agent interaction patterns and collaboration effectiveness
- **Workload Predictors**: Optimize task distribution and load balancing

### Training Data
Models are trained on:
- Historical task execution data
- Agent performance metrics
- Interaction patterns and collaboration outcomes
- Resource utilization patterns

## Model Persistence

Models are saved using Python's pickle format and can be loaded/saved using the provided utilities:

```python
from models.routing_models import load_routing_model, save_routing_model

# Load a trained model
router = load_routing_model("production_router", "task_router")

# Save a model
save_routing_model(router, "new_router", "task_router", 
                   metadata={"version": "2.0", "training_date": "2024-01-01"})
```

## Configuration Management

### Environment Variables
- `MODELS_DIR`: Override default models directory
- `LLM_PROVIDER`: Default LLM provider (openai, anthropic, bedrock)
- `DEFAULT_MODEL_CONFIG`: Default configuration name

### Configuration Validation
All configurations are validated for:
- Required fields (provider, model_name, temperature, max_tokens)
- Value ranges (temperature 0-2, positive timeouts)
- Provider-specific requirements (e.g., region for Bedrock)

## Performance Optimization

### Model Selection
- **Fast**: Optimized for low latency (GPT-3.5-turbo, Claude-haiku)
- **Balanced**: Good performance/cost trade-off (GPT-4, Claude-sonnet)  
- **Quality**: Maximum quality (GPT-4-turbo, Claude-opus)

### Cost Management
- Built-in cost estimation for all configurations
- Token usage tracking and cost monitoring
- Budget-aware configuration selection

## Integration with Platform

Models integrate seamlessly with the multi-agent platform:

1. **Automatic Loading**: Platform loads default models on startup
2. **Hot Swapping**: Models can be updated without restarting
3. **Fallback Logic**: Graceful degradation if models unavailable
4. **Monitoring**: Built-in performance tracking and metrics

## Development and Training

### Adding New Configurations
1. Add configuration to appropriate JSON file
2. Update provider_configs.py if needed
3. Test with config_loader validation
4. Document in README

### Training New Models
1. Collect training data from platform execution
2. Train model using provided utilities
3. Validate model performance
4. Save with metadata and documentation

## Monitoring and Maintenance

### Health Checks
- Model loading validation
- Configuration integrity checks
- Performance regression detection

### Updates
- Regular model retraining with new data
- Configuration updates for new LLM versions
- Performance tuning based on usage patterns

## Best Practices

1. **Always validate configurations** before deployment
2. **Use appropriate model types** for your use case
3. **Monitor costs** when using premium models
4. **Keep training data current** for routing models
5. **Test model updates** in staging before production