#!/usr/bin/env python3
"""
AWS Bedrock Migration Script

Helps migrate from simulated LLM responses to AWS Bedrock foundation models.
"""

import json
import argparse
import boto3
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Import the platform modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.integrations.aws_bedrock import BedrockIntegration, BedrockConfig, BedrockAgent, BedrockModel
from src.integrations.llm_providers import LLMProviderManager
from src.multi_agent_platform import MultiAgentPlatform


class AWSMigrationHelper:
    """Helper class for migrating to AWS Bedrock."""
    
    def __init__(self):
        self.bedrock_config = None
        self.platform = None
        self.llm_manager = LLMProviderManager()
    
    def check_aws_credentials(self) -> Dict[str, Any]:
        """Check AWS credentials and permissions."""
        print("🔐 Checking AWS Credentials...")
        
        try:
            # Try to create a session
            session = boto3.Session()
            credentials = session.get_credentials()
            
            if not credentials:
                return {
                    "status": "error",
                    "message": "No AWS credentials found. Please configure AWS credentials.",
                    "suggestions": [
                        "Run 'aws configure' to set up credentials",
                        "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables",
                        "Use IAM roles if running on AWS infrastructure"
                    ]
                }
            
            # Test Bedrock access
            try:
                bedrock_client = session.client('bedrock', region_name='us-east-1')
                models = bedrock_client.list_foundation_models()
                
                return {
                    "status": "success",
                    "message": "AWS credentials are valid and Bedrock is accessible",
                    "region": session.region_name or "us-east-1",
                    "available_models": len(models.get('modelSummaries', []))
                }
                
            except Exception as bedrock_error:
                return {
                    "status": "warning",
                    "message": "AWS credentials found but Bedrock access failed",
                    "error": str(bedrock_error),
                    "suggestions": [
                        "Ensure your AWS account has Bedrock access enabled",
                        "Check IAM permissions for Bedrock service",
                        "Verify the region supports Bedrock"
                    ]
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error checking AWS credentials: {str(e)}",
                "suggestions": [
                    "Install AWS CLI: pip install awscli",
                    "Configure credentials: aws configure"
                ]
            }
    
    def list_available_models(self, region: str = "us-east-1") -> List[Dict[str, Any]]:
        """List available Bedrock models."""
        print(f"📋 Listing available models in {region}...")
        
        try:
            bedrock_config = BedrockConfig(region_name=region)
            bedrock = BedrockIntegration(bedrock_config)
            models = bedrock.list_available_models()
            
            print(f"Found {len(models)} available models:")
            for model in models:
                print(f"  - {model.get('modelId', 'Unknown')}")
                print(f"    Provider: {model.get('providerName', 'Unknown')}")
                print(f"    Input: {', '.join(model.get('inputModalities', []))}")
                print(f"    Output: {', '.join(model.get('outputModalities', []))}")
                print()
            
            return models
            
        except Exception as e:
            print(f"❌ Error listing models: {str(e)}")
            # Return predefined model list as fallback
            return [
                {
                    "modelId": model.value,
                    "providerName": model.value.split('.')[0],
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"]
                }
                for model in BedrockModel
            ]
    
    async def test_bedrock_connection(self, model_id: str = None, region: str = "us-east-1") -> Dict[str, Any]:
        """Test connection to Bedrock with a simple query."""
        print("🧪 Testing Bedrock connection...")
        
        target_model = model_id or BedrockModel.CLAUDE_3_HAIKU.value
        
        try:
            bedrock_config = BedrockConfig(
                region_name=region,
                model_id=target_model
            )
            bedrock = BedrockIntegration(bedrock_config)
            
            test_prompt = "Hello! Please respond with a simple greeting to test the connection."
            start_time = asyncio.get_event_loop().time()
            
            response = await bedrock.generate_text(test_prompt)
            
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time
            
            return {
                "status": "success",
                "model": target_model,
                "region": region,
                "response_time": round(response_time, 2),
                "response_preview": response[:100] + "..." if len(response) > 100 else response
            }
            
        except Exception as e:
            return {
                "status": "error",
                "model": target_model,
                "region": region,
                "error": str(e)
            }
    
    def create_migration_config(self, model_id: str, region: str = "us-east-1", 
                              output_file: str = "bedrock_config.json") -> bool:
        """Create configuration file for Bedrock migration."""
        print(f"📝 Creating migration configuration...")
        
        config = {
            "bedrock": {
                "region_name": region,
                "model_id": model_id,
                "max_tokens": 4096,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "migration": {
                "backup_existing_agents": True,
                "gradual_rollout": True,
                "fallback_to_simulation": True
            },
            "agents": {
                "research": {
                    "model_id": model_id,
                    "temperature": 0.3,  # Lower for research accuracy
                    "max_tokens": 6000
                },
                "analysis": {
                    "model_id": model_id,
                    "temperature": 0.4,
                    "max_tokens": 8000
                },
                "summary": {
                    "model_id": model_id,
                    "temperature": 0.5,
                    "max_tokens": 2000
                }
            }
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"✅ Configuration saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving configuration: {str(e)}")
            return False
    
    async def migrate_agents_to_bedrock(self, config_file: str = "bedrock_config.json") -> bool:
        """Migrate existing agents to use Bedrock."""
        print("🚀 Migrating agents to Bedrock...")
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            bedrock_config = BedrockConfig(**config["bedrock"])
            
            # Create Bedrock-powered agents
            agents = {}
            for agent_id, agent_config in config["agents"].items():
                print(f"  Creating Bedrock agent: {agent_id}")
                
                agent_bedrock_config = BedrockConfig(
                    region_name=bedrock_config.region_name,
                    model_id=agent_config.get("model_id", bedrock_config.model_id),
                    max_tokens=agent_config.get("max_tokens", bedrock_config.max_tokens),
                    temperature=agent_config.get("temperature", bedrock_config.temperature)
                )
                
                agent = BedrockAgent(
                    agent_id=agent_id,
                    name=f"{agent_id.title()} Agent (Bedrock)",
                    description=f"Bedrock-powered {agent_id} agent using {agent_config.get('model_id', 'default model')}",
                    bedrock_config=agent_bedrock_config
                )
                
                agents[agent_id] = agent
            
            # Test agents
            for agent_id, agent in agents.items():
                print(f"  Testing {agent_id} agent...")
                test_task = {
                    "type": agent_id,
                    "description": f"Test task for {agent_id} agent migration",
                    "id": f"migration_test_{agent_id}"
                }
                
                result = await agent.process_task(test_task)
                if result.success:
                    print(f"    ✅ {agent_id} agent working correctly")
                else:
                    print(f"    ❌ {agent_id} agent test failed: {result.error_message}")
                    return False
            
            print("✅ All agents migrated successfully!")
            
            # Setup LLM provider manager
            self.llm_manager.setup_bedrock(
                region=bedrock_config.region_name,
                model=bedrock_config.model_id,
                set_as_default=True
            )
            
            print("✅ LLM Provider Manager configured for Bedrock")
            return True
            
        except Exception as e:
            print(f"❌ Migration failed: {str(e)}")
            return False
    
    async def benchmark_performance(self, model_ids: List[str], region: str = "us-east-1") -> Dict[str, Any]:
        """Benchmark performance of different Bedrock models."""
        print("📊 Benchmarking model performance...")
        
        test_prompts = [
            "Analyze the benefits and challenges of renewable energy adoption.",
            "Summarize the key principles of machine learning in 3 paragraphs.",
            "Research the latest developments in quantum computing and their implications."
        ]
        
        results = {}
        
        for model_id in model_ids:
            print(f"  Testing {model_id}...")
            model_results = []
            
            bedrock_config = BedrockConfig(region_name=region, model_id=model_id)
            bedrock = BedrockIntegration(bedrock_config)
            
            for prompt in test_prompts:
                try:
                    start_time = asyncio.get_event_loop().time()
                    response = await bedrock.generate_text(prompt)
                    end_time = asyncio.get_event_loop().time()
                    
                    model_results.append({
                        "prompt": prompt[:50] + "...",
                        "response_time": round(end_time - start_time, 2),
                        "response_length": len(response),
                        "success": True
                    })
                    
                except Exception as e:
                    model_results.append({
                        "prompt": prompt[:50] + "...",
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate averages
            successful_results = [r for r in model_results if r["success"]]
            if successful_results:
                avg_response_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
                avg_response_length = sum(r["response_length"] for r in successful_results) / len(successful_results)
                success_rate = len(successful_results) / len(model_results)
            else:
                avg_response_time = 0
                avg_response_length = 0
                success_rate = 0
            
            results[model_id] = {
                "avg_response_time": round(avg_response_time, 2),
                "avg_response_length": round(avg_response_length),
                "success_rate": round(success_rate, 2),
                "detailed_results": model_results
            }
        
        return results
    
    def generate_migration_report(self, benchmark_results: Dict[str, Any], 
                                output_file: str = "migration_report.md") -> bool:
        """Generate migration report with recommendations."""
        print("📄 Generating migration report...")
        
        report = f"""# AWS Bedrock Migration Report
Generated on: {asyncio.get_event_loop().time()}

## Executive Summary
This report provides analysis and recommendations for migrating the Multi-Agent Orchestration Platform to AWS Bedrock foundation models.

## Model Performance Benchmark

| Model | Avg Response Time (s) | Avg Response Length | Success Rate |
|-------|----------------------|---------------------|--------------|
"""
        
        for model_id, results in benchmark_results.items():
            report += f"| {model_id} | {results['avg_response_time']} | {results['avg_response_length']} | {results['success_rate']*100:.1f}% |\\n"
        
        report += f"""
## Recommendations

### Recommended Primary Model
Based on the benchmark results, we recommend using the model with the best balance of performance and cost.

### Migration Strategy
1. **Phase 1**: Migrate non-critical agents first
2. **Phase 2**: Gradually migrate production agents
3. **Phase 3**: Monitor performance and optimize

### Cost Considerations
- Consider token usage and pricing for each model
- Monitor usage patterns and optimize accordingly
- Implement caching for frequently asked questions

### Monitoring and Alerting
- Set up CloudWatch monitoring for Bedrock usage
- Implement error tracking and fallback mechanisms
- Monitor response times and quality metrics

## Technical Implementation
1. Update agent configurations to use Bedrock
2. Implement proper error handling and retries
3. Set up monitoring and logging
4. Test thoroughly before production deployment

## Next Steps
1. Review and approve this migration plan
2. Set up AWS infrastructure and permissions
3. Execute migration in phases
4. Monitor and optimize performance
"""
        
        try:
            with open(output_file, 'w') as f:
                f.write(report)
            
            print(f"✅ Migration report saved to {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")
            return False


async def main():
    """Main migration script."""
    parser = argparse.ArgumentParser(description="AWS Bedrock Migration Helper")
    parser.add_argument("--check-credentials", action="store_true", 
                       help="Check AWS credentials and permissions")
    parser.add_argument("--list-models", action="store_true",
                       help="List available Bedrock models")
    parser.add_argument("--test-connection", action="store_true",
                       help="Test connection to Bedrock")
    parser.add_argument("--model", default=BedrockModel.CLAUDE_3_HAIKU.value,
                       help="Model ID to use for testing")
    parser.add_argument("--region", default="us-east-1",
                       help="AWS region to use")
    parser.add_argument("--create-config", action="store_true",
                       help="Create migration configuration file")
    parser.add_argument("--migrate", action="store_true",
                       help="Perform migration to Bedrock")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark model performance")
    parser.add_argument("--config-file", default="bedrock_config.json",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    migration_helper = AWSMigrationHelper()
    
    print("🚀 AWS Bedrock Migration Helper")
    print("=" * 50)
    
    if args.check_credentials:
        result = migration_helper.check_aws_credentials()
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")
        if "suggestions" in result:
            print("Suggestions:")
            for suggestion in result["suggestions"]:
                print(f"  - {suggestion}")
        print()
    
    if args.list_models:
        models = migration_helper.list_available_models(args.region)
        print(f"Listed {len(models)} models\\n")
    
    if args.test_connection:
        result = await migration_helper.test_bedrock_connection(args.model, args.region)
        print(f"Connection test: {result['status']}")
        if result['status'] == 'success':
            print(f"Model: {result['model']}")
            print(f"Region: {result['region']}")
            print(f"Response time: {result['response_time']}s")
            print(f"Response preview: {result['response_preview']}")
        else:
            print(f"Error: {result['error']}")
        print()
    
    if args.create_config:
        success = migration_helper.create_migration_config(args.model, args.region, args.config_file)
        if success:
            print(f"✅ Configuration created: {args.config_file}\\n")
    
    if args.migrate:
        success = await migration_helper.migrate_agents_to_bedrock(args.config_file)
        if success:
            print("✅ Migration completed successfully!")
        else:
            print("❌ Migration failed!")
        print()
    
    if args.benchmark:
        models_to_test = [
            BedrockModel.CLAUDE_3_HAIKU.value,
            BedrockModel.CLAUDE_3_SONNET.value,
            BedrockModel.TITAN_TEXT_G1_LARGE.value
        ]
        
        results = await migration_helper.benchmark_performance(models_to_test, args.region)
        
        print("Benchmark Results:")
        for model, metrics in results.items():
            print(f"  {model}:")
            print(f"    Avg Response Time: {metrics['avg_response_time']}s")
            print(f"    Avg Response Length: {metrics['avg_response_length']} chars")
            print(f"    Success Rate: {metrics['success_rate']*100:.1f}%")
        
        # Generate report
        migration_helper.generate_migration_report(results)
        print()
    
    print("🎉 Migration helper completed!")


if __name__ == "__main__":
    asyncio.run(main())