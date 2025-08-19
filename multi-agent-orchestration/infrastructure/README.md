# Infrastructure Directory

This directory contains deployment configurations and Infrastructure as Code (IaC) templates for the Multi-Agent Orchestration Platform.

## Directory Structure

- **aws/**: AWS deployment configurations
  - **cloudformation/**: CloudFormation templates
  - **terraform/**: Terraform configurations
  - **scripts/**: Deployment automation scripts

- **airflow/**: Apache Airflow workflow orchestration
  - **dags/**: Airflow DAG definitions
  - **plugins/**: Custom operators and hooks

- **monitoring/**: Observability and monitoring configurations
  - **prometheus/**: Metrics collection configuration
  - **grafana/**: Dashboard definitions
  - **alerts/**: Alert rules and configurations

## Documentation

For detailed infrastructure documentation, see:
- [docs/infrastructure.md](../docs/infrastructure.md) - Complete infrastructure overview
- [docs/deployment_guide.md](../docs/deployment_guide.md) - Deployment instructions
- [docs/docker.md](../docs/docker.md) - Container configurations