# Deployment Strategies

## Overview

The Vision Object Classifier supports two main deployment strategies, each optimized for different use cases and requirements.

## Deployment Options Comparison

| Feature | AWS Deployment | Streamlit Deployment |
|---------|---------------|---------------------|
| **Target Audience** | Production/Enterprise | Demos/Prototypes/Internal Tools |
| **Scalability** | Auto-scaling (1-10+ instances) | Single instance |
| **Cost** | Variable (pay-per-use) | Free tier available |
| **Setup Complexity** | High (Infrastructure as Code) | Low (Simple web app) |
| **API Access** | Full REST API | Web interface + optional API |
| **Customization** | Full control | Limited to Streamlit features |
| **Monitoring** | CloudWatch, ALB metrics | Basic Streamlit metrics |
| **Security** | VPC, IAM, SSL/TLS | Basic authentication options |

## 🚀 AWS Production Deployment

### Use Cases
- **Production API services** with high availability
- **Enterprise integration** requiring scalability
- **High-traffic applications** (1000+ requests/day)
- **Multi-environment deployment** (dev/staging/prod)
- **Advanced monitoring and logging** requirements

### Architecture
```
Internet → ALB → ECS Fargate Tasks → S3 (Models/Outputs)
                        ↓
                  CloudWatch Logs
```

### Quick Start
```bash
cd infrastructure/aws/

# Set environment variables
export AWS_REGION=us-west-2
export ENVIRONMENT=dev

# Deploy infrastructure
./deploy.sh

# API will be available at the ALB DNS name
```

### Features
- **Auto-scaling**: 1-10 instances based on CPU usage
- **Load balancing**: Application Load Balancer with health checks  
- **Container orchestration**: ECS Fargate for serverless containers
- **Storage**: S3 for models and outputs
- **Monitoring**: CloudWatch logs and metrics
- **Security**: VPC, security groups, IAM roles

### Cost Estimate
- **Development**: $20-50/month
- **Production**: $100-300/month (depending on traffic)

## 🎨 Streamlit Interactive Deployment

### Use Cases
- **Interactive demos** and presentations
- **Internal tools** for data science teams
- **Proof of concepts** and prototypes
- **Educational purposes** and workshops
- **Quick testing** and validation

### Architecture
```
Browser → Streamlit App → Local/Remote API → Models
```

### Quick Start
```bash
cd infrastructure/streamlit/

# Local deployment with API
./deploy.sh local

# Streamlit app: http://localhost:8501
# API backend: http://localhost:8000
```

### Features
- **Interactive web interface** with drag-and-drop uploads
- **Real-time classification** with confidence scores
- **Batch processing** capabilities
- **Demo image gallery** for quick testing
- **Results visualization** with probability breakdowns
- **CSV export** for batch results

### Deployment Options

#### 1. Local Development
```bash
./deploy.sh local
```
- Starts both API and Streamlit locally
- Best for development and testing

#### 2. Streamlit Cloud (Free)
```bash
./deploy.sh cloud
```
- Deploys to Streamlit Cloud platform
- Free hosting with GitHub integration
- Public URL with sharing capabilities

#### 3. Docker Deployment
```bash
./deploy.sh docker
docker-compose -f infrastructure/streamlit/docker-compose.yml up
```
- Containerized deployment
- Includes both API and Streamlit services
- Easy to deploy on any Docker host

## 📁 Infrastructure Organization

Following our gold standard, both deployment strategies are properly organized:

```
infrastructure/
├── aws/                    # AWS production deployment
│   ├── terraform/         # Infrastructure as Code
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── iam.tf
│   │   ├── s3.tf
│   │   └── outputs.tf
│   └── deploy.sh          # Deployment script
└── streamlit/             # Streamlit demo deployment
    ├── app.py            # Main Streamlit application
    ├── config.toml       # Streamlit configuration
    ├── requirements.txt  # Streamlit-specific deps
    ├── Dockerfile        # Container setup
    ├── docker-compose.yml # Multi-service setup
    └── deploy.sh         # Deployment script
```

## 🔧 Configuration Management

### Environment Variables

**AWS Deployment:**
```bash
export AWS_REGION=us-west-2
export ENVIRONMENT=prod
export PROJECT_NAME=vision-classifier
```

**Streamlit Deployment:**
```bash
export STREAMLIT_PORT=8501
export API_PORT=8000
export API_URL=http://localhost:8000
```

### Secrets Management

**AWS:** Uses AWS Systems Manager Parameter Store and IAM roles

**Streamlit:** Uses secrets.toml file:
```toml
API_URL = "http://localhost:8000"
DEFAULT_MODEL_TYPE = "balanced"
ENABLE_BATCH_PROCESSING = true
```

## 🚀 Deployment Workflows

### AWS Production Pipeline
```bash
# 1. Infrastructure setup
cd infrastructure/aws/
terraform init
terraform plan
terraform apply

# 2. Build and deploy
./deploy.sh

# 3. Verify deployment  
curl http://<alb-dns-name>/health/status
```

### Streamlit Development Pipeline  
```bash
# 1. Local development
cd infrastructure/streamlit/
./deploy.sh local

# 2. Test and iterate
# Make changes to app.py
# Streamlit auto-reloads

# 3. Deploy to cloud
git commit -am "Update Streamlit app"
git push
./deploy.sh cloud
```

## 🛠️ Maintenance and Updates

### AWS Updates
```bash
# Update application
docker build -f docker/Dockerfile -t vision-classifier:latest .
./deploy.sh  # Automatically updates ECS service

# Update infrastructure
terraform plan
terraform apply
```

### Streamlit Updates
```bash
# Local updates - automatic reload
# Cloud updates - git push triggers rebuild
git push origin main
```

## 📊 Monitoring and Troubleshooting

### AWS Monitoring
- **CloudWatch Logs**: `/ecs/vision-classifier-{env}`
- **Metrics**: ECS service metrics, ALB metrics
- **Health Checks**: `/health/status` endpoint
- **Alarms**: CPU/Memory utilization

### Streamlit Monitoring  
- **Streamlit Metrics**: Built-in usage statistics
- **API Health**: Manual health check endpoints
- **Logs**: Local log files or container logs

## 🔐 Security Considerations

### AWS Security
- VPC with private subnets
- Security groups restricting access
- IAM roles with least privilege
- S3 bucket encryption
- ALB with potential SSL/TLS

### Streamlit Security
- Basic authentication options
- HTTPS in production
- Input validation
- File upload restrictions

## 📈 Scaling Strategies

### AWS Scaling
- **Horizontal**: Auto-scaling groups (1-10 instances)  
- **Vertical**: Increase Fargate CPU/memory
- **Geographic**: Multi-region deployment
- **Performance**: ElastiCache for model caching

### Streamlit Scaling
- **Limited scaling**: Single instance design
- **Load balancing**: Multiple Streamlit instances behind proxy
- **Alternative**: Migrate to AWS for higher scale

## 🎯 Decision Matrix

**Choose AWS when:**
- ✅ Production deployment required
- ✅ High availability needed (>99.9% uptime)
- ✅ Auto-scaling required
- ✅ Enterprise integration
- ✅ Advanced monitoring needed
- ✅ Team has DevOps expertise

**Choose Streamlit when:**
- ✅ Demo or prototype needed
- ✅ Interactive UI preferred
- ✅ Quick deployment required
- ✅ Free hosting acceptable
- ✅ Limited technical resources
- ✅ Educational or research use

## 🚀 Getting Started Recommendations

### For Production Use
1. Start with Streamlit for prototyping
2. Validate business requirements
3. Deploy to AWS when ready to scale
4. Use both: Streamlit for demos, AWS for production API

### For Development Teams
1. Use Streamlit for internal tools and demos
2. Develop and test with local Streamlit deployment
3. Deploy AWS infrastructure for production integrations
4. Maintain both deployment paths for different audiences