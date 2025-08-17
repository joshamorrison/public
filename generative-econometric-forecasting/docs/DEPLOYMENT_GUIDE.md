# 🚀 Portfolio Technology Stack Deployment Guide

**Complete deployment guide for R • Python • AWS • LangChain • LangSmith • Apache Airflow**

## 📋 Prerequisites

### **Required Software**
- **Python 3.8+** - Primary development environment
- **R 4.0+** - Statistical modeling and econometrics
- **AWS CLI** - Cloud deployment and management
- **PostgreSQL 12+** - Airflow metadata database
- **Redis** - Caching and message broker (optional)
- **Git** - Version control

### **Required Accounts & API Keys**
- **AWS Account** - Cloud infrastructure deployment
- **OpenAI API Key** - AI-powered analysis
- **LangChain/LangSmith Account** - AI monitoring
- **FRED API Key** - Economic data access

---

## 🐍 Python Environment Setup

### **1. Clone Repository**
```bash
git clone https://github.com/joshamorrison/public.git
cd public/generative-econometric-forecasting
```

### **2. Create Virtual Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### **3. Install Python Dependencies**
```bash
# Install all portfolio dependencies
pip install -r requirements.txt

# Verify installation
python -c "import pandas, numpy, sklearn, langchain; print('✅ Python stack ready')"
```

---

## 📊 R Integration Setup

### **1. Install R**
```bash
# Windows: Download from https://cran.r-project.org/bin/windows/base/
# macOS:
brew install r

# Linux (Ubuntu/Debian):
sudo apt-get update
sudo apt-get install r-base-dev
```

### **2. Install Required R Packages**
```r
# Open R console and run:
install.packages(c(
    "vars",         # Vector Autoregression
    "forecast",     # Time series forecasting
    "urca",         # Unit root and cointegration tests
    "tseries",      # Time series analysis
    "VARselect",    # VAR model selection
    "lmtest",       # Linear model testing
    "sandwich"      # Robust standard errors
))
```

### **3. Configure rpy2 Integration**
```bash
# Set R environment variables
export R_HOME=/usr/lib/R  # Linux
export R_HOME=/Library/Frameworks/R.framework/Resources  # macOS
set R_HOME=C:\Program Files\R\R-4.3.0  # Windows

# Test R integration
python -c "import rpy2.robjects as ro; print('✅ R integration ready')"
```

---

## ☁️ AWS Cloud Deployment

### **1. Configure AWS CLI**
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: us-east-1
# Default output format: json
```

### **2. Deploy Infrastructure**
```bash
# Navigate to AWS infrastructure
cd infrastructure/aws

# Deploy CloudFormation stack
aws cloudformation create-stack \
    --stack-name econometric-forecasting \
    --template-body file://cloudformation/main.yaml \
    --capabilities CAPABILITY_IAM

# Monitor deployment
aws cloudformation describe-stacks --stack-name econometric-forecasting
```

### **3. Create S3 Buckets**
```bash
# Create data lake buckets
aws s3 mb s3://econometric-forecasting-data
aws s3 mb s3://econometric-forecasting-models
aws s3 mb s3://econometric-forecasting-outputs

# Set bucket policies (optional)
aws s3api put-bucket-lifecycle-configuration \
    --bucket econometric-forecasting-data \
    --lifecycle-configuration file://s3-lifecycle.json
```

### **4. Deploy Lambda Functions**
```bash
# Package and deploy forecasting functions
cd lambda/
zip -r forecasting-function.zip .
aws lambda create-function \
    --function-name econometric-forecasting \
    --runtime python3.9 \
    --role arn:aws:iam::ACCOUNT:role/lambda-execution-role \
    --handler forecasting_handler.lambda_handler \
    --zip-file fileb://forecasting-function.zip
```

---

## 🌪️ Apache Airflow Setup

### **1. Install PostgreSQL**
```bash
# Windows: Download from https://www.postgresql.org/download/windows/
# macOS:
brew install postgresql
brew services start postgresql

# Linux (Ubuntu/Debian):
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
```

### **2. Create Airflow Database**
```sql
-- Connect to PostgreSQL as postgres user
sudo -u postgres psql

-- Create database and user
CREATE DATABASE airflow;
CREATE USER airflow WITH PASSWORD 'airflow';
GRANT ALL PRIVILEGES ON DATABASE airflow TO airflow;
\q
```

### **3. Install and Configure Airflow**
```bash
# Set Airflow home directory
export AIRFLOW_HOME=~/airflow

# Install Airflow with required providers
pip install apache-airflow[postgres,aws,celery]==2.8.0

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### **4. Configure Airflow Settings**
```bash
# Edit airflow.cfg
nano $AIRFLOW_HOME/airflow.cfg

# Key configurations:
# executor = LocalExecutor
# sql_alchemy_conn = postgresql+psycopg2://airflow:airflow@localhost/airflow
# load_examples = False
# dags_folder = /path/to/your/project/infrastructure/airflow/dags
```

### **5. Start Airflow Services**
```bash
# Start web server (in one terminal)
airflow webserver --port 8080

# Start scheduler (in another terminal)
airflow scheduler

# Access UI at http://localhost:8080
# Login: admin / admin
```

---

## 🔗 LangChain & LangSmith Setup

### **1. Configure LangChain**
```bash
# Install LangChain components
pip install langchain langchain-openai langchain-community

# Set OpenAI API key
export OPENAI_API_KEY="your_openai_api_key_here"
```

### **2. Setup LangSmith Monitoring**
```bash
# Create LangSmith account at https://smith.langchain.com/

# Configure LangSmith
export LANGCHAIN_API_KEY="your_langsmith_api_key_here"
export LANGCHAIN_PROJECT="econometric-forecasting"
export LANGCHAIN_TRACING_V2=true

# Test LangSmith integration
python scripts/langsmith_enhanced_monitoring.py
```

### **3. Configure Custom Monitoring**
```python
# Test enhanced monitoring setup
from scripts.langsmith_enhanced_monitoring import EconometricForecastingTracer

tracer = EconometricForecastingTracer()
print("✅ LangSmith monitoring ready")
```

---

## 🔬 Advanced Analytics Setup

### **1. Install Causal Inference Libraries**
```bash
# Install causal inference dependencies
pip install causalml dowhy econml

# Verify installation
python -c "from src.causal_inference.causal_models import CausalInferenceEngine; print('✅ Causal inference ready')"
```

### **2. Test Scenario Analysis**
```bash
# Verify scenario analysis engine
python -c "from src.scenario_analysis.scenario_engine import HighPerformanceScenarioEngine; print('✅ Scenario analysis ready')"
```

### **3. Test Sensitivity Analysis**
```bash
# Verify LLM-based sensitivity testing
python -c "from src.sensitivity_testing.automated_sensitivity import AutomatedSensitivityTester; print('✅ Sensitivity testing ready')"
```

---

## 🔧 Environment Configuration

### **1. Copy Environment Template**
```bash
# Copy and configure environment variables
cp .env.example .env
nano .env
```

### **2. Configure All API Keys**
```bash
# Core API keys
FRED_API_KEY=your_fred_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here

# AWS configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_DEFAULT_REGION=us-east-1

# R configuration
R_HOME=C:\Program Files\R\R-4.3.0
R_PACKAGES_REQUIRED=vars,forecast,urca,VARselect,tseries

# Airflow configuration
AIRFLOW_DATABASE_URL=postgresql://airflow:airflow@localhost:5432/airflow
AIRFLOW_ADMIN_USER=admin
AIRFLOW_ADMIN_PASSWORD=your_secure_password

# LangSmith monitoring
LANGSMITH_PROJECT=econometric-forecasting
LANGSMITH_TRACE_FORECASTING=true
```

---

## 🧪 Deployment Verification

### **1. Run Complete System Test**
```bash
# Test entire portfolio stack
python quick_start.py --test-all-components

# Expected output should show:
# ✅ Python environment ready
# ✅ R integration working
# ✅ AWS connectivity established
# ✅ LangChain AI framework operational
# ✅ LangSmith monitoring active
# ✅ Airflow scheduler running
# ✅ Causal inference models loaded
# ✅ Scenario analysis engine ready
# ✅ Sensitivity testing operational
```

### **2. Test Individual Components**
```bash
# Test R integration
python -c "from models.r_statistical_models import RStatisticalModels; r = RStatisticalModels(); print('R models:', r.list_available_models())"

# Test AWS connectivity
aws s3 ls s3://econometric-forecasting-data

# Test Airflow DAGs
airflow dags list | grep econometric

# Test LangSmith tracing
python -c "import langsmith; print('LangSmith client ready')"
```

### **3. Run Sample Workflow**
```bash
# Execute end-to-end portfolio demonstration
python scripts/portfolio_demo.py

# This will:
# 1. Fetch economic data
# 2. Run R statistical models
# 3. Execute causal inference analysis
# 4. Generate scenario forecasts
# 5. Perform sensitivity testing
# 6. Create executive summary with LangChain
# 7. Monitor with LangSmith
# 8. Store results in AWS S3
```

---

## 📊 Monitoring & Maintenance

### **1. Daily Operations**
```bash
# Check Airflow DAG status
airflow dags state econometric_forecasting_pipeline $(date +%Y-%m-%d)

# Monitor AWS costs
aws ce get-cost-and-usage --time-period Start=$(date -d '7 days ago' '+%Y-%m-%d'),End=$(date '+%Y-%m-%d') --granularity DAILY --metrics BlendedCost

# Check LangSmith traces
langsmith trace list --project econometric-forecasting --limit 10
```

### **2. Performance Monitoring**
```bash
# R model performance
Rscript -e "source('models/performance_check.R')"

# Python model metrics
python scripts/model_performance_check.py

# AWS Lambda metrics
aws logs describe-log-groups --log-group-name-prefix /aws/lambda/econometric
```

### **3. Backup and Recovery**
```bash
# Backup Airflow metadata
pg_dump airflow > airflow_backup_$(date +%Y%m%d).sql

# Backup model artifacts to S3
aws s3 sync models/ s3://econometric-forecasting-models/backup/

# Export environment configuration
cp .env .env.backup.$(date +%Y%m%d)
```

---

## 🚨 Troubleshooting

### **Common Issues and Solutions**

#### **R Integration Issues**
```bash
# Problem: rpy2 installation fails
# Solution: Install R development headers
sudo apt-get install r-base-dev  # Linux
# Windows: Use conda install rpy2

# Problem: R packages not found
# Solution: Reinstall R packages
Rscript -e "install.packages(c('vars', 'forecast', 'urca'), repos='https://cran.r-project.org')"
```

#### **AWS Deployment Issues**
```bash
# Problem: CloudFormation stack creation fails
# Solution: Check IAM permissions
aws iam get-user
aws iam list-attached-user-policies --user-name YOUR_USERNAME

# Problem: Lambda function timeout
# Solution: Increase timeout in CloudFormation template
# Timeout: 300  # 5 minutes
```

#### **Airflow Issues**
```bash
# Problem: Airflow webserver won't start
# Solution: Reset Airflow database
airflow db reset

# Problem: DAGs not appearing
# Solution: Check dags_folder path in airflow.cfg
grep dags_folder $AIRFLOW_HOME/airflow.cfg
```

#### **LangSmith Monitoring Issues**
```bash
# Problem: Traces not appearing in LangSmith
# Solution: Verify environment variables
echo $LANGCHAIN_API_KEY
echo $LANGCHAIN_TRACING_V2

# Problem: API rate limits
# Solution: Implement exponential backoff
# See scripts/langsmith_enhanced_monitoring.py
```

---

## 🎯 Production Deployment Checklist

- [ ] **Python Environment**: Virtual environment created and dependencies installed
- [ ] **R Integration**: R installed, packages configured, rpy2 working
- [ ] **AWS Infrastructure**: CloudFormation stack deployed, S3 buckets created
- [ ] **Apache Airflow**: PostgreSQL setup, Airflow running, DAGs visible
- [ ] **LangChain Framework**: API keys configured, models responding
- [ ] **LangSmith Monitoring**: Tracing active, custom metrics flowing
- [ ] **Environment Variables**: All API keys and configurations set
- [ ] **Security**: IAM roles configured, secrets properly managed
- [ ] **Monitoring**: CloudWatch alarms set, performance dashboards created
- [ ] **Backup Strategy**: Automated backups for database and models
- [ ] **Documentation**: Team trained on all portfolio technologies

---

## 🎓 Technology Learning Resources

### **Python Data Science**
- [Official Python Tutorial](https://docs.python.org/3/tutorial/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### **R Statistical Computing**
- [R for Data Science](https://r4ds.had.co.nz/)
- [Time Series Analysis with R](https://otexts.com/fpp3/)
- [Advanced R Programming](https://adv-r.hadley.nz/)

### **AWS Cloud Services**
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS Lambda Developer Guide](https://docs.aws.amazon.com/lambda/)
- [CloudFormation User Guide](https://docs.aws.amazon.com/cloudformation/)

### **Apache Airflow**
- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)

### **LangChain & AI**
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangSmith User Guide](https://docs.smith.langchain.com/)

---

**🎉 Congratulations! You now have a complete portfolio technology stack showcasing R • Python • AWS • LangChain • LangSmith • Apache Airflow in production.**