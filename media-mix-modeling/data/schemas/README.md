# Data Schemas

This directory contains JSON schema files for all sample data used in the Media Mix Modeling platform.

## Schema Files

### 1. Marketing Campaign Data
- **File**: `marketing_campaign_schema.json`
- **Data Source**: HuggingFace `RafaM97/marketing_social_media` (REAL DATA)
- **Description**: Social media marketing campaigns with instructions, context, and strategies
- **Size**: 689 rows
- **Use Cases**: Campaign strategy analysis, social media planning, audience targeting

### 2. Channel Performance Data
- **File**: `channel_performance_schema.json`
- **Data Source**: Generated realistic data
- **Description**: Daily performance metrics across 6 marketing channels
- **Size**: 180 rows (30 days × 6 channels)
- **Use Cases**: Performance monitoring, channel comparison, ROI analysis

### 3. Customer Journey Data
- **File**: `customer_journey_schema.json`
- **Data Source**: Generated attribution data
- **Description**: Customer touchpoint journeys for attribution modeling
- **Size**: 3,130 rows (1,000 customers, average 3.1 touchpoints each)
- **Use Cases**: Multi-touch attribution, customer journey analysis, conversion path optimization

### 4. Campaign Budget Data
- **File**: `campaign_budget_schema.json`
- **Data Source**: Generated budget data
- **Description**: Campaign budget allocation and optimization parameters
- **Size**: 20 campaigns
- **Use Cases**: Budget optimization, campaign planning, spend allocation

### 5. MMM Time Series Data
- **File**: `mmm_time_series_schema.json`
- **Data Source**: Generated with seasonal effects
- **Description**: Weekly media spend and outcome data for MMM modeling
- **Size**: 52 weeks
- **Use Cases**: Media mix modeling, attribution modeling, saturation analysis

## Data Quality Levels

- **REAL**: Actual data from external sources (HuggingFace, Kaggle)
- **SYNTHETIC**: Generated realistic data following industry patterns
- **DEMO**: Synthetic data optimized for demonstration purposes

## Schema Validation

Each schema follows JSON Schema Draft 7 specification and includes:
- Property definitions with types and constraints
- Required fields
- Example values
- Source metadata
- Data quality indicators

## Usage in API

These schemas are used by the API for:
- Request/response validation
- Data type checking
- Documentation generation
- Client SDK generation

## Sample Data Location

All sample data files are located in the `data/samples/` directory:
```
data/samples/
├── marketing_campaign_data.csv      # Real HuggingFace data
├── channel_performance_data.csv     # Generated performance data
├── customer_journey_data.csv        # Generated attribution data
├── campaign_budget_data.csv         # Generated budget data
└── mmm_time_series_data.csv         # Generated MMM data
```

## Data Integration

The `MediaDataClient` class automatically:
1. Attempts to fetch real data from HuggingFace/Kaggle
2. Falls back to generated realistic data if needed
3. Maintains consistent schemas across all sources
4. Provides data source transparency and quality indicators