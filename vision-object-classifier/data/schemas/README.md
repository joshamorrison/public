# Data Schemas

This folder contains schema definitions and API specifications for the Vision Object Classifier.

## Files

- `image_classification_schema.json` - Complete API and data format schema
- `README.md` - This file

## Usage

The schema file defines:
- Input/output formats for API endpoints
- Supported image formats and constraints
- Data folder organization structure
- Model types and their characteristics

## API Endpoints Schema

```json
{
  "classification": {
    "single": "POST /api/v1/classify/single",
    "batch": "POST /api/v1/batch/classify"
  },
  "health": {
    "status": "GET /health/status",
    "models": "GET /health/models"
  }
}
```

## Data Structure

```
data/
├── raw/           # Original datasets
├── processed/     # Clean and dirty labeled images
├── samples/       # Demo images for quick testing
└── synthetic/     # Data generation tools
```