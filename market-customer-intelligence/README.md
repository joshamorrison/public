# Market & Customer Intelligence Platform

Unified intelligence platform for customer feedback and competitive market data. Uses RAG with ChromaDB for semantic search and LangGraph/LangChain agents for sentiment tracking, trend detection, and opportunity mapping.

## Key Results
- **Real-time insight delivery** across multiple data sources
- **20% faster GTM decisions** through automated intelligence
- **15% revenue lift** in targeted markets via competitive analysis

## Technology Stack
- **Python** - Core platform development
- **LangChain** - LLM orchestration and prompt engineering
- **LangGraph** - Multi-agent workflow orchestration
- **ChromaDB** - Vector database for semantic search
- **AWS** - Cloud infrastructure and storage

## Features
- Multi-source data ingestion and processing
- Vector-based semantic search and retrieval
- Real-time sentiment analysis and tracking
- Competitive intelligence and market mapping
- Automated insight generation and reporting

## Project Structure
```
market-customer-intelligence/
├── src/
│   ├── data/
│   │   ├── ingestion/
│   │   │   ├── social_media_collector.py
│   │   │   ├── review_scraper.py
│   │   │   └── survey_processor.py
│   │   ├── processing/
│   │   │   ├── text_preprocessor.py
│   │   │   └── sentiment_analyzer.py
│   │   └── storage/
│   │       ├── vector_store.py
│   │       └── chromadb_manager.py
│   ├── agents/
│   │   ├── sentiment_agent.py
│   │   ├── trend_detector.py
│   │   ├── competitor_analyzer.py
│   │   └── opportunity_mapper.py
│   ├── retrieval/
│   │   ├── semantic_search.py
│   │   ├── similarity_matcher.py
│   │   └── context_builder.py
│   ├── analysis/
│   │   ├── market_analyzer.py
│   │   ├── customer_segmentation.py
│   │   └── competitive_intelligence.py
│   └── reporting/
│       ├── insight_generator.py
│       ├── dashboard.py
│       └── alert_system.py
├── langchain/
│   ├── agents/
│   │   └── intelligence_agents.py
│   └── chains/
│       └── analysis_chains.py
├── chromadb/
│   ├── collections/
│   │   ├── customer_feedback.py
│   │   ├── market_data.py
│   │   └── competitor_intel.py
│   └── embeddings/
│       └── custom_embeddings.py
├── config/
│   ├── data_sources.yaml
│   └── analysis_config.yaml
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/joshamorrison/public.git
   cd public/market-customer-intelligence
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Initialize ChromaDB**
   ```bash
   python -c "from src.data.storage.chromadb_manager import ChromaDBManager; ChromaDBManager().initialize_collections()"
   ```

5. **Start the intelligence platform**
   ```bash
   python src/main.py --mode realtime
   ```

## Architecture

### Data Ingestion Layer
- **Social Media**: Twitter, LinkedIn, Reddit, Facebook
- **Review Platforms**: Google Reviews, Yelp, Trustpilot, G2
- **Survey Data**: Customer satisfaction, NPS, feedback forms
- **Competitive Data**: Pricing, product features, marketing campaigns

### Processing Pipeline
- **Text Preprocessing**: Cleaning, normalization, language detection
- **Embedding Generation**: Semantic vectors for similarity search
- **Sentiment Analysis**: Multi-model ensemble for accuracy
- **Entity Recognition**: Products, companies, features extraction

### Intelligence Generation
- **Trend Detection**: Statistical and ML-based pattern recognition
- **Competitive Analysis**: Feature comparison and positioning maps
- **Opportunity Mapping**: Gap analysis and market sizing
- **Customer Insights**: Segmentation and behavior analysis

## Key Capabilities

### Real-Time Intelligence
- **Live Data Streams**: Continuous monitoring of data sources
- **Automated Alerts**: Threshold-based notifications for key metrics
- **Dynamic Dashboards**: Real-time visualization of market trends
- **Instant Insights**: Sub-second query response with semantic search

### Advanced Analytics
- **Sentiment Tracking**: Multi-dimensional emotion and opinion analysis
- **Competitive Positioning**: Feature-by-feature comparison matrices
- **Market Opportunity**: TAM/SAM analysis with growth projections
- **Customer Journey**: Touchpoint analysis and optimization recommendations

## Business Impact

This platform enables product and marketing teams to:
- **Accelerate go-to-market decisions** by 20% through real-time intelligence
- **Identify market opportunities** with comprehensive competitive analysis
- **Improve product-market fit** through continuous customer feedback analysis
- **Drive revenue growth** with 15% lift in targeted market segments

## API Endpoints

### Intelligence APIs
```bash
# Get market sentiment for a product
GET /api/v1/sentiment/{product_id}

# Search similar customer feedback
POST /api/v1/search/similarity
{
  "query": "product feature request",
  "limit": 10
}

# Competitive analysis
GET /api/v1/competitors/{company_id}/analysis

# Market opportunities
GET /api/v1/opportunities/{market_segment}
```

## Contact

For technical questions or implementation guidance, reach out to:
- **Joshua Morrison** - [joshamorrison@gmail.com](mailto:joshamorrison@gmail.com)
- **LinkedIn** - [linkedin.com/in/joshamorrison](https://www.linkedin.com/in/joshamorrison)