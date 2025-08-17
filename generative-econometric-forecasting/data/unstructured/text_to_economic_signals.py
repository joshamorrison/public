"""
Text to Economic Signals Converter
Converts unstructured text data into quantitative economic signals and indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

logger = logging.getLogger(__name__)


class EconomicSignal(BaseModel):
    """Structure for extracted economic signals."""
    indicator_name: str = Field(description="Name of the economic indicator")
    signal_type: str = Field(description="Type of signal (directional, magnitude, timing)")
    direction: str = Field(description="Direction of change (up, down, stable)")
    magnitude: str = Field(description="Magnitude of change (small, moderate, large)")
    timeframe: str = Field(description="Expected timeframe (short, medium, long)")
    confidence: float = Field(description="Confidence in signal (0-1)")
    supporting_text: str = Field(description="Text that supports this signal")


class TextToEconomicSignals:
    """Converts text data into quantitative economic signals."""
    
    def __init__(self, openai_model: str = "gpt-3.5-turbo"):
        """
        Initialize text to signals converter.
        
        Args:
            openai_model: OpenAI model for analysis
        """
        self.llm = ChatOpenAI(model=openai_model, temperature=0.1)
        self.signal_parser = JsonOutputParser(pydantic_object=EconomicSignal)
        
        # Economic indicator patterns
        self.indicator_patterns = {
            'gdp': {
                'patterns': [
                    r'gdp.*?(grow|decline|increase|decrease|expand|contract|rise|fall)',
                    r'economic.*?(growth|contraction|expansion|recession)',
                    r'gross domestic product.*?(up|down|rising|falling)'
                ],
                'keywords': ['gdp', 'gross domestic product', 'economic growth', 'economic output']
            },
            'inflation': {
                'patterns': [
                    r'inflation.*?(rise|fall|increase|decrease|surge|drop)',
                    r'price.*?(pressure|increase|growth|spike|decline)',
                    r'consumer price.*?(index|cpi).*?(up|down|rising|falling)'
                ],
                'keywords': ['inflation', 'prices', 'cpi', 'consumer price index', 'price pressure']
            },
            'employment': {
                'patterns': [
                    r'unemployment.*?(rate|fall|rise|increase|decrease)',
                    r'job.*?(creation|loss|growth|market|openings)',
                    r'employment.*?(growth|decline|rate|level)'
                ],
                'keywords': ['unemployment', 'employment', 'jobs', 'labor market', 'workforce']
            },
            'interest_rates': {
                'patterns': [
                    r'interest.*?rate.*?(rise|fall|increase|decrease|hike|cut)',
                    r'federal.*?reserve.*?(raise|lower|cut|hike)',
                    r'monetary.*?policy.*?(tighten|ease|accommodate)'
                ],
                'keywords': ['interest rates', 'federal reserve', 'monetary policy', 'fed funds']
            },
            'consumer_confidence': {
                'patterns': [
                    r'consumer.*?confidence.*?(rise|fall|increase|decrease)',
                    r'consumer.*?sentiment.*?(improve|worsen|positive|negative)',
                    r'spending.*?(increase|decrease|rise|fall)'
                ],
                'keywords': ['consumer confidence', 'consumer sentiment', 'consumer spending']
            },
            'housing': {
                'patterns': [
                    r'housing.*?(market|prices|sales).*?(rise|fall|increase|decrease)',
                    r'home.*?(sales|prices).*?(up|down|rising|falling)',
                    r'real.*?estate.*?(boom|bust|growth|decline)'
                ],
                'keywords': ['housing market', 'home sales', 'real estate', 'housing prices']
            }
        }
        
        # Signal extraction prompt
        self.signal_prompt = PromptTemplate(
            input_variables=["text", "indicator"],
            template="""
            Extract economic signals from the following text related to {indicator}.
            
            Text: {text}
            
            Identify signals about {indicator} and provide:
            - indicator_name: Name of the economic indicator
            - signal_type: "directional" (up/down), "magnitude" (how much), or "timing" (when)
            - direction: "up", "down", or "stable"
            - magnitude: "small", "moderate", or "large"
            - timeframe: "short" (0-3 months), "medium" (3-12 months), or "long" (12+ months)
            - confidence: Confidence level from 0.0 to 1.0
            - supporting_text: Specific text that supports this signal
            
            Only extract signals that are clearly supported by the text.
            
            {format_instructions}
            """
        )
        
        # Quantification patterns
        self.quantification_patterns = {
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'basis_points': r'(\d+)\s*basis\s*points?',
            'currency': r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            'numbers': r'(\d+(?:,\d{3})*(?:\.\d+)?)',
            'multipliers': {
                'million': 1_000_000,
                'billion': 1_000_000_000,
                'trillion': 1_000_000_000_000,
                'thousand': 1_000
            }
        }
        
        logger.info("Text to Economic Signals converter initialized")
    
    def extract_signals_from_text(self, 
                                 text: str,
                                 indicators: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract economic signals from text for specified indicators.
        
        Args:
            text: Text to analyze
            indicators: List of indicators to extract (all if None)
        
        Returns:
            Dictionary of signals by indicator
        """
        if indicators is None:
            indicators = list(self.indicator_patterns.keys())
        
        all_signals = {}
        
        for indicator in indicators:
            try:
                # Check if indicator is mentioned in text
                if self._is_indicator_mentioned(text, indicator):
                    signals = self._extract_indicator_signals(text, indicator)
                    if signals:
                        all_signals[indicator] = signals
                        logger.info(f"Extracted {len(signals)} signals for {indicator}")
                
            except Exception as e:
                logger.error(f"Error extracting signals for {indicator}: {e}")
                continue
        
        return all_signals
    
    def _is_indicator_mentioned(self, text: str, indicator: str) -> bool:
        """Check if indicator is mentioned in text."""
        text_lower = text.lower()
        
        # Check keywords
        keywords = self.indicator_patterns[indicator]['keywords']
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return True
        
        # Check patterns
        patterns = self.indicator_patterns[indicator]['patterns']
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _extract_indicator_signals(self, text: str, indicator: str) -> List[Dict[str, Any]]:
        """Extract signals for a specific indicator."""
        try:
            # Format prompt
            formatted_prompt = self.signal_prompt.partial(
                format_instructions=self.signal_parser.get_format_instructions()
            )
            
            prompt = formatted_prompt.format(
                text=text[:2000],  # Limit text length
                indicator=indicator
            )
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Try to parse as JSON
            try:
                signal = self.signal_parser.parse(response.content)
                return [signal.dict()]
            except:
                # Fallback to pattern-based extraction
                return self._extract_signals_with_patterns(text, indicator)
                
        except Exception as e:
            logger.error(f"LLM signal extraction failed for {indicator}: {e}")
            return self._extract_signals_with_patterns(text, indicator)
    
    def _extract_signals_with_patterns(self, text: str, indicator: str) -> List[Dict[str, Any]]:
        """Extract signals using regex patterns as fallback."""
        signals = []
        text_lower = text.lower()
        
        patterns = self.indicator_patterns[indicator]['patterns']
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            
            for match in matches:
                # Extract direction from match
                match_text = match.group(0)
                
                if any(word in match_text for word in ['grow', 'increase', 'rise', 'expand', 'up', 'surge']):
                    direction = 'up'
                elif any(word in match_text for word in ['decline', 'decrease', 'fall', 'contract', 'down', 'drop']):
                    direction = 'down'
                else:
                    direction = 'stable'
                
                # Estimate magnitude
                magnitude = 'moderate'  # Default
                if any(word in match_text for word in ['surge', 'spike', 'soar', 'plunge', 'crash']):
                    magnitude = 'large'
                elif any(word in match_text for word in ['slight', 'small', 'minor']):
                    magnitude = 'small'
                
                signal = {
                    'indicator_name': indicator,
                    'signal_type': 'directional',
                    'direction': direction,
                    'magnitude': magnitude,
                    'timeframe': 'medium',  # Default
                    'confidence': 0.6,  # Pattern-based confidence
                    'supporting_text': match.group(0)
                }
                
                signals.append(signal)
        
        return signals
    
    def quantify_economic_mentions(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract quantitative economic data from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of quantified economic mentions
        """
        quantified_data = {
            'percentages': [],
            'currency_amounts': [],
            'basis_points': [],
            'numerical_values': []
        }
        
        # Extract percentages
        percentage_matches = re.finditer(self.quantification_patterns['percentage'], text)
        for match in percentage_matches:
            context = self._get_context_around_match(text, match)
            quantified_data['percentages'].append({
                'value': float(match.group(1)),
                'context': context,
                'position': match.span()
            })
        
        # Extract currency amounts
        currency_matches = re.finditer(self.quantification_patterns['currency'], text)
        for match in currency_matches:
            context = self._get_context_around_match(text, match)
            amount = float(match.group(1).replace(',', ''))
            
            # Check for multipliers
            multiplier = self._find_multiplier_in_context(context)
            final_amount = amount * multiplier
            
            quantified_data['currency_amounts'].append({
                'value': final_amount,
                'original_text': match.group(0),
                'context': context,
                'position': match.span()
            })
        
        # Extract basis points
        bp_matches = re.finditer(self.quantification_patterns['basis_points'], text)
        for match in bp_matches:
            context = self._get_context_around_match(text, match)
            quantified_data['basis_points'].append({
                'value': int(match.group(1)),
                'percentage_equivalent': int(match.group(1)) / 100,
                'context': context,
                'position': match.span()
            })
        
        return quantified_data
    
    def _get_context_around_match(self, text: str, match: re.Match, window: int = 50) -> str:
        """Get context around a regex match."""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end]
    
    def _find_multiplier_in_context(self, context: str) -> float:
        """Find numerical multipliers in context."""
        context_lower = context.lower()
        
        for word, multiplier in self.quantification_patterns['multipliers'].items():
            if word in context_lower:
                return multiplier
        
        return 1.0  # No multiplier found
    
    def convert_signals_to_timeseries(self, 
                                    signals_by_time: Dict[datetime, Dict[str, List[Dict[str, Any]]]],
                                    indicator: str) -> pd.Series:
        """
        Convert extracted signals to time series format.
        
        Args:
            signals_by_time: Signals organized by timestamp
            indicator: Indicator to convert
        
        Returns:
            Time series of signal strengths
        """
        timestamps = []
        signal_strengths = []
        
        for timestamp, signals_dict in signals_by_time.items():
            indicator_signals = signals_dict.get(indicator, [])
            
            if not indicator_signals:
                signal_strength = 0.0
            else:
                # Calculate aggregate signal strength
                total_strength = 0.0
                total_weight = 0.0
                
                for signal in indicator_signals:
                    direction = signal.get('direction', 'stable')
                    magnitude = signal.get('magnitude', 'moderate')
                    confidence = signal.get('confidence', 0.5)
                    
                    # Direction multiplier
                    direction_mult = {'up': 1, 'down': -1, 'stable': 0}[direction]
                    
                    # Magnitude multiplier
                    magnitude_mult = {'small': 0.3, 'moderate': 0.6, 'large': 1.0}[magnitude]
                    
                    # Calculate signal strength
                    strength = direction_mult * magnitude_mult * confidence
                    weight = confidence
                    
                    total_strength += strength * weight
                    total_weight += weight
                
                signal_strength = total_strength / total_weight if total_weight > 0 else 0.0
            
            timestamps.append(timestamp)
            signal_strengths.append(signal_strength)
        
        return pd.Series(signal_strengths, index=timestamps, name=f'{indicator}_signal')
    
    def analyze_text_batch(self, 
                          texts: List[str],
                          timestamps: Optional[List[datetime]] = None,
                          indicators: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Analyze a batch of texts for economic signals.
        
        Args:
            texts: List of texts to analyze
            timestamps: Corresponding timestamps
            indicators: Indicators to extract
        
        Returns:
            Batch analysis results
        """
        if timestamps is None:
            timestamps = [datetime.now() - timedelta(hours=i) for i in range(len(texts))]
        
        if len(timestamps) != len(texts):
            raise ValueError("Number of timestamps must match number of texts")
        
        batch_results = {
            'signals_by_time': {},
            'aggregated_signals': {},
            'signal_timeseries': {},
            'summary_stats': {}
        }
        
        # Extract signals for each text
        for text, timestamp in zip(texts, timestamps):
            signals = self.extract_signals_from_text(text, indicators)
            batch_results['signals_by_time'][timestamp] = signals
        
        # Aggregate signals by indicator
        all_indicators = set()
        for signals_dict in batch_results['signals_by_time'].values():
            all_indicators.update(signals_dict.keys())
        
        for indicator in all_indicators:
            # Convert to time series
            ts = self.convert_signals_to_timeseries(
                batch_results['signals_by_time'], indicator
            )
            batch_results['signal_timeseries'][indicator] = ts
            
            # Calculate summary statistics
            batch_results['summary_stats'][indicator] = {
                'mean_signal': ts.mean(),
                'signal_volatility': ts.std(),
                'positive_signals': (ts > 0).sum(),
                'negative_signals': (ts < 0).sum(),
                'neutral_signals': (ts == 0).sum(),
                'max_signal': ts.max(),
                'min_signal': ts.min()
            }
        
        logger.info(f"Analyzed {len(texts)} texts, found signals for {len(all_indicators)} indicators")
        return batch_results
    
    def export_signals(self, signals: Dict[str, Any], filename: str) -> str:
        """Export signals to JSON file."""
        import json
        
        # Convert pandas objects to serializable format
        export_data = {}
        for key, value in signals.items():
            if isinstance(value, pd.Series):
                export_data[key] = {
                    'values': value.tolist(),
                    'index': value.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                }
            elif isinstance(value, dict):
                export_data[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, pd.Series):
                        export_data[key][sub_key] = {
                            'values': sub_value.tolist(),
                            'index': sub_value.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                        }
                    else:
                        export_data[key][sub_key] = sub_value
            else:
                export_data[key] = value
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported signals to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting signals: {e}")
            return ""


if __name__ == "__main__":
    # Example usage
    converter = TextToEconomicSignals()
    
    # Sample economic texts
    sample_texts = [
        "GDP growth accelerated to 3.2% in the third quarter, exceeding economists' forecasts of 2.8%. The robust economic expansion was driven by strong consumer spending and business investment.",
        "Unemployment fell to 3.8% last month, marking the lowest level in two decades. Job creation remained strong with 250,000 new positions added.",
        "The Federal Reserve raised interest rates by 25 basis points to combat rising inflation, which reached 4.1% annually. Officials signaled additional rate hikes may be necessary.",
        "Consumer confidence declined sharply as concerns about economic stability grew. Retail sales dropped 2.1% month-over-month, reflecting cautious spending behavior.",
        "Housing market showed signs of cooling with home sales falling 5.3% and average prices declining in major metropolitan areas."
    ]
    
    # Extract signals from individual texts
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text[:100]}...")
        signals = converter.extract_signals_from_text(text)
        
        for indicator, signal_list in signals.items():
            print(f"  {indicator}: {len(signal_list)} signals")
            for signal in signal_list:
                print(f"    - {signal['direction']} ({signal['magnitude']}, conf: {signal['confidence']:.2f})")
    
    # Batch analysis
    timestamps = [datetime.now() - timedelta(hours=i*6) for i in range(len(sample_texts))]
    batch_results = converter.analyze_text_batch(sample_texts, timestamps)
    
    print(f"\nBatch Analysis Results:")
    print(f"Found signals for {len(batch_results['signal_timeseries'])} indicators")
    
    for indicator, stats in batch_results['summary_stats'].items():
        print(f"\n{indicator.upper()}:")
        print(f"  Mean signal: {stats['mean_signal']:.3f}")
        print(f"  Volatility: {stats['signal_volatility']:.3f}")
        print(f"  Positive/Negative/Neutral: {stats['positive_signals']}/{stats['negative_signals']}/{stats['neutral_signals']}")
    
    # Quantify economic mentions
    quantified = converter.quantify_economic_mentions(sample_texts[0])
    print(f"\nQuantified data from first text:")
    for data_type, values in quantified.items():
        if values:
            print(f"  {data_type}: {len(values)} mentions")
            for value in values[:2]:  # Show first 2
                print(f"    - {value}")
    
    # Export results
    export_file = converter.export_signals(batch_results, 'economic_signals_export.json')
    if export_file:
        print(f"\nResults exported to: {export_file}")