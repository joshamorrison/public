"""
Researcher Agent

Specialist agent focused on information gathering, research, 
and knowledge acquisition from various sources.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

from .base_agent import BaseAgent, AgentResult


class ResearcherAgent(BaseAgent):
    """
    Researcher agent specialized in information gathering and analysis.
    
    The researcher agent:
    - Gathers information from multiple sources
    - Conducts literature reviews and fact-checking
    - Synthesizes research findings
    - Provides comprehensive information summaries
    """

    def __init__(self, agent_id: str = "researcher-001"):
        super().__init__(
            agent_id=agent_id,
            name="Researcher Agent",
            description="Information gathering specialist for comprehensive research tasks"
        )
        self.research_sources = [
            "internal_knowledge",
            "web_search", 
            "document_analysis",
            "data_sources"
        ]

    async def process_task(self, task: Dict[str, Any]) -> AgentResult:
        """
        Process a research task by gathering and analyzing information.
        
        Args:
            task: Research task specification
            
        Returns:
            AgentResult: Research findings and analysis
        """
        start_time = datetime.now()
        
        try:
            task_type = task.get("type", "general")
            task_description = task.get("description", "")
            
            # Simulate research process
            research_findings = await self._conduct_research(task_description, task_type)
            
            # Analyze and synthesize findings
            analysis = await self._analyze_findings(research_findings, task_description)
            
            # Create comprehensive research report
            report = await self._generate_research_report(research_findings, analysis, task)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate confidence based on number of sources and consistency
            confidence = self._calculate_research_confidence(research_findings)
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=report,
                confidence=confidence,
                metadata={
                    "sources_consulted": len(research_findings),
                    "research_method": "comprehensive",
                    "processing_time": processing_time,
                    "task_type": task_type
                },
                timestamp=datetime.now()
            )
            
            self.update_performance_metrics(result, processing_time)
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = AgentResult(
                agent_id=self.agent_id,
                task_id=task.get("task_id", "unknown"),
                content=f"Research failed: {str(e)}",
                confidence=0.0,
                metadata={"error": str(e), "processing_time": processing_time},
                timestamp=datetime.now(),
                success=False,
                error_message=str(e)
            )
            
            self.update_performance_metrics(result, processing_time)
            return result

    async def _conduct_research(self, description: str, task_type: str) -> List[Dict[str, Any]]:
        """
        Conduct research by gathering information from multiple sources.
        
        Args:
            description: Research topic description
            task_type: Type of research task
            
        Returns:
            List of research findings from different sources
        """
        findings = []
        
        # Simulate research from different sources
        for source in self.research_sources:
            finding = await self._research_from_source(source, description, task_type)
            if finding:
                findings.append(finding)
        
        return findings

    async def _research_from_source(self, source: str, description: str, 
                                  task_type: str) -> Dict[str, Any]:
        """
        Simulate research from a specific source.
        
        Args:
            source: Research source identifier
            description: Research topic
            task_type: Type of research
            
        Returns:
            Research finding from the source
        """
        # Simulate different research approaches based on source
        if source == "internal_knowledge":
            return {
                "source": source,
                "content": f"Internal knowledge base indicates that {description} involves multiple factors including historical context, current trends, and future implications.",
                "reliability": 0.8,
                "timestamp": datetime.now()
            }
        
        elif source == "web_search":
            return {
                "source": source,
                "content": f"Web search results for '{description}' show significant interest in related topics, with key themes around methodology, implementation, and best practices.",
                "reliability": 0.7,
                "timestamp": datetime.now()
            }
        
        elif source == "document_analysis":
            return {
                "source": source,
                "content": f"Document analysis reveals that {description} has been extensively studied, with consensus on core principles and ongoing debate about specific applications.",
                "reliability": 0.9,
                "timestamp": datetime.now()
            }
        
        elif source == "data_sources":
            return {
                "source": source,
                "content": f"Data analysis shows quantitative evidence supporting key aspects of {description}, with statistical significance in primary metrics.",
                "reliability": 0.85,
                "timestamp": datetime.now()
            }
        
        return None

    async def _analyze_findings(self, findings: List[Dict[str, Any]], 
                              description: str) -> Dict[str, Any]:
        """
        Analyze research findings to extract key insights.
        
        Args:
            findings: List of research findings
            description: Original research topic
            
        Returns:
            Analysis of the research findings
        """
        if not findings:
            return {"summary": "No findings to analyze", "confidence": 0.0}
        
        # Calculate overall reliability
        total_reliability = sum(f.get("reliability", 0) for f in findings)
        avg_reliability = total_reliability / len(findings)
        
        # Extract key themes
        themes = ["methodology", "implementation", "best practices", "historical context"]
        
        # Generate analysis summary
        analysis = {
            "summary": f"Research on '{description}' reveals consistent themes across {len(findings)} sources.",
            "key_themes": themes,
            "source_diversity": len(set(f["source"] for f in findings)),
            "overall_reliability": avg_reliability,
            "consistency_score": 0.8,  # Simulated consistency
            "gaps_identified": ["Need for more recent data", "Limited cross-domain analysis"]
        }
        
        return analysis

    async def _generate_research_report(self, findings: List[Dict[str, Any]], 
                                       analysis: Dict[str, Any], 
                                       task: Dict[str, Any]) -> str:
        """
        Generate a comprehensive research report.
        
        Args:
            findings: Research findings from sources
            analysis: Analysis of the findings
            task: Original task specification
            
        Returns:
            Formatted research report
        """
        report_lines = []
        
        # Header
        report_lines.append("RESEARCH REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Topic: {task.get('description', 'Unknown')}")
        report_lines.append(f"Research completed: {datetime.now()}")
        report_lines.append(f"Sources consulted: {len(findings)}")
        report_lines.append("")
        
        # Executive Summary
        report_lines.append("EXECUTIVE SUMMARY")
        report_lines.append("-" * 20)
        report_lines.append(analysis.get("summary", "No summary available"))
        report_lines.append(f"Overall reliability: {analysis.get('overall_reliability', 0):.1%}")
        report_lines.append(f"Source diversity: {analysis.get('source_diversity', 0)} different sources")
        report_lines.append("")
        
        # Key Findings
        report_lines.append("KEY FINDINGS")
        report_lines.append("-" * 20)
        for i, theme in enumerate(analysis.get("key_themes", []), 1):
            report_lines.append(f"{i}. {theme.title()}: Extensively covered across sources")
        report_lines.append("")
        
        # Source Analysis
        report_lines.append("SOURCE ANALYSIS")
        report_lines.append("-" * 20)
        for finding in findings:
            report_lines.append(f"Source: {finding['source']}")
            report_lines.append(f"Reliability: {finding.get('reliability', 0):.1%}")
            report_lines.append(f"Content: {finding['content'][:150]}...")
            report_lines.append("")
        
        # Research Gaps
        report_lines.append("IDENTIFIED GAPS & RECOMMENDATIONS")
        report_lines.append("-" * 35)
        for gap in analysis.get("gaps_identified", []):
            report_lines.append(f"• {gap}")
        report_lines.append("")
        
        # Conclusions
        report_lines.append("CONCLUSIONS")
        report_lines.append("-" * 20)
        report_lines.append("Research provides comprehensive coverage of the topic with good source diversity.")
        report_lines.append("Findings show strong consistency across different research methods.")
        report_lines.append("Additional investigation recommended for identified gaps.")
        
        return "\n".join(report_lines)

    def _calculate_research_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on research quality.
        
        Args:
            findings: List of research findings
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not findings:
            return 0.0
        
        # Base confidence on source diversity and reliability
        source_diversity = len(set(f["source"] for f in findings)) / len(self.research_sources)
        avg_reliability = sum(f.get("reliability", 0) for f in findings) / len(findings)
        
        # Combine factors
        confidence = (source_diversity * 0.4) + (avg_reliability * 0.6)
        
        return min(confidence, 1.0)

    def get_capabilities(self) -> List[str]:
        """Return researcher agent capabilities."""
        return [
            "information_gathering",
            "literature_review",
            "fact_checking",
            "research_synthesis",
            "source_analysis",
            "knowledge_extraction"
        ]

    def get_research_sources(self) -> List[str]:
        """Return available research sources."""
        return self.research_sources.copy()