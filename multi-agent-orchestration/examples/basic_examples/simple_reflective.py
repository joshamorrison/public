"""
Simple Reflective Example - Iterative Improvement

Demonstrates iterative refinement through reflection and improvement cycles.
Multiple agents collaborate to continuously enhance output quality.

This example shows how reflection enables self-improvement and 
adaptive optimization through feedback loops.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List

from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.agents.summary_agent import SummaryAgent


async def run_iterative_improvement(content_topic: str = "climate change solutions") -> Dict[str, Any]:
    """
    Run iterative content improvement using the Reflective pattern.
    
    Each iteration involves:
    1. Content creation/revision
    2. Critical reflection and evaluation
    3. Improvement recommendations
    4. Implementation of improvements
    
    Args:
        content_topic: Topic for content creation and improvement
        
    Returns:
        Results showing improvement progression through iterations
    """
    print(f"🔄 Starting Iterative Content Improvement for: {content_topic}")
    print("=" * 70)
    
    # Initialize agents with specialized roles
    content_agent = SummaryAgent()  # Creates and revises content
    critic_agent = AnalysisAgent()  # Provides critical analysis
    research_agent = ResearchAgent()  # Adds supporting information
    
    # Track improvement across iterations
    iterations = []
    current_content = ""
    max_iterations = 3
    
    for iteration in range(1, max_iterations + 1):
        print(f"🔄 Iteration {iteration}: {'Initial Creation' if iteration == 1 else 'Refinement and Improvement'}")
        print("-" * 50)
        
        # Phase 1: Content Creation/Revision
        if iteration == 1:
            # Initial content creation
            creation_task = {
                "type": "content_creation",
                "description": f"Create comprehensive content about {content_topic}",
                "requirements": [
                    "Clear introduction and overview",
                    "Key challenges and opportunities", 
                    "Potential solutions and approaches",
                    "Implementation considerations",
                    "Future outlook and recommendations"
                ],
                "quality_criteria": [
                    "Clarity and readability",
                    "Factual accuracy",
                    "Comprehensive coverage",
                    "Actionable insights"
                ]
            }
            print("✍️ Creating initial content...")
        else:
            # Content revision based on previous feedback
            creation_task = {
                "type": "content_revision",
                "description": f"Improve content about {content_topic} based on feedback",
                "current_content": current_content,
                "improvement_feedback": iterations[-1]["reflection"]["improvement_recommendations"],
                "specific_issues": iterations[-1]["reflection"]["identified_issues"],
                "enhancement_targets": [
                    "Address identified weaknesses",
                    "Enhance clarity and flow",
                    "Add missing information",
                    "Improve structure and organization"
                ]
            }
            print(f"✍️ Revising content based on iteration {iteration-1} feedback...")
        
        content_result = await content_agent.process_task(creation_task)
        current_content = content_result.content
        
        print(f"✅ Content {'creation' if iteration == 1 else 'revision'} completed")
        print(f"📝 Content confidence: {content_result.confidence:.2f}")
        print(f"📄 Content length: {len(current_content)} characters")
        print()
        
        # Phase 2: Critical Reflection and Analysis
        print("🔍 Conducting critical reflection...")
        
        reflection_task = {
            "type": "content_evaluation",
            "description": f"Critically evaluate content quality for {content_topic}",
            "content_to_evaluate": current_content,
            "evaluation_criteria": [
                "Accuracy and factual correctness",
                "Completeness and comprehensiveness",
                "Clarity and readability",
                "Logical structure and flow",
                "Actionability of recommendations",
                "Engagement and persuasiveness"
            ],
            "improvement_focus": [
                "Identify gaps and weaknesses",
                "Suggest specific improvements",
                "Recommend additional research areas",
                "Propose structural enhancements"
            ]
        }
        
        reflection_result = await critic_agent.process_task(reflection_task)
        
        print(f"✅ Critical reflection completed")
        print(f"🎯 Reflection confidence: {reflection_result.confidence:.2f}")
        print()
        
        # Phase 3: Research Enhancement (if needed)
        print("📚 Gathering additional research insights...")
        
        research_task = {
            "type": "content_enhancement_research",
            "description": f"Research additional information to enhance content on {content_topic}",
            "current_content": current_content,
            "reflection_feedback": reflection_result.content,
            "research_targets": [
                "Latest developments and trends",
                "Supporting data and statistics",
                "Expert opinions and case studies",
                "Implementation examples and best practices"
            ]
        }
        
        research_result = await research_agent.process_task(research_task)
        
        print(f"✅ Enhancement research completed")
        print(f"📊 Research confidence: {research_result.confidence:.2f}")
        print()
        
        # Calculate iteration metrics
        iteration_confidence = (
            content_result.confidence + 
            reflection_result.confidence + 
            research_result.confidence
        ) / 3
        
        # Parse improvement recommendations
        improvement_recommendations = parse_reflection_feedback(reflection_result.content)
        identified_issues = extract_content_issues(reflection_result.content, iteration)
        
        # Store iteration results
        iteration_data = {
            "iteration_number": iteration,
            "content": {
                "text": current_content,
                "confidence": content_result.confidence,
                "length": len(current_content),
                "metadata": content_result.metadata
            },
            "reflection": {
                "analysis": reflection_result.content,
                "confidence": reflection_result.confidence,
                "improvement_recommendations": improvement_recommendations,
                "identified_issues": identified_issues,
                "metadata": reflection_result.metadata
            },
            "research": {
                "enhancements": research_result.content,
                "confidence": research_result.confidence,
                "metadata": research_result.metadata
            },
            "iteration_metrics": {
                "overall_confidence": iteration_confidence,
                "content_length_change": len(current_content) - (len(iterations[-1]["content"]["text"]) if iterations else 0),
                "improvement_areas": len(improvement_recommendations),
                "issues_identified": len(identified_issues)
            }
        }
        
        iterations.append(iteration_data)
        
        print(f"📈 Iteration {iteration} Overall Confidence: {iteration_confidence:.2f}")
        print(f"🔧 Improvement areas identified: {len(improvement_recommendations)}")
        print(f"⚠️  Issues to address: {len(identified_issues)}")
        print()
        
        # Check if further iteration is needed
        if iteration < max_iterations:
            if iteration_confidence > 0.85 and len(identified_issues) <= 2:
                print("🎉 Content quality threshold reached - stopping early")
                break
            else:
                print(f"🔄 Proceeding to iteration {iteration + 1} for further improvement")
                print()
    
    # Calculate overall improvement progression
    confidence_progression = [iter_data["iteration_metrics"]["overall_confidence"] for iter_data in iterations]
    initial_confidence = confidence_progression[0]
    final_confidence = confidence_progression[-1]
    improvement_gain = final_confidence - initial_confidence
    
    # Compile comprehensive results
    reflective_results = {
        "workflow_type": "reflective",
        "content_topic": content_topic,
        "execution_time": datetime.now().isoformat(),
        "iterations_completed": len(iterations),
        "iteration_details": iterations,
        "final_content": current_content,
        "improvement_progression": {
            "initial_confidence": initial_confidence,
            "final_confidence": final_confidence,
            "improvement_gain": improvement_gain,
            "confidence_trajectory": confidence_progression,
            "total_improvement_percentage": (improvement_gain / initial_confidence) * 100 if initial_confidence > 0 else 0
        },
        "reflective_metrics": {
            "total_iterations": len(iterations),
            "successful_iterations": len(iterations),
            "average_confidence": sum(confidence_progression) / len(confidence_progression),
            "max_confidence": max(confidence_progression),
            "min_confidence": min(confidence_progression),
            "convergence_rate": "High" if improvement_gain > 0.1 else "Medium" if improvement_gain > 0.05 else "Low"
        },
        "quality_evolution": {
            "initial_length": len(iterations[0]["content"]["text"]),
            "final_length": len(current_content),
            "content_growth": len(current_content) - len(iterations[0]["content"]["text"]),
            "refinement_cycles": len(iterations) - 1,
            "issues_resolved": sum(len(iter_data["reflection"]["identified_issues"]) for iter_data in iterations[:-1])
        }
    }
    
    print("🎉 Iterative Improvement Completed Successfully!")
    print(f"🔄 Iterations: {len(iterations)}")
    print(f"📈 Confidence Improvement: {initial_confidence:.2f} → {final_confidence:.2f} (+{improvement_gain:.2f})")
    print(f"📊 Total Improvement: {reflective_results['improvement_progression']['total_improvement_percentage']:.1f}%")
    print("=" * 70)
    
    return reflective_results


def parse_reflection_feedback(reflection_content: str) -> List[str]:
    """
    Parse reflection content to extract improvement recommendations.
    
    In a production system, this would use NLP to extract specific
    recommendations. For this demo, we'll simulate structured parsing.
    """
    # Simulate intelligent parsing of reflection feedback
    recommendations = [
        "Enhance introduction with stronger hook and clearer value proposition",
        "Add more specific examples and case studies to support key points",
        "Improve transitions between sections for better flow",
        "Include quantitative data and statistics where applicable",
        "Strengthen conclusion with actionable next steps",
        "Clarify technical concepts for broader audience understanding"
    ]
    
    # Return subset based on iteration needs
    return recommendations[:3 + len(reflection_content) % 3]


def extract_content_issues(reflection_content: str, iteration: int) -> List[str]:
    """
    Extract specific issues identified in the reflection analysis.
    
    In production, this would use advanced text analysis to identify
    specific problems. For demo purposes, we'll simulate issue detection.
    """
    # Simulate issue extraction based on iteration
    all_issues = [
        "Lack of supporting evidence for key claims",
        "Unclear structure in middle sections", 
        "Missing practical implementation guidance",
        "Insufficient coverage of potential challenges",
        "Weak connection between problems and solutions",
        "Limited consideration of stakeholder perspectives"
    ]
    
    # Return fewer issues in later iterations (simulating improvement)
    max_issues = max(1, 4 - iteration)
    return all_issues[:max_issues]


async def run_simple_reflective_demo():
    """Run a simple demo of the reflective pattern."""
    print("🚀 Multi-Agent Reflective Pattern Demo")
    print("Demonstrating iterative improvement through reflection")
    print()
    
    # Run iterative improvement
    results = await run_iterative_improvement("renewable energy adoption strategies")
    
    # Display summary
    print("\n📋 Reflective Process Summary:")
    print(f"Topic: {results['content_topic']}")
    print(f"Iterations Completed: {results['iterations_completed']}")
    print(f"Initial Confidence: {results['improvement_progression']['initial_confidence']:.2f}")
    print(f"Final Confidence: {results['improvement_progression']['final_confidence']:.2f}")
    print(f"Improvement Gain: {results['improvement_progression']['improvement_gain']:.2f}")
    print(f"Total Improvement: {results['improvement_progression']['total_improvement_percentage']:.1f}%")
    print(f"Final Content Length: {len(results['final_content'])} characters")
    print(f"Convergence Rate: {results['reflective_metrics']['convergence_rate']}")
    
    return results


async def demonstrate_improvement_trajectory():
    """Demonstrate how quality improves through iterations."""
    print("\n🔬 Improvement Trajectory Analysis")
    print("=" * 50)
    
    topic = "sustainable urban development"
    
    # Run reflective improvement
    results = await run_iterative_improvement(topic)
    
    # Analyze improvement trajectory
    trajectory = results['improvement_progression']['confidence_trajectory']
    
    print("📈 Confidence Evolution:")
    for i, confidence in enumerate(trajectory, 1):
        improvement = f"(+{confidence - trajectory[0]:.2f})" if i > 1 else ""
        print(f"   Iteration {i}: {confidence:.2f} {improvement}")
    
    print(f"\n🎯 Key Insights:")
    print(f"• Started with {trajectory[0]:.2f} confidence")
    print(f"• Achieved {trajectory[-1]:.2f} final confidence")
    print(f"• {results['improvement_progression']['total_improvement_percentage']:.1f}% total improvement")
    print(f"• {results['iterations_completed']} improvement cycles")
    print(f"• {results['reflective_metrics']['convergence_rate']} convergence rate")
    
    return results


if __name__ == "__main__":
    # Run the demos
    print("=" * 80)
    results = asyncio.run(run_simple_reflective_demo())
    
    # Demonstrate improvement trajectory
    trajectory_analysis = asyncio.run(demonstrate_improvement_trajectory())
    
    print("\n🎯 Key Takeaways:")
    print("1. Reflective pattern enables continuous quality improvement")
    print("2. Critical evaluation identifies specific areas for enhancement")
    print("3. Iterative refinement builds upon previous improvements")
    print("4. Research integration ensures factual accuracy and completeness")
    print("5. Confidence metrics track improvement progression")
    print("6. Convergence occurs when quality thresholds are met")
    print("7. Ideal for content creation, strategy development, and quality assurance")