"""
Workflow Visualizer

Creates visual representations of workflows and execution states.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .graph_builder import WorkflowGraph, NodeType, EdgeType
from .workflow_executor import WorkflowExecution, ExecutionState


@dataclass
class VisualizationConfig:
    """Configuration for workflow visualization."""
    include_metadata: bool = True
    show_timing: bool = True
    show_agent_details: bool = True
    color_by_status: bool = True
    layout: str = "hierarchical"  # hierarchical, circular, force-directed


class WorkflowVisualizer:
    """
    Creates visual representations of workflows.
    
    Supports multiple output formats:
    - Mermaid diagrams
    - Graphviz DOT format
    - JSON for web-based visualizers
    - HTML interactive diagrams
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize workflow visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
    
    def generate_mermaid(self, graph: WorkflowGraph, 
                        execution: WorkflowExecution = None) -> str:
        """
        Generate Mermaid diagram representation.
        
        Args:
            graph: Workflow graph to visualize
            execution: Optional execution state for status coloring
            
        Returns:
            Mermaid diagram string
        """
        lines = ["graph TD"]
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            node_shape, node_label = self._get_mermaid_node_info(node, execution)
            lines.append(f"    {node_id}{node_shape}")
            
        # Add edges
        for edge in graph.edges.values():
            edge_label = self._get_mermaid_edge_label(edge)
            arrow = self._get_mermaid_arrow(edge.edge_type)
            
            if edge_label:
                lines.append(f"    {edge.source_node} {arrow}|{edge_label}| {edge.target_node}")
            else:
                lines.append(f"    {edge.source_node} {arrow} {edge.target_node}")
        
        # Add styling if execution provided
        if execution and self.config.color_by_status:
            lines.extend(self._get_mermaid_styling(execution))
        
        return "\\n".join(lines)
    
    def _get_mermaid_node_info(self, node, execution=None) -> tuple:
        """Get Mermaid node shape and label."""
        label = node.name
        
        # Add execution info if available
        if execution and node.id in execution.node_executions:
            node_exec = execution.node_executions[node.id]
            if self.config.show_timing and node_exec.start_time:
                duration = ""
                if node_exec.end_time:
                    duration = f" ({(node_exec.end_time - node_exec.start_time).total_seconds():.1f}s)"
                label += duration
        
        # Choose shape based on node type
        if node.node_type == NodeType.START:
            return f"[{label}]", label
        elif node.node_type == NodeType.END:
            return f"[{label}]", label
        elif node.node_type == NodeType.AGENT:
            return f"({label})", label
        elif node.node_type == NodeType.CONDITION:
            return f"{{{label}}}", label
        elif node.node_type == NodeType.PARALLEL:
            return f"[/{label}/]", label
        else:
            return f"[{label}]", label
    
    def _get_mermaid_edge_label(self, edge) -> str:
        """Get edge label for Mermaid."""
        if edge.edge_type == EdgeType.CONDITIONAL and edge.condition:
            return edge.condition
        elif edge.edge_type == EdgeType.PARALLEL_BRANCH:
            return "parallel"
        elif edge.edge_type == EdgeType.MERGE:
            return "merge"
        return ""
    
    def _get_mermaid_arrow(self, edge_type: EdgeType) -> str:
        """Get arrow style for edge type."""
        if edge_type == EdgeType.CONDITIONAL:
            return "-.->"]
        elif edge_type in [EdgeType.PARALLEL_BRANCH, EdgeType.MERGE]:
            return "==>"
        else:
            return "-->"
    
    def _get_mermaid_styling(self, execution: WorkflowExecution) -> List[str]:
        """Get Mermaid styling for execution states."""
        styles = []
        
        for node_id, node_exec in execution.node_executions.items():
            if node_exec.state == ExecutionState.COMPLETED:
                styles.append(f"    classDef completed fill:#90EE90")
                styles.append(f"    class {node_id} completed")
            elif node_exec.state == ExecutionState.RUNNING:
                styles.append(f"    classDef running fill:#FFD700")
                styles.append(f"    class {node_id} running")
            elif node_exec.state == ExecutionState.FAILED:
                styles.append(f"    classDef failed fill:#FFB6C1")
                styles.append(f"    class {node_id} failed")
        
        return styles
    
    def generate_graphviz(self, graph: WorkflowGraph, 
                         execution: WorkflowExecution = None) -> str:
        """
        Generate Graphviz DOT representation.
        
        Args:
            graph: Workflow graph to visualize
            execution: Optional execution state
            
        Returns:
            DOT format string
        """
        lines = [
            "digraph workflow {",
            "    rankdir=TD;",
            "    node [fontname=\"Arial\", fontsize=10];",
            "    edge [fontname=\"Arial\", fontsize=8];"
        ]
        
        # Add nodes
        for node_id, node in graph.nodes.items():
            attributes = self._get_graphviz_node_attributes(node, execution)
            lines.append(f"    {node_id} {attributes};")
        
        # Add edges
        for edge in graph.edges.values():
            attributes = self._get_graphviz_edge_attributes(edge)
            lines.append(f"    {edge.source_node} -> {edge.target_node} {attributes};")
        
        lines.append("}")
        return "\\n".join(lines)
    
    def _get_graphviz_node_attributes(self, node, execution=None) -> str:
        """Get Graphviz node attributes."""
        attrs = [f'label="{node.name}"']
        
        # Shape based on node type
        if node.node_type == NodeType.START:
            attrs.append('shape=ellipse, style=filled, fillcolor=lightgreen')
        elif node.node_type == NodeType.END:
            attrs.append('shape=ellipse, style=filled, fillcolor=lightcoral')
        elif node.node_type == NodeType.AGENT:
            attrs.append('shape=box, style=filled, fillcolor=lightblue')
        elif node.node_type == NodeType.CONDITION:
            attrs.append('shape=diamond, style=filled, fillcolor=lightyellow')
        elif node.node_type == NodeType.PARALLEL:
            attrs.append('shape=parallelogram, style=filled, fillcolor=lightgray')
        
        # Add execution state coloring
        if execution and node.id in execution.node_executions:
            node_exec = execution.node_executions[node.id]
            if node_exec.state == ExecutionState.COMPLETED:
                attrs.append('fillcolor=green')
            elif node_exec.state == ExecutionState.RUNNING:
                attrs.append('fillcolor=yellow')
            elif node_exec.state == ExecutionState.FAILED:
                attrs.append('fillcolor=red')
        
        return f"[{', '.join(attrs)}]"
    
    def _get_graphviz_edge_attributes(self, edge) -> str:
        """Get Graphviz edge attributes."""
        attrs = []
        
        if edge.edge_type == EdgeType.CONDITIONAL:
            attrs.append('style=dashed')
            if edge.condition:
                attrs.append(f'label="{edge.condition}"')
        elif edge.edge_type == EdgeType.PARALLEL_BRANCH:
            attrs.append('color=blue, style=bold')
            attrs.append('label="parallel"')
        elif edge.edge_type == EdgeType.MERGE:
            attrs.append('color=purple')
            attrs.append('label="merge"')
        
        return f"[{', '.join(attrs)}]" if attrs else ""
    
    def generate_json_for_web(self, graph: WorkflowGraph, 
                            execution: WorkflowExecution = None) -> str:
        """
        Generate JSON representation for web-based visualizers.
        
        Args:
            graph: Workflow graph
            execution: Optional execution state
            
        Returns:
            JSON string for web visualization
        """
        nodes = []
        edges = []
        
        # Convert nodes
        for node_id, node in graph.nodes.items():
            node_data = {
                "id": node_id,
                "label": node.name,
                "type": node.node_type.value,
                "group": node.node_type.value
            }
            
            # Add execution state
            if execution and node_id in execution.node_executions:
                node_exec = execution.node_executions[node_id]
                node_data["status"] = node_exec.state.value
                
                if node_exec.start_time:
                    node_data["start_time"] = node_exec.start_time.isoformat()
                if node_exec.end_time:
                    node_data["end_time"] = node_exec.end_time.isoformat()
                    duration = (node_exec.end_time - node_exec.start_time).total_seconds()
                    node_data["duration"] = duration
            
            # Add metadata if configured
            if self.config.include_metadata:
                node_data["metadata"] = node.metadata
                if node.agent_id:
                    node_data["agent_id"] = node.agent_id
            
            nodes.append(node_data)
        
        # Convert edges
        for edge_id, edge in graph.edges.items():
            edge_data = {
                "id": edge_id,
                "from": edge.source_node,
                "to": edge.target_node,
                "type": edge.edge_type.value,
                "weight": edge.weight
            }
            
            if edge.condition:
                edge_data["condition"] = edge.condition
            
            if self.config.include_metadata:
                edge_data["metadata"] = edge.metadata
            
            edges.append(edge_data)
        
        # Create complete visualization data
        viz_data = {
            "workflow": {
                "id": graph.id,
                "name": graph.name,
                "description": graph.description
            },
            "nodes": nodes,
            "edges": edges,
            "layout": self.config.layout
        }
        
        # Add execution summary if available
        if execution:
            viz_data["execution"] = {
                "id": execution.execution_id,
                "state": execution.state.value,
                "start_time": execution.start_time.isoformat() if execution.start_time else None,
                "end_time": execution.end_time.isoformat() if execution.end_time else None
            }
        
        return json.dumps(viz_data, indent=2)
    
    def generate_html_interactive(self, graph: WorkflowGraph, 
                                execution: WorkflowExecution = None) -> str:
        """
        Generate interactive HTML visualization using vis.js.
        
        Args:
            graph: Workflow graph
            execution: Optional execution state
            
        Returns:
            Complete HTML page with interactive visualization
        """
        json_data = self.generate_json_for_web(graph, execution)
        
        html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Workflow Visualization: {graph.name}</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        #workflow-container {{
            width: 100%;
            height: 600px;
            border: 1px solid lightgray;
        }}
        .info-panel {{
            margin: 20px 0;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <h1>Workflow: {graph.name}</h1>
    <div class="info-panel">
        <p><strong>Description:</strong> {graph.description}</p>
        <p><strong>Nodes:</strong> {len(graph.nodes)} | <strong>Edges:</strong> {len(graph.edges)}</p>
    </div>
    
    <div id="workflow-container"></div>
    
    <script type="text/javascript">
        const workflowData = {json_data};
        
        // Convert data for vis.js
        const nodes = new vis.DataSet(workflowData.nodes.map(node => ({{
            id: node.id,
            label: node.label,
            group: node.type,
            color: getNodeColor(node.status || 'pending'),
            title: getNodeTooltip(node)
        }})));
        
        const edges = new vis.DataSet(workflowData.edges.map(edge => ({{
            from: edge.from,
            to: edge.to,
            label: edge.condition || '',
            arrows: 'to',
            color: getEdgeColor(edge.type)
        }})));
        
        function getNodeColor(status) {{
            switch(status) {{
                case 'completed': return '#90EE90';
                case 'running': return '#FFD700';
                case 'failed': return '#FFB6C1';
                default: return '#E6E6FA';
            }}
        }}
        
        function getEdgeColor(type) {{
            switch(type) {{
                case 'conditional': return '#FF6B6B';
                case 'parallel_branch': return '#4ECDC4';
                case 'merge': return '#45B7D1';
                default: return '#95A5A6';
            }}
        }}
        
        function getNodeTooltip(node) {{
            let tooltip = `Type: ${{node.type}}`;
            if (node.agent_id) tooltip += `\\nAgent: ${{node.agent_id}}`;
            if (node.status) tooltip += `\\nStatus: ${{node.status}}`;
            if (node.duration) tooltip += `\\nDuration: ${{node.duration.toFixed(2)}}s`;
            return tooltip;
        }}
        
        // Network options
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                    nodeSpacing: 150
                }}
            }},
            physics: {{
                enabled: false
            }},
            nodes: {{
                shape: 'box',
                margin: 10,
                font: {{ size: 12 }}
            }},
            edges: {{
                font: {{ size: 10, align: 'middle' }},
                arrows: {{ to: {{ scaleFactor: 0.5 }} }}
            }}
        }};
        
        // Create network
        const container = document.getElementById('workflow-container');
        const data = {{ nodes: nodes, edges: edges }};
        const network = new vis.Network(container, data, options);
        
        // Add event listeners
        network.on('selectNode', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const nodeData = workflowData.nodes.find(n => n.id === nodeId);
                console.log('Selected node:', nodeData);
            }}
        }});
    </script>
</body>
</html>
        """
        
        return html_template
    
    def save_visualization(self, graph: WorkflowGraph, output_path: str, 
                         format: str = "html", execution: WorkflowExecution = None):
        """
        Save visualization to file.
        
        Args:
            graph: Workflow graph
            output_path: Path to save file
            format: Output format (html, mermaid, dot, json)
            execution: Optional execution state
        """
        if format.lower() == "html":
            content = self.generate_html_interactive(graph, execution)
        elif format.lower() == "mermaid":
            content = self.generate_mermaid(graph, execution)
        elif format.lower() == "dot":
            content = self.generate_graphviz(graph, execution)
        elif format.lower() == "json":
            content = self.generate_json_for_web(graph, execution)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)