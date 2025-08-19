"""
Workflow Graph Builder

Creates and manages complex workflow graphs using LangGraph-inspired patterns.
"""

import json
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class NodeType(Enum):
    """Types of nodes in workflow graph."""
    AGENT = "agent"
    CONDITION = "condition"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    START = "start"
    END = "end"


class EdgeType(Enum):
    """Types of edges in workflow graph."""
    NORMAL = "normal"
    CONDITIONAL = "conditional"
    PARALLEL_BRANCH = "parallel_branch"
    MERGE = "merge"


@dataclass
class WorkflowNode:
    """Node in workflow graph."""
    id: str
    name: str
    node_type: NodeType
    agent_id: Optional[str] = None
    condition_func: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class WorkflowEdge:
    """Edge in workflow graph."""
    id: str
    source_node: str
    target_node: str
    edge_type: EdgeType
    condition: Optional[str] = None
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowGraph:
    """Complete workflow graph structure."""
    id: str
    name: str
    description: str
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    edges: Dict[str, WorkflowEdge] = field(default_factory=dict)
    start_node: Optional[str] = None
    end_nodes: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowGraphBuilder:
    """
    Builder class for creating complex workflow graphs.
    
    Provides a fluent interface for building multi-agent workflows with
    conditional branching, parallel execution, and complex routing logic.
    """
    
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        """
        Initialize workflow graph builder.
        
        Args:
            workflow_id: Unique identifier for the workflow
            name: Human-readable name
            description: Workflow description
        """
        self.graph = WorkflowGraph(
            id=workflow_id,
            name=name,
            description=description
        )
        self._node_counter = 0
        self._edge_counter = 0
    
    def add_start_node(self, node_id: str = None, name: str = "Start") -> 'WorkflowGraphBuilder':
        """Add start node to workflow."""
        if node_id is None:
            node_id = f"start_{self._node_counter}"
            self._node_counter += 1
        
        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=NodeType.START
        )
        
        self.graph.nodes[node_id] = node
        self.graph.start_node = node_id
        return self
    
    def add_end_node(self, node_id: str = None, name: str = "End") -> 'WorkflowGraphBuilder':
        """Add end node to workflow."""
        if node_id is None:
            node_id = f"end_{self._node_counter}"
            self._node_counter += 1
        
        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=NodeType.END
        )
        
        self.graph.nodes[node_id] = node
        self.graph.end_nodes.add(node_id)
        return self
    
    def add_agent_node(self, agent_id: str, node_id: str = None, 
                      name: str = None, **metadata) -> 'WorkflowGraphBuilder':
        """Add agent processing node."""
        if node_id is None:
            node_id = f"agent_{agent_id}_{self._node_counter}"
            self._node_counter += 1
        
        if name is None:
            name = f"Agent {agent_id}"
        
        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=NodeType.AGENT,
            agent_id=agent_id,
            metadata=metadata
        )
        
        self.graph.nodes[node_id] = node
        return self
    
    def add_condition_node(self, condition_func: Callable, node_id: str = None,
                          name: str = "Condition", **metadata) -> 'WorkflowGraphBuilder':
        """Add conditional branching node."""
        if node_id is None:
            node_id = f"condition_{self._node_counter}"
            self._node_counter += 1
        
        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=NodeType.CONDITION,
            condition_func=condition_func,
            metadata=metadata
        )
        
        self.graph.nodes[node_id] = node
        return self
    
    def add_parallel_node(self, node_id: str = None, name: str = "Parallel",
                         **metadata) -> 'WorkflowGraphBuilder':
        """Add parallel execution node."""
        if node_id is None:
            node_id = f"parallel_{self._node_counter}"
            self._node_counter += 1
        
        node = WorkflowNode(
            id=node_id,
            name=name,
            node_type=NodeType.PARALLEL,
            metadata=metadata
        )
        
        self.graph.nodes[node_id] = node
        return self
    
    def add_edge(self, source: str, target: str, edge_type: EdgeType = EdgeType.NORMAL,
                condition: str = None, weight: float = 1.0, **metadata) -> 'WorkflowGraphBuilder':
        """Add edge between nodes."""
        edge_id = f"edge_{self._edge_counter}"
        self._edge_counter += 1
        
        if source not in self.graph.nodes:
            raise ValueError(f"Source node '{source}' not found in graph")
        if target not in self.graph.nodes:
            raise ValueError(f"Target node '{target}' not found in graph")
        
        edge = WorkflowEdge(
            id=edge_id,
            source_node=source,
            target_node=target,
            edge_type=edge_type,
            condition=condition,
            weight=weight,
            metadata=metadata
        )
        
        self.graph.edges[edge_id] = edge
        return self
    
    def add_conditional_edge(self, source: str, target: str, condition: str,
                           weight: float = 1.0, **metadata) -> 'WorkflowGraphBuilder':
        """Add conditional edge with condition string."""
        return self.add_edge(
            source, target, EdgeType.CONDITIONAL, 
            condition=condition, weight=weight, **metadata
        )
    
    def add_parallel_edges(self, source: str, targets: List[str],
                          **metadata) -> 'WorkflowGraphBuilder':
        """Add parallel edges to multiple targets."""
        for target in targets:
            self.add_edge(
                source, target, EdgeType.PARALLEL_BRANCH, **metadata
            )
        return self
    
    def create_pipeline(self, agent_ids: List[str], 
                       start_name: str = "Pipeline Start",
                       end_name: str = "Pipeline End") -> 'WorkflowGraphBuilder':
        """Create simple pipeline workflow."""
        # Add start node
        start_id = "pipeline_start"
        self.add_start_node(start_id, start_name)
        
        # Add agent nodes and connect them sequentially
        prev_node = start_id
        for i, agent_id in enumerate(agent_ids):
            node_id = f"agent_{agent_id}"
            self.add_agent_node(agent_id, node_id, f"Agent {agent_id}")
            self.add_edge(prev_node, node_id)
            prev_node = node_id
        
        # Add end node
        end_id = "pipeline_end"
        self.add_end_node(end_id, end_name)
        self.add_edge(prev_node, end_id)
        
        return self
    
    def create_parallel_workflow(self, agent_ids: List[str],
                               start_name: str = "Parallel Start",
                               end_name: str = "Parallel End") -> 'WorkflowGraphBuilder':
        """Create parallel execution workflow."""
        # Add start node
        start_id = "parallel_start"
        self.add_start_node(start_id, start_name)
        
        # Add parallel execution node
        parallel_id = "parallel_exec"
        self.add_parallel_node(parallel_id, "Parallel Execution")
        self.add_edge(start_id, parallel_id)
        
        # Add agent nodes in parallel
        agent_nodes = []
        for agent_id in agent_ids:
            node_id = f"agent_{agent_id}"
            self.add_agent_node(agent_id, node_id, f"Agent {agent_id}")
            agent_nodes.append(node_id)
        
        # Connect parallel node to all agent nodes
        self.add_parallel_edges(parallel_id, agent_nodes)
        
        # Add end node and connect all agents to it
        end_id = "parallel_end"
        self.add_end_node(end_id, end_name)
        for node_id in agent_nodes:
            self.add_edge(node_id, end_id, EdgeType.MERGE)
        
        return self
    
    def validate_graph(self) -> bool:
        """Validate graph structure."""
        # Check for start node
        if not self.graph.start_node:
            return False
        
        # Check for end nodes
        if not self.graph.end_nodes:
            return False
        
        # Check for orphaned nodes
        connected_nodes = set()
        for edge in self.graph.edges.values():
            connected_nodes.add(edge.source_node)
            connected_nodes.add(edge.target_node)
        
        all_nodes = set(self.graph.nodes.keys())
        orphaned = all_nodes - connected_nodes
        
        # Start and end nodes can be "orphaned" if they're only connected by edges
        valid_orphans = {self.graph.start_node} | self.graph.end_nodes
        actual_orphans = orphaned - valid_orphans
        
        return len(actual_orphans) == 0
    
    def get_node_dependencies(self, node_id: str) -> List[str]:
        """Get list of nodes that must complete before this node."""
        dependencies = []
        for edge in self.graph.edges.values():
            if edge.target_node == node_id:
                dependencies.append(edge.source_node)
        return dependencies
    
    def get_node_dependents(self, node_id: str) -> List[str]:
        """Get list of nodes that depend on this node."""
        dependents = []
        for edge in self.graph.edges.values():
            if edge.source_node == node_id:
                dependents.append(edge.target_node)
        return dependents
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert graph to dictionary representation."""
        return {
            "id": self.graph.id,
            "name": self.graph.name,
            "description": self.graph.description,
            "start_node": self.graph.start_node,
            "end_nodes": list(self.graph.end_nodes),
            "created_at": self.graph.created_at.isoformat(),
            "nodes": {
                node_id: {
                    "id": node.id,
                    "name": node.name,
                    "type": node.node_type.value,
                    "agent_id": node.agent_id,
                    "metadata": node.metadata,
                    "created_at": node.created_at.isoformat()
                }
                for node_id, node in self.graph.nodes.items()
            },
            "edges": {
                edge_id: {
                    "id": edge.id,
                    "source": edge.source_node,
                    "target": edge.target_node,
                    "type": edge.edge_type.value,
                    "condition": edge.condition,
                    "weight": edge.weight,
                    "metadata": edge.metadata
                }
                for edge_id, edge in self.graph.edges.items()
            },
            "metadata": self.graph.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert graph to JSON representation."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def build(self) -> WorkflowGraph:
        """Build and return the completed workflow graph."""
        if not self.validate_graph():
            raise ValueError("Invalid graph structure")
        
        return self.graph