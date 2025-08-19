"""
State Manager

In-memory state management for multi-agent workflows and pattern execution.
Handles workflow context, conversation history, and execution state.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json


class StateManager:
    """
    In-memory state manager for workflow and agent context.
    
    The state manager:
    - Stores workflow execution state and context
    - Manages conversation history between agents
    - Provides state persistence and retrieval
    - Handles state cleanup and expiration
    - Supports state snapshots and rollbacks
    """

    def __init__(self, retention_hours: int = 24):
        """
        Initialize the state manager.
        
        Args:
            retention_hours: Hours to retain state before cleanup
        """
        self.retention_hours = retention_hours
        
        # State storage
        self.workflow_states: Dict[str, Dict[str, Any]] = {}
        self.agent_conversations: Dict[str, List[Dict[str, Any]]] = {}
        self.pattern_contexts: Dict[str, Dict[str, Any]] = {}
        self.state_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        
        # Metadata
        self.state_metadata = {
            "created_at": datetime.now(),
            "last_cleanup": datetime.now(),
            "total_states": 0,
            "active_workflows": 0
        }

    async def store_state(self, workflow_id: str, state_data: Dict[str, Any], 
                         state_type: str = "workflow") -> bool:
        """
        Store state data for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            state_data: State data to store
            state_type: Type of state (workflow, pattern, agent)
            
        Returns:
            True if stored successfully
        """
        try:
            timestamp = datetime.now()
            
            state_entry = {
                "workflow_id": workflow_id,
                "state_type": state_type,
                "data": state_data,
                "timestamp": timestamp,
                "ttl": timestamp + timedelta(hours=self.retention_hours)
            }
            
            # Store based on type
            if state_type == "workflow":
                if workflow_id not in self.workflow_states:
                    self.workflow_states[workflow_id] = {"states": [], "metadata": {}}
                
                self.workflow_states[workflow_id]["states"].append(state_entry)
                self.workflow_states[workflow_id]["metadata"].update({
                    "last_updated": timestamp,
                    "state_count": len(self.workflow_states[workflow_id]["states"])
                })
                
            elif state_type == "pattern":
                if workflow_id not in self.pattern_contexts:
                    self.pattern_contexts[workflow_id] = {"contexts": [], "metadata": {}}
                
                self.pattern_contexts[workflow_id]["contexts"].append(state_entry)
                self.pattern_contexts[workflow_id]["metadata"].update({
                    "last_updated": timestamp,
                    "context_count": len(self.pattern_contexts[workflow_id]["contexts"])
                })
            
            # Update global metadata
            self.state_metadata["total_states"] += 1
            if state_data.get("status") == "running":
                self.state_metadata["active_workflows"] += 1
            
            print(f"[STATE_MANAGER] Stored {state_type} state for {workflow_id}")
            return True
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to store state: {str(e)}")
            return False

    async def get_state(self, workflow_id: str, state_type: str = "workflow") -> Optional[Dict[str, Any]]:
        """
        Retrieve the latest state for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            state_type: Type of state to retrieve
            
        Returns:
            Latest state data or None if not found
        """
        try:
            if state_type == "workflow" and workflow_id in self.workflow_states:
                states = self.workflow_states[workflow_id]["states"]
                if states:
                    return states[-1]["data"]  # Return latest state
                    
            elif state_type == "pattern" and workflow_id in self.pattern_contexts:
                contexts = self.pattern_contexts[workflow_id]["contexts"]
                if contexts:
                    return contexts[-1]["data"]  # Return latest context
            
            return None
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to retrieve state: {str(e)}")
            return None

    async def get_state_history(self, workflow_id: str, 
                              state_type: str = "workflow",
                              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get state history for a workflow.
        
        Args:
            workflow_id: Workflow identifier
            state_type: Type of state history
            limit: Maximum number of states to return
            
        Returns:
            List of historical states
        """
        try:
            history = []
            
            if state_type == "workflow" and workflow_id in self.workflow_states:
                states = self.workflow_states[workflow_id]["states"]
                history = [
                    {
                        "timestamp": state["timestamp"].isoformat(),
                        "data": state["data"],
                        "state_type": state["state_type"]
                    }
                    for state in states
                ]
                
            elif state_type == "pattern" and workflow_id in self.pattern_contexts:
                contexts = self.pattern_contexts[workflow_id]["contexts"]
                history = [
                    {
                        "timestamp": context["timestamp"].isoformat(),
                        "data": context["data"],
                        "state_type": context["state_type"]
                    }
                    for context in contexts
                ]
            
            # Sort by timestamp (newest first) and apply limit
            history.sort(key=lambda x: x["timestamp"], reverse=True)
            if limit:
                history = history[:limit]
            
            return history
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to retrieve state history: {str(e)}")
            return []

    async def store_conversation(self, conversation_id: str, 
                               sender_id: str, recipient_id: str,
                               message: str, message_type: str = "info") -> bool:
        """
        Store a conversation message between agents.
        
        Args:
            conversation_id: Conversation identifier
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message: Message content
            message_type: Type of message
            
        Returns:
            True if stored successfully
        """
        try:
            if conversation_id not in self.agent_conversations:
                self.agent_conversations[conversation_id] = []
            
            conversation_entry = {
                "message_id": f"{conversation_id}_{len(self.agent_conversations[conversation_id])}",
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now(),
                "ttl": datetime.now() + timedelta(hours=self.retention_hours)
            }
            
            self.agent_conversations[conversation_id].append(conversation_entry)
            
            print(f"[STATE_MANAGER] Stored conversation message: {sender_id} -> {recipient_id}")
            return True
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to store conversation: {str(e)}")
            return False

    async def get_conversation_history(self, conversation_id: str, 
                                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            conversation_id: Conversation identifier
            limit: Maximum number of messages to return
            
        Returns:
            List of conversation messages
        """
        try:
            if conversation_id not in self.agent_conversations:
                return []
            
            messages = self.agent_conversations[conversation_id]
            
            # Convert to serializable format
            history = [
                {
                    "message_id": msg["message_id"],
                    "sender_id": msg["sender_id"],
                    "recipient_id": msg["recipient_id"],
                    "message": msg["message"],
                    "message_type": msg["message_type"],
                    "timestamp": msg["timestamp"].isoformat()
                }
                for msg in messages
            ]
            
            # Sort by timestamp and apply limit
            history.sort(key=lambda x: x["timestamp"])
            if limit:
                history = history[-limit:]  # Get most recent messages
            
            return history
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to retrieve conversation history: {str(e)}")
            return []

    async def create_snapshot(self, snapshot_id: str, workflow_ids: List[str]) -> bool:
        """
        Create a snapshot of workflow states.
        
        Args:
            snapshot_id: Snapshot identifier
            workflow_ids: List of workflow IDs to include in snapshot
            
        Returns:
            True if snapshot created successfully
        """
        try:
            snapshot_data = {
                "snapshot_id": snapshot_id,
                "created_at": datetime.now(),
                "workflow_states": {},
                "pattern_contexts": {},
                "conversations": {}
            }
            
            # Capture workflow states
            for workflow_id in workflow_ids:
                if workflow_id in self.workflow_states:
                    snapshot_data["workflow_states"][workflow_id] = self.workflow_states[workflow_id]
                
                if workflow_id in self.pattern_contexts:
                    snapshot_data["pattern_contexts"][workflow_id] = self.pattern_contexts[workflow_id]
                
                if workflow_id in self.agent_conversations:
                    snapshot_data["conversations"][workflow_id] = self.agent_conversations[workflow_id]
            
            if snapshot_id not in self.state_snapshots:
                self.state_snapshots[snapshot_id] = []
            
            self.state_snapshots[snapshot_id].append(snapshot_data)
            
            print(f"[STATE_MANAGER] Created snapshot {snapshot_id} for {len(workflow_ids)} workflows")
            return True
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to create snapshot: {str(e)}")
            return False

    async def restore_snapshot(self, snapshot_id: str, snapshot_index: int = -1) -> bool:
        """
        Restore state from a snapshot.
        
        Args:
            snapshot_id: Snapshot identifier
            snapshot_index: Index of snapshot to restore (-1 for latest)
            
        Returns:
            True if restored successfully
        """
        try:
            if snapshot_id not in self.state_snapshots:
                return False
            
            snapshots = self.state_snapshots[snapshot_id]
            if not snapshots or abs(snapshot_index) > len(snapshots):
                return False
            
            snapshot_data = snapshots[snapshot_index]
            
            # Restore states
            if "workflow_states" in snapshot_data:
                self.workflow_states.update(snapshot_data["workflow_states"])
            
            if "pattern_contexts" in snapshot_data:
                self.pattern_contexts.update(snapshot_data["pattern_contexts"])
            
            if "conversations" in snapshot_data:
                self.agent_conversations.update(snapshot_data["conversations"])
            
            print(f"[STATE_MANAGER] Restored snapshot {snapshot_id}")
            return True
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to restore snapshot: {str(e)}")
            return False

    async def cleanup_expired_states(self) -> Dict[str, int]:
        """
        Clean up expired states and conversations.
        
        Returns:
            Cleanup statistics
        """
        current_time = datetime.now()
        cleanup_stats = {
            "workflow_states_cleaned": 0,
            "pattern_contexts_cleaned": 0,
            "conversations_cleaned": 0
        }
        
        try:
            # Clean workflow states
            for workflow_id in list(self.workflow_states.keys()):
                states = self.workflow_states[workflow_id]["states"]
                original_count = len(states)
                
                # Filter non-expired states
                unexpired_states = [s for s in states if s["ttl"] > current_time]
                
                if unexpired_states:
                    self.workflow_states[workflow_id]["states"] = unexpired_states
                    self.workflow_states[workflow_id]["metadata"]["state_count"] = len(unexpired_states)
                else:
                    del self.workflow_states[workflow_id]
                
                cleanup_stats["workflow_states_cleaned"] += original_count - len(unexpired_states)
            
            # Clean pattern contexts
            for workflow_id in list(self.pattern_contexts.keys()):
                contexts = self.pattern_contexts[workflow_id]["contexts"]
                original_count = len(contexts)
                
                unexpired_contexts = [c for c in contexts if c["ttl"] > current_time]
                
                if unexpired_contexts:
                    self.pattern_contexts[workflow_id]["contexts"] = unexpired_contexts
                    self.pattern_contexts[workflow_id]["metadata"]["context_count"] = len(unexpired_contexts)
                else:
                    del self.pattern_contexts[workflow_id]
                
                cleanup_stats["pattern_contexts_cleaned"] += original_count - len(unexpired_contexts)
            
            # Clean conversations
            for conversation_id in list(self.agent_conversations.keys()):
                messages = self.agent_conversations[conversation_id]
                original_count = len(messages)
                
                unexpired_messages = [m for m in messages if m["ttl"] > current_time]
                
                if unexpired_messages:
                    self.agent_conversations[conversation_id] = unexpired_messages
                else:
                    del self.agent_conversations[conversation_id]
                
                cleanup_stats["conversations_cleaned"] += original_count - len(unexpired_messages)
            
            self.state_metadata["last_cleanup"] = current_time
            
            total_cleaned = sum(cleanup_stats.values())
            if total_cleaned > 0:
                print(f"[STATE_MANAGER] Cleaned up {total_cleaned} expired state entries")
            
            return cleanup_stats
            
        except Exception as e:
            print(f"[STATE_MANAGER] Failed to cleanup states: {str(e)}")
            return cleanup_stats

    def get_state_metrics(self) -> Dict[str, Any]:
        """Get state management metrics."""
        # Calculate current active states
        active_workflows = len([
            wf_id for wf_id, wf_data in self.workflow_states.items()
            if wf_data["states"] and wf_data["states"][-1]["data"].get("status") == "running"
        ])
        
        return {
            "state_manager_info": {
                "created_at": self.state_metadata["created_at"].isoformat(),
                "last_cleanup": self.state_metadata["last_cleanup"].isoformat(),
                "retention_hours": self.retention_hours
            },
            "current_state": {
                "workflow_states": len(self.workflow_states),
                "pattern_contexts": len(self.pattern_contexts), 
                "active_conversations": len(self.agent_conversations),
                "snapshots": len(self.state_snapshots),
                "active_workflows": active_workflows
            },
            "storage_details": {
                "total_workflow_state_entries": sum(len(wf["states"]) for wf in self.workflow_states.values()),
                "total_pattern_context_entries": sum(len(pc["contexts"]) for pc in self.pattern_contexts.values()),
                "total_conversation_messages": sum(len(conv) for conv in self.agent_conversations.values())
            }
        }

    def clear_all_state(self):
        """Clear all stored state data."""
        self.workflow_states.clear()
        self.agent_conversations.clear()
        self.pattern_contexts.clear()
        self.state_snapshots.clear()
        
        self.state_metadata.update({
            "last_cleanup": datetime.now(),
            "total_states": 0,
            "active_workflows": 0
        })
        
        print("[STATE_MANAGER] All state data cleared")

    def export_state(self, workflow_id: str) -> Dict[str, Any]:
        """Export state for a specific workflow (for debugging/analysis)."""
        exported_state = {
            "workflow_id": workflow_id,
            "exported_at": datetime.now().isoformat(),
            "workflow_states": [],
            "pattern_contexts": [],
            "conversations": []
        }
        
        # Export workflow states
        if workflow_id in self.workflow_states:
            states = self.workflow_states[workflow_id]["states"]
            exported_state["workflow_states"] = [
                {
                    "timestamp": state["timestamp"].isoformat(),
                    "state_type": state["state_type"],
                    "data": state["data"]
                }
                for state in states
            ]
        
        # Export pattern contexts
        if workflow_id in self.pattern_contexts:
            contexts = self.pattern_contexts[workflow_id]["contexts"]
            exported_state["pattern_contexts"] = [
                {
                    "timestamp": context["timestamp"].isoformat(),
                    "state_type": context["state_type"],
                    "data": context["data"]
                }
                for context in contexts
            ]
        
        # Export conversations
        if workflow_id in self.agent_conversations:
            messages = self.agent_conversations[workflow_id]
            exported_state["conversations"] = [
                {
                    "message_id": msg["message_id"],
                    "sender_id": msg["sender_id"],
                    "recipient_id": msg["recipient_id"],
                    "message": msg["message"],
                    "message_type": msg["message_type"],
                    "timestamp": msg["timestamp"].isoformat()
                }
                for msg in messages
            ]
        
        return exported_state