# nanda_hub.py
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import uuid


@dataclass
class AgentRegistration:
    agent_id: str
    agent_type: str
    display_name: str
    personality_summary: str = ""


@dataclass
class A2AMessage:
    message_id: str
    from_agent_id: str
    to_agent_id: str
    payload: Dict[str, Any]


class NandaHub:
    """
    Very simple in-memory hub for:
    - registering agents
    - sending A2A messages
    - pulling messages from an agent's inbox
    """

    def __init__(self) -> None:
        self.agents: Dict[str, AgentRegistration] = {}
        self.inboxes: Dict[str, List[A2AMessage]] = {}

    # ---------- Agent registration ----------

    def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        display_name: str,
        personality_summary: str = "",
    ) -> AgentRegistration:
        reg = AgentRegistration(
            agent_id=agent_id,
            agent_type=agent_type,
            display_name=display_name,
            personality_summary=personality_summary,
        )
        self.agents[agent_id] = reg
        # ensure inbox exists
        if agent_id not in self.inboxes:
            self.inboxes[agent_id] = []
        return reg

    def get_agent(self, agent_id: str) -> Optional[AgentRegistration]:
        return self.agents.get(agent_id)

    def list_agents(self) -> List[AgentRegistration]:
        return list(self.agents.values())

    # ---------- A2A messaging ----------

    def send_message(
        self,
        from_agent_id: str,
        to_agent_id: str,
        payload: Dict[str, Any],
    ) -> A2AMessage:
        if to_agent_id not in self.inboxes:
            # create inbox automatically
            self.inboxes[to_agent_id] = []
        msg = A2AMessage(
            message_id=str(uuid.uuid4()),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            payload=payload,
        )
        self.inboxes[to_agent_id].append(msg)
        return msg

    def get_inbox(
        self,
        agent_id: str,
        clear: bool = True,
    ) -> List[A2AMessage]:
        msgs = list(self.inboxes.get(agent_id, []))
        if clear:
            self.inboxes[agent_id] = []
        return msgs


# Global singleton instance used by api_server.py
hub = NandaHub()
