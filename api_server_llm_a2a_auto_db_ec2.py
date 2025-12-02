import os
import sqlite3
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from freigent_real_json import RealJsonFreigent
from nanda_hub import hub  # NandaHub singleton (in-memory)


# ----------------------------------------------------------------------
# FastAPI app (EC2-ready variant)
# ----------------------------------------------------------------------
app = FastAPI(
    title="Freigent LLM JSON API + NandaHub A2A AUTO + DB (SQLite) [EC2]"
)


# ----------------------------------------------------------------------
# SQLite setup (profiles & agents persistence)
# ----------------------------------------------------------------------
DB_PATH = os.getenv("FREIGENT_DB_PATH", "freigent.db")


def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Table for agents (Freigents)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS agents (
            agent_id TEXT PRIMARY KEY,
            agent_type TEXT NOT NULL,
            display_name TEXT NOT NULL,
            personality_summary TEXT DEFAULT ''
        )
        """
    )

    # Table for profiles (1:1 with user_id)
    # IMPORTANT: use 'values_text' instead of SQL reserved word 'values'
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            personality TEXT NOT NULL,
            values_text TEXT NOT NULL
        )
        """
    )

    # Table for product experiences (many per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            notes TEXT NOT NULL,
            rating INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES profiles(user_id)
        )
        """
    )

    conn.commit()
    conn.close()


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize DB at import time
init_db()


# ----------------------------------------------------------------------
# In-memory cache of RealJsonFreigent objects (LLM client)
# ----------------------------------------------------------------------
FREIGENTS: Dict[str, RealJsonFreigent] = {}


def get_or_create_freigent(user_id: str) -> RealJsonFreigent:
    if user_id not in FREIGENTS:
        FREIGENTS[user_id] = RealJsonFreigent(agent_id=user_id)
    return FREIGENTS[user_id]


# ----------------------------------------------------------------------
# DB helper functions
# ----------------------------------------------------------------------
def db_upsert_agent(
    agent_id: str,
    agent_type: str,
    display_name: str,
    personality_summary: str,
) -> None:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO agents (agent_id, agent_type, display_name, personality_summary)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            agent_type = excluded.agent_type,
            display_name = excluded.display_name,
            personality_summary = excluded.personality_summary
        """,
        (agent_id, agent_type, display_name, personality_summary),
    )
    conn.commit()
    conn.close()


def db_upsert_profile(user_id: str, profile_dict: Dict[str, Any]) -> None:
    """
    Stores/updates the profile row and replaces all experiences.
    profile_dict structure is:
    {
      "name": ...,
      "personality": ...,
      "values": ...,
      "experiences": [
        {"name": ..., "notes": ..., "rating": ...},
        ...
      ]
    }
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # NOTE: use 'values_text' column in DB
    cur.execute(
        """
        INSERT INTO profiles (user_id, name, personality, values_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            personality = excluded.personality,
            values_text = excluded.values_text
        """,
        (
            user_id,
            profile_dict.get("name", ""),
            profile_dict.get("personality", ""),
            profile_dict.get("values", ""),
        ),
    )

    # Replace experiences for this user
    cur.execute("DELETE FROM experiences WHERE user_id = ?", (user_id,))
    for exp in profile_dict.get("experiences", []):
        cur.execute(
            """
            INSERT INTO experiences (user_id, name, notes, rating)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                exp.get("name", ""),
                exp.get("notes", ""),
                int(exp.get("rating", 0)),
            ),
        )

    conn.commit()
    conn.close()


def db_load_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Loads profile + experiences for a given user_id.
    Returns None if no profile exists.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # NOTE: select values_text and map it back to 'values'
    cur.execute(
        "SELECT user_id, name, personality, values_text FROM profiles WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    profile = {
        "name": row["name"],
        "personality": row["personality"],
        "values": row["values_text"],
        "experiences": [],
    }

    cur.execute(
        "SELECT name, notes, rating FROM experiences WHERE user_id = ?",
        (user_id,),
    )
    exp_rows = cur.fetchall()
    for e in exp_rows:
        profile["experiences"].append(
            {
                "name": e["name"],
                "notes": e["notes"],
                "rating": e["rating"],
            }
        )

    conn.close()
    return profile


def db_list_helper_agent_ids(base_user_id: str) -> List[str]:
    """
    Returns a list of other freigent agent_ids that:
    - are type='freigent'
    - are not base_user_id
    - have a stored profile
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT a.agent_id
        FROM agents a
        JOIN profiles p ON a.agent_id = p.user_id
        WHERE a.agent_type = 'freigent'
          AND a.agent_id != ?
        """,
        (base_user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return [r["agent_id"] for r in rows]


# ----------------------------------------------------------------------
# Pydantic models for Freigent profile & search
# ----------------------------------------------------------------------
class ExperienceModel(BaseModel):
    name: str
    notes: str
    rating: int


class UserProfileModel(BaseModel):
    name: str
    personality: str
    values: str
    experiences: List[ExperienceModel] = []


class SearchRequest(BaseModel):
    query: str


class SearchResponse(BaseModel):
    products: List[Dict[str, Any]]
    summary_for_user: str


# ----------------------------------------------------------------------
# Pydantic models for NandaHub / A2A
# ----------------------------------------------------------------------
class AgentRegisterRequest(BaseModel):
    agent_id: str
    agent_type: str = "freigent"
    display_name: str
    personality_summary: str = ""


class AgentRegisterResponse(BaseModel):
    agent_id: str
    agent_type: str
    display_name: str
    personality_summary: str


class A2ASendRequest(BaseModel):
    from_agent_id: str
    to_agent_id: str
    payload: Dict[str, Any]


class A2AMessageModel(BaseModel):
    message_id: str
    from_agent_id: str
    to_agent_id: str
    payload: Dict[str, Any]


# ----------------------------------------------------------------------
# AUTO multi-agent search response
# ----------------------------------------------------------------------
class HelperResult(BaseModel):
    agent_id: str
    result: Dict[str, Any]


class AutoSearchResponse(BaseModel):
    base_agent_id: str
    helper_agent_ids: List[str]
    base_result: Dict[str, Any]
    helper_results: List[HelperResult]
    merged_products: List[Dict[str, Any]]
    merged_summary_for_user: str


# ----------------------------------------------------------------------
# Worker-style processing for one agent (reads from hub, uses DB profiles)
# ----------------------------------------------------------------------
def process_recommendation_requests_for_agent(
    worker_id: str,
    max_messages: int = 10,
) -> List[Dict[str, Any]]:
    """
    Internal helper (non-HTTP) that processes A2A messages for one agent.

    - Reads messages from worker_id's inbox.
    - For each message with payload.type == 'recommendation_request':
        * Uses the profile from DB (payload.from_user_id or from_agent_id).
        * Calls RealJsonFreigent.generate_recommendations_json(...)
        * Sends back 'recommendation_response' to the requester.
    - Returns a list of processed-message summaries.
    """
    freigent = get_or_create_freigent(worker_id)
    msgs = hub.get_inbox(worker_id, clear=True)
    processed: List[Dict[str, Any]] = []

    for m in msgs[:max_messages]:
        payload = m.payload or {}
        msg_type = payload.get("type")

        if msg_type != "recommendation_request":
            processed.append(
                {
                    "request_message_id": m.message_id,
                    "status": "ignored",
                    "reason": f"Unsupported payload.type '{msg_type}'",
                }
            )
            continue

        from_agent = m.from_agent_id
        query = payload.get("query", "")
        profile_user_id = payload.get("from_user_id", from_agent)

        profile = db_load_profile(profile_user_id)
        if not profile:
            error_text = f"No profile found for user_id '{profile_user_id}'"
            hub.send_message(
                from_agent_id=worker_id,
                to_agent_id=from_agent,
                payload={
                    "type": "recommendation_error",
                    "reason": error_text,
                    "original_message_id": m.message_id,
                },
            )
            processed.append(
                {
                    "request_message_id": m.message_id,
                    "status": "error",
                    "error": error_text,
                }
            )
            continue

        result = freigent.generate_recommendations_json(
            user_profile=profile,
            query=query,
        )

        reply_payload = {
            "type": "recommendation_response",
            "original_message_id": m.message_id,
            "query": query,
            "profile_user_id": profile_user_id,
            "result": result,
        }

        hub.send_message(
            from_agent_id=worker_id,
            to_agent_id=from_agent,
            payload=reply_payload,
        )

        processed.append(
            {
                "request_message_id": m.message_id,
                "status": "ok",
                "sent_to": from_agent,
            }
        )

    return processed


# ----------------------------------------------------------------------
# Health
# ----------------------------------------------------------------------
@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# ----------------------------------------------------------------------
# Freigent: profile & basic single-agent search (DB-backed)
# ----------------------------------------------------------------------
@app.post(
    "/freigent/{user_id}/profile",
    summary="Set or update the user profile for this Freigent (stored in SQLite)",
)
def freigent_set_profile(user_id: str, profile: UserProfileModel) -> Dict[str, Any]:
    """
    Stores the user's profile in SQLite DB. This profile will be used
    later when calling /freigent/{user_id}/search or /auto_search.

    In addition, we auto-register this user as an agent in NandaHub
    and in the agents table.
    """
    profile_dict = {
        "name": profile.name,
        "personality": profile.personality,
        "values": profile.values,
        "experiences": [
            {
                "name": exp.name,
                "notes": exp.notes,
                "rating": exp.rating,
            }
            for exp in profile.experiences
        ],
    }

    # Save to DB
    db_upsert_profile(user_id, profile_dict)

    # Ensure LLM client exists
    get_or_create_freigent(user_id)

    # Register agent in DB and NandaHub
    db_upsert_agent(
        agent_id=user_id,
        agent_type="freigent",
        display_name=profile.name,
        personality_summary=profile.personality,
    )
    hub.register_agent(
        agent_id=user_id,
        agent_type="freigent",
        display_name=profile.name,
        personality_summary=profile.personality,
    )

    return {"status": "ok", "user_id": user_id}


@app.post(
    "/freigent/{user_id}/search",
    response_model=SearchResponse,
    summary="Get product recommendations using the stored profile + query (single agent, DB-backed)",
)
def freigent_search(user_id: str, req: SearchRequest) -> Dict[str, Any]:
    """
    Uses the stored user profile from DB + the search query to ask the LLM
    (via RealJsonFreigent) for structured JSON recommendations.
    """
    profile = db_load_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No profile stored for user_id '{user_id}'. "
                f"Call POST /freigent/{user_id}/profile first."
            ),
        )

    freigent = get_or_create_freigent(user_id)
    result = freigent.generate_recommendations_json(
        user_profile=profile,
        query=req.query,
    )
    return result


# ----------------------------------------------------------------------
# NandaHub: agent registration & basic A2A messaging over HTTP
# ----------------------------------------------------------------------
@app.post(
    "/nanda/register_agent",
    response_model=AgentRegisterResponse,
    summary="Register an agent in NandaHub (does NOT affect DB)",
)
def nanda_register_agent(req: AgentRegisterRequest) -> Dict[str, Any]:
    reg = hub.register_agent(
        agent_id=req.agent_id,
        agent_type=req.agent_type,
        display_name=req.display_name,
        personality_summary=req.personality_summary,
    )
    return {
        "agent_id": reg.agent_id,
        "agent_type": reg.agent_type,
        "display_name": reg.display_name,
        "personality_summary": reg.personality_summary,
    }


@app.get(
    "/nanda/agents",
    summary="List all registered agents in NandaHub (in-memory)",
)
def nanda_list_agents() -> List[Dict[str, Any]]:
    agents = hub.list_agents()
    return [
        {
            "agent_id": a.agent_id,
            "agent_type": a.agent_type,
            "display_name": a.display_name,
            "personality_summary": a.personality_summary,
        }
        for a in agents
    ]


@app.post(
    "/nanda/a2a/send",
    response_model=A2AMessageModel,
    summary="Send an A2A message via NandaHub",
)
def nanda_a2a_send(req: A2ASendRequest) -> Dict[str, Any]:
    msg = hub.send_message(
        from_agent_id=req.from_agent_id,
        to_agent_id=req.to_agent_id,
        payload=req.payload,
    )
    return {
        "message_id": msg.message_id,
        "from_agent_id": msg.from_agent_id,
        "to_agent_id": msg.to_agent_id,
        "payload": msg.payload,
    }


@app.get(
    "/nanda/a2a/inbox/{agent_id}",
    response_model=List[A2AMessageModel],
    summary="Get and clear the inbox of an agent",
)
def nanda_a2a_inbox(agent_id: str) -> List[Dict[str, Any]]:
    msgs = hub.get_inbox(agent_id, clear=True)
    return [
        {
            "message_id": m.message_id,
            "from_agent_id": m.from_agent_id,
            "to_agent_id": m.to_agent_id,
            "payload": m.payload,
        }
        for m in msgs
    ]


# ----------------------------------------------------------------------
# AUTO multi-agent endpoint (DB-backed profiles)
# ----------------------------------------------------------------------
@app.post(
    "/freigent/{user_id}/auto_search",
    response_model=AutoSearchResponse,
    summary=(
        "Automatic multi-agent search: base Freigent + helper Freigents via NandaHub A2A (profiles in SQLite)"
    ),
)
def freigent_auto_search(user_id: str, req: SearchRequest) -> AutoSearchResponse:
    """
    End-to-end flow:

    1. Use RealJsonFreigent for the base user_id to get a first recommendation
       using the profile from DB.
    2. Find helper Freigent agents from DB (other users with profiles).
    3. For each helper:
       - Send them an A2A recommendation_request.
       - Process their inbox using process_recommendation_requests_for_agent.
    4. Read the responses from the origin agent's inbox.
    5. Merge products and return a combined JSON response.
    """
    profile = db_load_profile(user_id)
    if not profile:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No profile stored for user_id '{user_id}'. "
                f"Call POST /freigent/{user_id}/profile first."
            ),
        )

    base_freigent = get_or_create_freigent(user_id)

    # 1) Base recommendation
    base_result = base_freigent.generate_recommendations_json(
        user_profile=profile,
        query=req.query,
    )

    # 2) Helper agents from DB
    helper_ids: List[str] = db_list_helper_agent_ids(user_id)

    # Also ensure they are registered in NandaHub (for inboxes)
    for hid in helper_ids:
        prof = db_load_profile(hid)
        if prof:
            hub.register_agent(
                agent_id=hid,
                agent_type="freigent",
                display_name=prof["name"],
                personality_summary=prof["personality"],
            )

    # 3) Send A2A recommendation_request messages to helpers
    for helper_id in helper_ids:
        hub.send_message(
            from_agent_id=user_id,
            to_agent_id=helper_id,
            payload={
                "type": "recommendation_request",
                "from_user_id": user_id,
                "query": req.query,
            },
        )

    # 4) Simulate worker processing for each helper
    for helper_id in helper_ids:
        process_recommendation_requests_for_agent(helper_id, max_messages=10)

    # 5) Read responses from the origin agent's inbox
    incoming = hub.get_inbox(user_id, clear=True)
    helper_results: List[HelperResult] = []
    merged_products: List[Dict[str, Any]] = []

    # Start with base products
    base_products = base_result.get("products", [])
    if isinstance(base_products, list):
        merged_products.extend(base_products)

    for msg in incoming:
        payload = msg.payload or {}
        p_type = payload.get("type")
        if p_type != "recommendation_response":
            continue

        from_agent = msg.from_agent_id
        result = payload.get("result", {})

        helper_results.append(
            HelperResult(
                agent_id=from_agent,
                result=result,
            )
        )

        products = result.get("products", [])
        if isinstance(products, list):
            merged_products.extend(products)

    helper_count = len(helper_results)
    merged_summary = (
        f"This response combines the base Freigent '{user_id}' recommendations "
        f"with {helper_count} helper Freigent(s): {', '.join(helper_ids) if helper_ids else 'none'}."
    )

    return AutoSearchResponse(
        base_agent_id=user_id,
        helper_agent_ids=helper_ids,
        base_result=base_result,
        helper_results=helper_results,
        merged_products=merged_products,
        merged_summary_for_user=merged_summary,
    )
