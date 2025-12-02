import os
import sqlite3
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from freigent_real_json import RealJsonFreigent


DB_PATH = os.getenv("FREIGENT_DB_PATH", "freigent.db")

app = FastAPI(title="Friegent HTTP API Multi-Agent")


# -------------------------------------------------------------------
# DB helpers
# -------------------------------------------------------------------


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cur = conn.cursor()

    # Profiles table (one row per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            personality TEXT,
            values_text TEXT
        )
        """
    )

    # Experiences table (many rows per user)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            name TEXT,
            notes TEXT,
            rating INTEGER,
            FOREIGN KEY (user_id) REFERENCES profiles (user_id)
        )
        """
    )

    conn.commit()
    conn.close()


def db_upsert_profile(user_id: str, profile: "UserProfile") -> None:
    conn = get_db_connection()
    cur = conn.cursor()

    # upsert profile row
    cur.execute(
        """
        INSERT INTO profiles (user_id, name, personality, values_text)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name=excluded.name,
            personality=excluded.personality,
            values_text=excluded.values_text
        """,
        (user_id, profile.name, profile.personality, profile.values),
    )

    # delete old experiences
    cur.execute("DELETE FROM experiences WHERE user_id = ?", (user_id,))

    # insert new experiences
    for exp in profile.experiences:
        cur.execute(
            """
            INSERT INTO experiences (user_id, name, notes, rating)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, exp.name, exp.notes, exp.rating),
        )

    conn.commit()
    conn.close()


def db_get_profile(user_id: str) -> Optional["UserProfile"]:
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT user_id, name, personality, values_text FROM profiles WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None

    cur.execute(
        "SELECT name, notes, rating FROM experiences WHERE user_id = ? ORDER BY id ASC",
        (user_id,),
    )
    exp_rows = cur.fetchall()
    conn.close()

    experiences: List[ProductExperience] = []
    for er in exp_rows:
        experiences.append(
            ProductExperience(
                name=er["name"],
                notes=er["notes"],
                rating=er["rating"],
            )
        )

    return UserProfile(
        name=row["name"] or "",
        personality=row["personality"] or "",
        values=row["values_text"] or "",
        experiences=experiences,
    )


def db_get_other_profiles(
    user_id: str, limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Return up to `limit` other users' profiles + experiences.
    Used as 'friends' for multi-agent recommendations.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT user_id, name, personality, values_text
        FROM profiles
        WHERE user_id != ?
        ORDER BY user_id ASC
        LIMIT ?
        """,
        (user_id, limit),
    )
    rows = cur.fetchall()

    results: List[Dict[str, Any]] = []
    for row in rows:
        other_id = row["user_id"]
        cur.execute(
            """
            SELECT name, notes, rating
            FROM experiences
            WHERE user_id = ?
            ORDER BY id ASC
            """,
            (other_id,),
        )
        exp_rows = cur.fetchall()
        exps: List[Dict[str, Any]] = []
        for er in exp_rows:
            exps.append(
                {
                    "name": er["name"],
                    "notes": er["notes"],
                    "rating": er["rating"],
                }
            )

        results.append(
            {
                "user_id": other_id,
                "profile": {
                    "name": row["name"] or "",
                    "personality": row["personality"] or "",
                    "values": row["values_text"] or "",
                    "experiences": exps,
                },
            }
        )

    conn.close()
    return results


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------


class ProductExperience(BaseModel):
    name: str
    notes: str
    rating: int = Field(ge=1, le=5)


class UserProfile(BaseModel):
    name: str
    personality: str
    values: str
    experiences: List[ProductExperience] = Field(default_factory=list)


class ProfileUpsertRequest(BaseModel):
    user_id: str
    profile: UserProfile


class RecommendMultiRequest(BaseModel):
    user_id: str
    query: str
    num_friends: int = 3


class ProductRecommendation(BaseModel):
    name: str
    short_description: str
    why_match: str
    estimated_price_range: str
    source_user_id: str
    source_kind: str  # "self" or "friend"


class RecommendMultiResponse(BaseModel):
    products: List[ProductRecommendation]
    summary_for_user: str
    sources: Dict[str, Any]


# -------------------------------------------------------------------
# Global Freigent core
# -------------------------------------------------------------------

# This is the main Freigent used for the requesting user.
CORE_FREIGENT = RealJsonFreigent(agent_id="freigent-core")


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    init_db()


# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "friegent-http-api-multi"}


@app.post("/api/profile")
def upsert_profile(req: ProfileUpsertRequest) -> Dict[str, Any]:
    """
    Create or update a user's profile in the DB.

    This is what ChatGPT should call (via a tool) to persist or update
    the user's profile based on the free-text description they gave.
    """
    db_upsert_profile(req.user_id, req.profile)
    return {
        "status": "ok",
        "user_id": req.user_id,
        "num_experiences": len(req.profile.experiences),
    }


@app.get("/api/profile/{user_id}")
def get_profile(user_id: str) -> Dict[str, Any]:
    prof = db_get_profile(user_id)
    if not prof:
        raise HTTPException(status_code=404, detail=f"No profile found for user_id={user_id}")
    return {
        "user_id": user_id,
        "profile": prof.dict(),
    }


@app.post("/api/recommend_multi", response_model=RecommendMultiResponse)
def recommend_multi(req: RecommendMultiRequest) -> RecommendMultiResponse:
    """
    Main multi-agent recommendation endpoint.

    Flow:
    1. Load main user's profile from DB.
    2. Ask CORE_FREIGENT for recommendations for that user.
    3. Look up other profiles in DB (other users) as 'friend Freigents'.
    4. For each friend, call a separate RealJsonFreigent instance with that profile.
    5. Merge everything and return products + combined summary + metadata.
    """

    # 1. Get main profile
    main_profile = db_get_profile(req.user_id)
    if not main_profile:
        raise HTTPException(
            status_code=400,
            detail=f"No profile found for user_id={req.user_id}. Please set profile first via /api/profile.",
        )

    # 2. Main recommendations
    main_result = CORE_FREIGENT.generate_recommendations_json(
        user_profile=main_profile.dict(), query=req.query
    )

    main_products_raw = main_result.get("products", []) or []
    main_summary = main_result.get("summary_for_user", "")

    products: List[ProductRecommendation] = []

    # Normalize main products
    for p in main_products_raw:
        try:
            products.append(
                ProductRecommendation(
                    name=p.get("name", ""),
                    short_description=p.get("short_description", ""),
                    why_match=p.get("why_match", ""),
                    estimated_price_range=p.get("estimated_price_range", ""),
                    source_user_id=req.user_id,
                    source_kind="self",
                )
            )
        except Exception:
            # Skip bad product entries
            continue

    # 3. Friend profiles
    friend_rows = db_get_other_profiles(req.user_id, limit=max(0, req.num_friends))
    friend_ids: List[str] = []

    for row in friend_rows:
        friend_id = row["user_id"]
        friend_ids.append(friend_id)
        friend_profile_dict = row["profile"]

        # Create a dedicated Freigent for this friend
        friend_freigent = RealJsonFreigent(agent_id=f"freigent-{friend_id}")

        try:
            friend_result = friend_freigent.generate_recommendations_json(
                user_profile=friend_profile_dict, query=req.query
            )
        except Exception:
            # If the LLM fails for a friend, just skip that friend
            continue

        friend_products_raw = friend_result.get("products", []) or []

        for p in friend_products_raw:
            try:
                products.append(
                    ProductRecommendation(
                        name=p.get("name", ""),
                        short_description=p.get("short_description", ""),
                        why_match=p.get("why_match", ""),
                        estimated_price_range=p.get("estimated_price_range", ""),
                        source_user_id=friend_id,
                        source_kind="friend",
                    )
                )
            except Exception:
                continue

    # 4. Build combined summary
    if friend_ids:
        friend_line = (
            f"\n\nI also asked {len(friend_ids)} other Friegent agents "
            f"({', '.join(friend_ids)}) that have their own profiles in your network, "
            "and combined their recommendations with yours."
        )
    else:
        friend_line = (
            "\n\nRight now I only used your own profile. Once more Friegent agents "
            "are created and stored in the system, I'll be able to consult them as well."
        )

    combined_summary = main_summary + friend_line

    sources_meta: Dict[str, Any] = {
        "main_user_id": req.user_id,
        "friend_user_ids": friend_ids,
        "num_friend_agents": len(friend_ids),
    }

    return RecommendMultiResponse(
        products=products,
        summary_for_user=combined_summary,
        sources=sources_meta,
    )
