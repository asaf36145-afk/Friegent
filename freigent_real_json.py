import os
import json
from typing import Dict, Any, List
from dataclasses import dataclass

from anthropic import Anthropic

@dataclass
class ProductExperience:
    name: str
    notes: str
    rating: int

@dataclass
class UserProfile:
    name: str
    personality: str
    values: str
    experiences: List[ProductExperience] # כדאי לוודא ש-List מיובא (כפי שמופיע בראש הקובץ)

class RealJsonFreigent:
    """
    A 'real' Freigent that talks to Anthropic and returns
    structured JSON with product recommendations.

    This class does NOT know anything about HTTP or FastAPI.
    It just gets a user profile + query, and returns a Python dict.
    """

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in environment/.env")

        # Use the same model config that already works for you in main.py
        self.model_name = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
        self.client = Anthropic(api_key=api_key)

    # ------------------------------------------------------------------
    # Helper: turn profile dict into a readable text block
    # ------------------------------------------------------------------
    def _profile_to_text(self, profile: Dict[str, Any]) -> str:
        name = profile.get("name", "Unknown user")
        personality = profile.get("personality", "")
        values = profile.get("values", "")
        experiences: List[Dict[str, Any]] = profile.get("experiences", [])

        exp_lines = []
        for e in experiences:
            exp_name = e.get("name", "Unknown product")
            notes = e.get("notes", "")
            rating = e.get("rating", None)
            if rating is not None:
                exp_lines.append(f"- {exp_name} (rating {rating}/5): {notes}")
            else:
                exp_lines.append(f"- {exp_name}: {notes}")

        if not exp_lines:
            exp_text = "No concrete past product experience."
        else:
            exp_text = "\n".join(exp_lines)

        return (
            f"User name: {name}\n"
            f"Personality: {personality}\n"
            f"Values in products: {values}\n"
            f"Past product experience:\n{exp_text}\n"
        )

    # ------------------------------------------------------------------
    # Main method: generate JSON recommendations
    # ------------------------------------------------------------------
    def generate_recommendations_json(
        self,
        user_profile: Dict[str, Any],
        query: str,
    ) -> Dict[str, Any]:
        """
        Calls Anthropic and asks it to return ONLY valid JSON with
        a list of products and a summary_for_user.

        Returns a Python dict:
        {
          "products": [...],
          "summary_for_user": "..."
        }
        """
        profile_text = self._profile_to_text(user_profile)

        system_prompt = (
            "You are a product recommendation engine for an AI shopping friend "
            "(Freigent). You receive:\n"
            "1) A detailed user profile (personality, values, previous products).\n"
            "2) A free-text product search query.\n\n"
            "Your job is to suggest 3-5 concrete product ideas that match the user.\n"
            "IMPORTANT:\n"
            "- You MUST respond with ONLY valid JSON.\n"
            "- Do NOT include any markdown, backticks, or plain text outside JSON.\n"
            "- The JSON must have this exact structure:\n"
            "{\n"
            '  \"products\": [\n'
            "    {\n"
            '      \"name\": \"string\",\n'
            '      \"short_description\": \"string\",\n'
            '      \"why_match\": \"string\",\n'
            '      \"estimated_price_range\": \"string\"\n'
            "    },\n"
            "    ... 3 to 5 items ...\n"
            "  ],\n"
            '  \"summary_for_user\": \"short, friendly paragraph explaining the recommendations\"\n'
            "}\n"
            "- The JSON must be parseable by json.loads in Python.\n"
        )

        user_prompt = (
            "Here is the user profile:\n"
            "----------------------------------------\n"
            f"{profile_text}\n"
            "----------------------------------------\n\n"
            "Here is the user's product search query:\n"
            f"{query}\n\n"
            "Now generate the JSON response as specified. Remember: JSON only."
        )

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
            )

            # Anthropic returns a list of content blocks; we take the first text
            raw_text = ""
            if response.content and len(response.content) > 0:
                # each item is usually {'type': 'text', 'text': '...'}
                raw_text = response.content[0].text.strip()

            # Try to parse JSON
            parsed = json.loads(raw_text)

            # Ensure the expected keys exist
            if "products" not in parsed:
                parsed["products"] = []
            if "summary_for_user" not in parsed:
                parsed["summary_for_user"] = (
                    "No summary_for_user provided by the model."
                )

            return parsed

        except Exception as e:
            # If anything fails (LLM error, JSON error, etc.), return a safe fallback
            return {
                "products": [],
                "summary_for_user": (
                    "The agent tried to return a result, but there was an error "
                    "parsing the JSON output.\n\nError: " + str(e)
                ),
            }
