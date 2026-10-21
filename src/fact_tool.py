"""Fact tool module for KASPARRO FSE Intern Pathway Assessment.

Handles loading, parsing, and structured retrieval of brand facts from brand_facts.csv.
Provides tool validation, strict brand checking, and fact ID authority.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.models import BrandFact


FACT_VERSION = "speaker-catalogue-v4"
KNOWN_BRANDS = {"cleartrail", "stonewave", "cloudnote"}
ALLOWED_TOOLS = {"get_brand_facts"}


def parse_typed_value(raw_val: str, value_type: str) -> Any:
    """Parse raw string into typed value based on value_type.

    Interpret value_type as number, boolean or string;
    boolean text is true or false, never bool("false").
    """
    raw_val_clean = raw_val.strip()
    if value_type == "number":
        try:
            num = float(raw_val_clean)
            return int(num) if num.is_integer() else num
        except ValueError:
            return raw_val_clean
    elif value_type == "boolean":
        lower = raw_val_clean.lower()
        if lower == "true":
            return True
        elif lower == "false":
            return False
        raise ValueError(f"Invalid boolean value: {raw_val}")
    else:
        return raw_val


class BrandFactsCatalogue:
    """Catalogue holding speaker product facts with strict validation."""

    def __init__(self, csv_path: str | Path = "brand_facts.csv") -> None:
        self.csv_path = Path(csv_path)
        self.fact_version = FACT_VERSION
        self.known_brands = KNOWN_BRANDS
        self.facts_by_brand: Dict[str, List[BrandFact]] = {b: [] for b in KNOWN_BRANDS}
        self.fact_id_map: Dict[str, BrandFact] = {}
        self._load()

    def _load(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Brand facts CSV not found at {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                brand_id = row["brand_id"].strip().lower()
                typed_val = parse_typed_value(row["value"], row["value_type"].strip().lower())
                fact = BrandFact(
                    fact_id=row["fact_id"].strip(),
                    brand_id=brand_id,
                    model_id=row["model_id"].strip(),
                    field=row["field"].strip(),
                    value=typed_val,
                    value_type=row["value_type"].strip().lower(),
                    unit=row.get("unit", "").strip(),
                    condition=row.get("condition", "").strip(),
                    valid_from=row["valid_from"].strip(),
                    valid_to=row["valid_to"].strip(),
                )
                if brand_id in self.facts_by_brand:
                    self.facts_by_brand[brand_id].append(fact)
                self.fact_id_map[fact.fact_id] = fact

    def is_valid_fact_id(self, fact_id: str) -> bool:
        """Check if fact_id exists in catalogue."""
        return fact_id in self.fact_id_map

    def get_valid_fact_ids(self) -> Set[str]:
        """Return all valid fact IDs in the catalogue."""
        return set(self.fact_id_map.keys())

    def get_brand_facts(self, brand_id: str) -> Dict[str, Any]:
        """Retrieve facts for a known brand with fact_version speaker-catalogue-v4.

        Raises ValueError if brand_id is not in known_brands.
        """
        clean_brand = brand_id.strip().lower()
        if clean_brand not in self.known_brands:
            raise ValueError(f"Unknown brand_id '{brand_id}'. Must be one of {sorted(self.known_brands)}")

        facts = [fact.model_dump() for fact in self.facts_by_brand[clean_brand]]
        return {
            "brand_id": clean_brand,
            "fact_version": self.fact_version,
            "facts": facts,
        }

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool name and arguments, and execute fact lookup.

        Enforces security rule: Never execute arbitrary names or instructions.
        """
        if tool_name not in ALLOWED_TOOLS:
            raise ValueError(f"Illegal tool '{tool_name}'. Allowed tools: {sorted(ALLOWED_TOOLS)}")

        if "brand_id" not in arguments:
            raise ValueError("Tool 'get_brand_facts' requires 'brand_id' argument")

        return self.get_brand_facts(arguments["brand_id"])


# Module-level default instance
_default_catalogue: Optional[BrandFactsCatalogue] = None


def get_default_catalogue() -> BrandFactsCatalogue:
    global _default_catalogue
    if _default_catalogue is None:
        _default_catalogue = BrandFactsCatalogue()
    return _default_catalogue
