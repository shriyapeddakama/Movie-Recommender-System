"""Configuration schemas shared by serving components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Router configuration


@dataclass
class Condition:
    field: str
    operator: str = "eq"
    value: Any = None

    def evaluate(self, resolver: callable) -> bool:
        lhs = resolver(self.field)
        op = self.operator.lower()
        if op in {"eq", "=="}:
            return lhs == self.value
        if op in {"ne", "!="}:
            return lhs != self.value
        if op in {"lt", "<"}:
            return lhs is not None and lhs < self.value
        if op in {"lte", "<="}:
            return lhs is not None and lhs <= self.value
        if op in {"gt", ">"}:
            return lhs is not None and lhs > self.value
        if op in {"gte", ">="}:
            return lhs is not None and lhs >= self.value
        if op == "in":
            return lhs in self._ensure_iterable(self.value)
        if op in {"not_in", "notin"}:
            return lhs not in self._ensure_iterable(self.value)
        if op == "exists":
            return lhs is not None
        if op == "missing":
            return lhs is None
        raise ValueError(f"Unsupported operator: {self.operator}")

    @staticmethod
    def _ensure_iterable(value: Any) -> Iterable[Any]:
        if isinstance(value, (list, tuple, set)):
            return value
        return [value]


@dataclass
class RouterRule:
    name: str
    model: str
    conditions: List[Condition] = field(default_factory=list)
    match: str = "all"  # all | any
    priority: int = 0

    def matches(self, resolver: callable) -> bool:
        results = [cond.evaluate(resolver) for cond in self.conditions]
        if not results:
            return True
        if self.match == "any":
            return any(results)
        return all(results)


@dataclass
class RouterConfig:
    default_model: str
    rules: List[RouterRule] = field(default_factory=list)
    fallback_model: Optional[str] = None

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "RouterConfig":
        if "default_model" not in payload:
            raise ValueError("Router config requires 'default_model'")
        rules = []
        for entry in payload.get("rules", []):
            conditions = [
                Condition(
                    field=cond["field"],
                    operator=cond.get("operator", "eq"),
                    value=cond.get("value"),
                )
                for cond in entry.get("conditions", [])
            ]
            rules.append(
                RouterRule(
                    name=entry.get("name", entry.get("model", "rule")),
                    model=entry["model"],
                    conditions=conditions,
                    match=entry.get("match", "all"),
                    priority=entry.get("priority", 0),
                )
            )
        rules.sort(key=lambda rule: rule.priority, reverse=True)
        return RouterConfig(
            default_model=payload["default_model"],
            fallback_model=payload.get("fallback_model"),
            rules=rules,
        )


# ---------------------------------------------------------------------------
# Model manifest (per model directory)


@dataclass
class ModelManifest:
    name: str
    version: str
    class_path: str
    runtime: Dict[str, Any] = field(default_factory=dict)
    artefacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "ModelManifest":
        required = {"name", "version", "class_path"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"Model config missing fields: {sorted(missing)}")
        return ModelManifest(
            name=payload["name"],
            version=str(payload["version"]),
            class_path=payload["class_path"],
            runtime=payload.get("runtime", {}),
            artefacts=payload.get("artefacts", {}),
            metadata=payload.get("metadata", {}),
        )
