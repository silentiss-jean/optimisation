"""
tools/memory_store.py
─────────────────────
Mémoire persistante de l'agent — P3-002.

Deux niveaux :
  • session_history  : échanges user↔agent de la session courante
                       (rechargés au démarrage, remplacés à chaque session)
  • long_term        : faits importants mémorisés manuellement ou extraits
                       (persist entre toutes les sessions)

Fichier stocké dans ~/.optimisation_agent/memory.json
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_DIR  = Path.home() / ".optimisation_agent"
_DEFAULT_FILE = _DEFAULT_DIR / "memory.json"

_EMPTY_STORE: dict[str, Any] = {
    "version": 1,
    "last_saved": None,
    "session_history": [],
    "long_term": [],
}


class MemoryStore:
    """Charge, met à jour et sauvegarde la mémoire persistante de l'agent."""

    def __init__(self, path: Path | None = None):
        self._path = Path(path) if path else _DEFAULT_FILE
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = self._load()

    # ──────────────────────────────────────────────── I/O
    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                # migration : assurer que toutes les clés existent
                for k, v in _EMPTY_STORE.items():
                    data.setdefault(k, v)
                return data
            except Exception:
                pass
        import copy
        return copy.deepcopy(_EMPTY_STORE)

    def save(self):
        """Écrit le store sur disque."""
        self._data["last_saved"] = datetime.now().isoformat(timespec="seconds")
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    # ──────────────────────────────────────────────── session_history
    @property
    def session_history(self) -> list[dict]:
        return self._data["session_history"]

    def set_session_history(self, history: list[dict]):
        """Remplace l'historique de session par la liste fournie puis sauvegarde."""
        self._data["session_history"] = list(history)
        self.save()

    def clear_session_history(self):
        self._data["session_history"] = []
        self.save()

    # ──────────────────────────────────────────────── long_term
    @property
    def long_term(self) -> list[str]:
        return self._data["long_term"]

    def add_long_term(self, fact: str):
        """Ajoute un fait à la mémoire longue durée (dédupliqué) et sauvegarde."""
        if fact and fact not in self._data["long_term"]:
            self._data["long_term"].append(fact)
            self.save()

    def remove_long_term(self, fact: str):
        """Supprime un fait de la mémoire longue durée."""
        try:
            self._data["long_term"].remove(fact)
            self.save()
        except ValueError:
            pass

    def clear_long_term(self):
        self._data["long_term"] = []
        self.save()

    # ──────────────────────────────────────────────── helpers
    def clear_all(self):
        """Efface toute la mémoire (session + longue durée)."""
        self._data["session_history"] = []
        self._data["long_term"] = []
        self.save()

    def summary_for_prompt(self) -> str:
        """
        Retourne un bloc texte à injecter dans le system prompt :
        - faits longue durée (si présents)
        - derniers échanges de session (5 max pour ne pas saturer le contexte)
        """
        parts: list[str] = []

        if self._data["long_term"]:
            facts = "\n".join(f"  • {f}" for f in self._data["long_term"])
            parts.append(f"Mémoire longue durée :\n{facts}")

        history = self._data["session_history"]
        if history:
            # on garde les 10 derniers messages (5 tours user/agent)
            recent = history[-10:]
            lines = []
            for msg in recent:
                role    = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"Utilisateur: {content}")
                elif role == "assistant":
                    lines.append(f"Agent: {content}")
                # on n'expose pas les observations brutes dans le résumé inter-requêtes
            if lines:
                parts.append("Échanges précédents (session) :\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else ""

    @property
    def path(self) -> Path:
        return self._path
