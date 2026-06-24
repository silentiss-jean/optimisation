import json
from typing import Dict, Any, Optional, Callable

from providers.llm_provider_interface import LLMProvider
from tools.tool_dispatcher import ToolDispatcher


SYSTEM_PROMPT_TEMPLATE = """Tu es un agent IA autonome. Tu dois accomplir la tâche demandée en utilisant les outils disponibles.

{tools_description}

Mode de sécurité actif : {security_mode}{scope_info}

Règles strictes :
1. Réponds UNIQUEMENT en JSON valide, sans texte autour.
2. Si tu dois utiliser un outil, réponds avec :
   {{"action": "tool", "tool": "<nom_outil>", "params": {{...}}}}
3. Si tu as la réponse finale, réponds avec :
   {{"action": "final_answer", "answer": "<ta réponse complète>"}}
4. En mode MONITORING, tu ne peux PAS utiliser d'outils. Donne directement une final_answer.

Historique de la conversation :
{history}

Tâche : {task}"""


# Types d'événement envoyés vers l'UI
EVT_THINKING   = "thinking"    # chunk de pensée du LLM en temps réel
EVT_ACTION     = "action"      # appel d'outil (tool_name + params)
EVT_OBSERVE    = "observe"     # résultat de l'outil
EVT_ANSWER     = "answer"      # réponse finale
EVT_STEP       = "step"        # début d'étape (numéro)
EVT_WARNING    = "warning"     # erreur non fatale
EVT_SECURITY   = "security"    # action bloquée


class SecurityMode:
    MONITORING    = "MONITORING"
    LIMITED_SCOPE = "LIMITED_SCOPE"
    FULL_CONTROL  = "FULL_CONTROL"


class AgentOrchestrator:
    """
    Orchestrateur ReAct : Pensée → Action → Observation, jusqu'à réponse finale.
    Supporte un callback on_event(event_type: str, data: str) pour le streaming temps réel.
    """
    def __init__(self, llm_provider: LLMProvider, dispatcher: ToolDispatcher,
                 on_event: Optional[Callable[[str, str], None]] = None):
        self.llm_provider = llm_provider
        self.dispatcher   = dispatcher
        self.on_event     = on_event or (lambda evt, data: print(f"[{evt}] {data}", end="", flush=True))
        self.security_mode: str      = SecurityMode.MONITORING
        self.current_scope: Optional[str] = None
        self.chat_history: list      = []

    def _emit(self, event_type: str, data: str):
        """Envoie un événement vers le callback UI (non bloquant)."""
        try:
            self.on_event(event_type, data)
        except Exception:
            pass

    def set_safety_mode(self, mode: str, scope: Optional[str] = None):
        if mode not in [SecurityMode.MONITORING, SecurityMode.LIMITED_SCOPE, SecurityMode.FULL_CONTROL]:
            raise ValueError(f"Mode de sécurité invalide : {mode}")
        self.security_mode = mode
        self.current_scope = scope

    def _check_security(self, tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        if self.security_mode == SecurityMode.MONITORING:
            return False, "🛑 Action bloquée : mode MONITORING actif."
        if self.security_mode == SecurityMode.LIMITED_SCOPE and self.current_scope:
            path = params.get('file_path') or params.get('directory', '')
            if path and not str(path).startswith(self.current_scope):
                return False, f"🛑 Hors scope '{self.current_scope}' : {path}"
        return True, ""

    def _build_system_prompt(self, task: str) -> str:
        history_text = ""
        for msg in self.chat_history:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                history_text += f"Agent: {content}\n"
            elif role == "tool":
                history_text += f"Observation: {content}\n"
        scope_info = f" | Scope : {self.current_scope}" if self.current_scope else ""
        return SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=self.dispatcher.get_tools_description(),
            security_mode=self.security_mode,
            scope_info=scope_info,
            history=history_text or "(aucun historique)",
            task=task
        )

    def _parse_llm_response(self, raw: str) -> Optional[Dict]:
        raw   = raw.strip()
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    def run_agentic_cycle(self, initial_prompt: str, max_steps: int = 8) -> Optional[str]:
        """
        Boucle ReAct avec streaming temps réel via on_event callback.
        Chaque chunk LLM est émis immédiatement vers l'UI.
        """
        task = initial_prompt

        for step in range(max_steps):
            # --- Signal début d'étape ---
            self._emit(EVT_STEP, f"Étape {step + 1}/{max_steps}")

            # --- Construction du prompt ---
            system_prompt = self._build_system_prompt(task)

            # --- Streaming LLM chunk par chunk vers l'UI ---
            self._emit(EVT_THINKING, "")  # signal début pensée
            raw_chunks = []
            for chunk in self.llm_provider.stream_response(system_prompt, []):
                raw_chunks.append(chunk)
                self._emit(EVT_THINKING, chunk)  # ← chaque token arrivé

            raw_response = "".join(raw_chunks).strip()
            self.chat_history.append({"role": "assistant", "content": raw_response})

            # --- Parsing JSON ---
            parsed = self._parse_llm_response(raw_response)

            if not parsed:
                msg = "[FORMAT] Réponse non-JSON reçue, nouvelle tentative..."
                self._emit(EVT_WARNING, msg)
                self.chat_history.append({"role": "tool", "content":
                    "[FORMAT ERROR] Réponds UNIQUEMENT en JSON valide."})
                continue

            action = parsed.get("action")

            # --- Réponse finale ---
            if action == "final_answer":
                answer = parsed.get("answer", "")
                self._emit(EVT_ANSWER, answer)
                return answer

            # --- Appel d'outil ---
            elif action == "tool":
                tool_name = parsed.get("tool", "")
                params    = parsed.get("params", {})
                self._emit(EVT_ACTION, f"{tool_name} | {json.dumps(params, ensure_ascii=False)}")

                allowed, reason = self._check_security(tool_name, params)
                if not allowed:
                    self._emit(EVT_SECURITY, reason)
                    observation = reason
                else:
                    observation = self.dispatcher.dispatch_call(tool_name=tool_name, **params)

                obs_preview = str(observation)[:500]
                self._emit(EVT_OBSERVE, obs_preview)
                self.chat_history.append({"role": "tool", "content": str(observation)})

            else:
                msg = f"[ERREUR] Action '{action}' inconnue."
                self._emit(EVT_WARNING, msg)
                self.chat_history.append({"role": "tool", "content":
                    f"Action '{action}' inconnue. Utilise 'tool' ou 'final_answer'."})

        self._emit(EVT_WARNING, "[TIMEOUT] Nombre maximum d'étapes atteint.")
        return "[TIMEOUT] L'agent n'a pas pu terminer dans le nombre d'étapes autorisé."
