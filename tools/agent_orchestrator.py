import json
import re
from typing import Dict, Any, Optional, Callable

from providers.llm_provider_interface import LLMProvider
from tools.tool_dispatcher import ToolDispatcher


SYSTEM_PROMPT_TEMPLATE = """Tu es un agent IA autonome. Tu dois accomplir la tâche demandée en utilisant les outils disponibles.

{tools_description}

Mode de sécurité actif : {security_mode}{scope_info}

Règles ABSOLUES — à respecter sans exception :
1. Réponds UNIQUEMENT avec UN SEUL objet JSON valide, sans texte autour.
2. UNE SEULE action par réponse. Si la tâche demande 2 actions, fais la première,
   attends l'observation, puis fais la seconde dans la réponse suivante.
3. Pour utiliser un outil :
   {{"action": "tool", "tool": "<nom_outil>", "params": {{...}}}}
4. Pour donner la réponse finale :
   {{"action": "final_answer", "answer": "<ta réponse complète>"}}
5. En mode MONITORING, pas d'outils — final_answer directement.
6. Pour ouvrir une URL dans l'onglet actif : browser_navigate.
   Pour ouvrir un NOUVEL onglet sans fermer la page courante : browser_new_tab.
7. Pour lire le contenu d'une page (repos, texte, données) : browser_navigate PUIS browser_get_text.
   N'utilise JAMAIS open_url pour récupérer du contenu — open_url ne lit pas la page.
8. Si un outil retourne [ERREUR], essaie une URL alternative avant de rendre final_answer d'échec.

Historique de la conversation :
{history}

Tâche : {task}"""


EVT_THINKING = "thinking"
EVT_ACTION   = "action"
EVT_OBSERVE  = "observe"
EVT_ANSWER   = "answer"
EVT_STEP     = "step"
EVT_WARNING  = "warning"
EVT_SECURITY = "security"


class SecurityMode:
    MONITORING    = "MONITORING"
    LIMITED_SCOPE = "LIMITED_SCOPE"
    FULL_CONTROL  = "FULL_CONTROL"


ALLOWED_TOOLS: Dict[str, set] = {
    SecurityMode.MONITORING: set(),
    SecurityMode.LIMITED_SCOPE: {
        "read_file", "write_file", "find_files"
    },
    SecurityMode.FULL_CONTROL: {
        "read_file", "write_file", "find_files",
        "command_line_execute", "open_url", "web_scrape",
        "browser_navigate", "browser_new_tab",
        "browser_click", "browser_fill",
        "browser_screenshot", "browser_get_text",
        "browser_scroll", "browser_wait_for",
    },
}

DONE_TRIGGERS: set = {
    "browser_navigate", "browser_new_tab", "browser_click",
    "browser_fill", "browser_screenshot", "open_url",
}


class AgentOrchestrator:
    def __init__(self, llm_provider: LLMProvider, dispatcher: ToolDispatcher,
                 on_event: Optional[Callable[[str, str], None]] = None):
        self.llm_provider = llm_provider
        self.dispatcher   = dispatcher
        self.on_event     = on_event or (lambda e, d: print(f"[{e}] {d}", end="", flush=True))
        self.security_mode: str           = SecurityMode.MONITORING
        self.current_scope: Optional[str] = None
        self.chat_history: list           = []

    def _emit(self, event_type: str, data: str):
        try:
            self.on_event(event_type, data)
        except Exception:
            pass

    def set_safety_mode(self, mode: str, scope: Optional[str] = None):
        if mode not in [SecurityMode.MONITORING, SecurityMode.LIMITED_SCOPE, SecurityMode.FULL_CONTROL]:
            raise ValueError(f"Mode invalide : {mode}")
        self.security_mode = mode
        self.current_scope = scope

    def _check_security(self, tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        allowed_set = ALLOWED_TOOLS.get(self.security_mode, set())
        if tool_name not in allowed_set:
            if self.security_mode == SecurityMode.MONITORING:
                return False, "🛑 Action bloquée : mode MONITORING actif."
            return False, (
                f"🛑 Outil '{tool_name}' non autorisé en mode {self.security_mode}. "
                f"Outils disponibles : {', '.join(sorted(allowed_set)) or 'aucun'}."
            )
        if self.security_mode == SecurityMode.LIMITED_SCOPE and self.current_scope:
            path = params.get('file_path') or params.get('directory', '')
            if path and not str(path).startswith(self.current_scope):
                return False, f"🛑 Chemin '{path}' hors du scope autorisé '{self.current_scope}'."
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

    def _parse_llm_response(self, raw: str) -> tuple[Optional[Dict], bool]:
        """
        Extrait le PREMIER objet JSON valide de la réponse.
        Retourne (parsed, had_multiple) où had_multiple=True si plusieurs JSON détectés.
        """
        raw = raw.strip()
        # Trouver tous les blocs JSON candidats
        candidates = [m.start() for m in re.finditer(r'\{', raw)]
        had_multiple = False
        first_parsed = None

        for start in candidates:
            # Trouver la fin de ce bloc JSON
            depth, i = 0, start
            in_string = False
            escape_next = False
            while i < len(raw):
                c = raw[i]
                if escape_next:
                    escape_next = False
                elif c == '\\' and in_string:
                    escape_next = True
                elif c == '"':
                    in_string = not in_string
                elif not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            candidate_str = raw[start:i + 1]
                            try:
                                parsed = json.loads(candidate_str)
                                if first_parsed is None:
                                    first_parsed = parsed
                                else:
                                    had_multiple = True
                                    break  # pas besoin de chercher plus
                            except json.JSONDecodeError:
                                pass
                            break
                i += 1
            if had_multiple:
                break

        return first_parsed, had_multiple

    def _make_done_answer(self, tool_name: str, params: Dict[str, Any], observation: str) -> str:
        if tool_name in ("browser_navigate", "browser_new_tab"):
            url = observation.split(":", 2)[-1].strip()
            label = "Nouvel onglet ouvert" if tool_name == "browser_new_tab" else "Navigateur ouvert"
            return f"{label} sur : {url}"
        if tool_name == "browser_click":
            return f"Clic effectué sur '{params.get('selector', '')}'."
        if tool_name == "browser_fill":
            return f"Champ '{params.get('selector', '')}' rempli."
        if tool_name == "browser_screenshot":
            return f"Capture d'écran sauvegardée : {params.get('save_path', 'screenshot.png')}"
        if tool_name == "open_url":
            return f"URL ouverte : {params.get('url', '')}"
        return observation.replace("[DONE]", "").strip()

    def run_agentic_cycle(self, initial_prompt: str, max_steps: int = 8) -> Optional[str]:
        task = initial_prompt
        for step in range(max_steps):
            self._emit(EVT_STEP, f"Étape {step + 1}/{max_steps}")
            system_prompt = self._build_system_prompt(task)

            self._emit(EVT_THINKING, "")
            raw_chunks = []
            for chunk in self.llm_provider.stream_response(system_prompt, []):
                raw_chunks.append(chunk)
                self._emit(EVT_THINKING, chunk)

            raw_response = "".join(raw_chunks).strip()
            self.chat_history.append({"role": "assistant", "content": raw_response})

            parsed, had_multiple = self._parse_llm_response(raw_response)

            if not parsed:
                self._emit(EVT_WARNING, "[FORMAT] Réponse non-JSON, nouvelle tentative...")
                self.chat_history.append({"role": "tool",
                    "content": "[FORMAT ERROR] Réponds UNIQUEMENT avec UN SEUL objet JSON valide, sans texte autour."})
                continue

            if had_multiple:
                # On exécute le premier JSON et on prévient le LLM
                self._emit(EVT_WARNING, "[FORMAT] Plusieurs JSON détectés — seule la première action est exécutée.")
                self.chat_history.append({"role": "tool",
                    "content": "[RAPPEL] UNE SEULE action par réponse. J'ai exécuté la première, attends l'observation avant d'envoyer la suivante."})

            action = parsed.get("action")

            if action == "final_answer":
                answer = parsed.get("answer", "")
                self._emit(EVT_ANSWER, answer)
                return answer

            elif action == "tool":
                tool_name = parsed.get("tool", "")
                params    = parsed.get("params", {})
                self._emit(EVT_ACTION, f"{tool_name} | {json.dumps(params, ensure_ascii=False)}")

                allowed, reason = self._check_security(tool_name, params)
                if not allowed:
                    self._emit(EVT_SECURITY, reason)
                    observation = reason + "\nUtilise 'final_answer' pour expliquer à l'utilisateur."
                else:
                    observation = self.dispatcher.dispatch_call(tool_name=tool_name, **params)

                self._emit(EVT_OBSERVE, str(observation)[:500])
                self.chat_history.append({"role": "tool", "content": str(observation)})

                if "[DONE]" in str(observation) and tool_name in DONE_TRIGGERS:
                    answer = self._make_done_answer(tool_name, params, str(observation))
                    self._emit(EVT_ANSWER, answer)
                    return answer

            else:
                self._emit(EVT_WARNING, f"[ERREUR] Action '{action}' inconnue.")
                self.chat_history.append({"role": "tool",
                    "content": f"Action '{action}' inconnue. Utilise 'tool' ou 'final_answer'."})

        self._emit(EVT_WARNING, "[TIMEOUT] Nombre maximum d'étapes atteint.")
        return "[TIMEOUT] L'agent n'a pas pu terminer dans le nombre d'étapes autorisé."
