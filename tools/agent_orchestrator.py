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
2. UNE SEULE action par réponse. Si la tâche demande plusieurs actions, fais la première,
   attends l'observation, puis fais la suivante.
3. Pour utiliser un outil :
   {{"action": "tool", "tool": "<nom_outil>", "params": {{...}}}}
4. Quand TOUTES les actions demandées sont terminées, appelle final_answer :
   {{"action": "final_answer", "answer": "<résumé de ce qui a été fait>"}}
5. En mode MONITORING, pas d'outils — final_answer directement.
6. Pour ouvrir une URL dans l'onglet actif : browser_navigate.
   Pour ouvrir un NOUVEL onglet sans fermer la page courante : browser_new_tab.
7. Pour lire le contenu d'une page (repos, texte, données) : browser_navigate PUIS browser_get_text.
   N'utilise JAMAIS open_url pour récupérer du contenu — open_url ne lit pas la page.
8. Si un outil retourne [ERREUR], essaie un outil DIFFÉRENT — ne répète jamais la même commande.
9. "Ouvrir" une URL, un site, un profil GitHub = toujours browser_navigate ou browser_new_tab.
   Ne jamais utiliser command_line_execute pour naviguer sur le web.
10. Pour lister les repos GitHub d'un utilisateur : utilise web_scrape avec
    https://api.github.com/users/{{username}}/repos — c'est plus fiable que de scraper le DOM.
11. Certains outils LIVRENT directement une information (browser_get_text, web_scrape, read_file,
    command_line_execute, browser_screenshot) — leur observation EST la réponse à la question.
    Lis l'observation et appelle immédiatement final_answer avec cette information.
    Ne navigue PAS vers une page si tu viens déjà de récupérer la donnée demandée.

Historique de la conversation :
{history}

Tâche : {task}"""


EVT_THINKING   = "thinking"
EVT_ACTION     = "action"
EVT_OBSERVE    = "observe"
EVT_ANSWER     = "answer"
EVT_STEP       = "step"
EVT_WARNING    = "warning"
EVT_SECURITY   = "security"
EVT_SCREENSHOT = "screenshot"   # nouveau — data = chemin absolu vers l'image

# Nombre de caractères affichés dans le log UI pour une observation
OBS_UI_MAX   = 2000
# Nombre de caractères transmis dans l'historique LLM pour une observation brute
OBS_LLM_MAX  = 4000


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

# Outils qui LIVRENT une donnée — l'observation est la réponse, pas une étape intermédiaire.
# Leur hint pousse l'agent à appeler final_answer immédiatement avec le contenu reçu.
TERMINAL_TOOLS: set = {
    "web_scrape", "browser_get_text", "read_file",
    "command_line_execute", "browser_screenshot",
}

# Outils qui exécutent une action sans livrer de donnée directement utile à l'utilisateur.
# Leur hint indique à l'agent de continuer si d'autres étapes sont nécessaires, ou d'appeler
# final_answer si toutes les tâches sont terminées.
INTERMEDIATE_TOOLS: set = {
    "browser_navigate", "browser_new_tab", "browser_click",
    "browser_fill", "browser_scroll", "browser_wait_for", "open_url",
    "write_file", "find_files",
}


def _strip_think_blocks(text: str) -> str:
    """Supprime les blocs <think>...</think> générés par certains modèles (ex: qwen3)."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def _extract_json_objects(text: str) -> list[dict]:
    """
    Extrait tous les objets JSON valides et complets d'une chaîne,
    même si la chaîne globale est tronquée (tableau incomplet).
    """
    objects = []
    i = 0
    while i < len(text):
        if text[i] != '{':
            i += 1
            continue
        depth = 0
        in_string = False
        escape_next = False
        j = i
        while j < len(text):
            c = text[j]
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
                        candidate = text[i:j + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict):
                                objects.append(obj)
                        except json.JSONDecodeError:
                            pass
                        break
            j += 1
        i = j + 1
    return objects


def _compact_github_repos(raw: str) -> str:
    """
    Détecte une réponse contenant des objets repo GitHub (complets ou tronqués)
    et retourne une liste lisible nom : description (url).
    Fonctionne même si le JSON global est incomplet/tronqué.
    Retourne raw intact si aucun repo détecté.
    """
    stripped = raw.strip()
    if 'html_url' not in stripped or 'full_name' not in stripped:
        return raw

    if stripped.startswith('['):
        try:
            repos = json.loads(stripped)
            if isinstance(repos, list) and repos and 'name' in repos[0]:
                lines = [
                    f"- {r.get('name', '?')} : {r.get('description') or '(pas de description)'}  "
                    f"({r.get('html_url', '')})"
                    for r in repos if isinstance(r, dict)
                ]
                return "Repos GitHub (" + str(len(lines)) + ") :\n" + "\n".join(lines)
        except json.JSONDecodeError:
            pass

    objects = _extract_json_objects(stripped)
    repos = [o for o in objects if 'name' in o and 'html_url' in o and 'full_name' in o]
    if repos:
        lines = [
            f"- {r.get('name', '?')} : {r.get('description') or '(pas de description)'}  "
            f"({r.get('html_url', '')})"
            for r in repos
        ]
        suffix = " (liste potentiellement partielle)" if not stripped.endswith(']') else ""
        return f"Repos GitHub ({len(lines)}{suffix}) :\n" + "\n".join(lines)

    return raw


def _extract_screenshot_path(obs: str) -> Optional[str]:
    """
    Si l'observation d'un browser_screenshot contient un chemin de fichier image,
    le retourne. Sinon retourne None.
    Format attendu: "[DONE] Screenshot sauvegardé : /chemin/absolu/screenshot.png"
    """
    m = re.search(r'Screenshot sauvegardé\s*:\s*(.+\.png)', obs)
    if m:
        path = m.group(1).strip()
        import os
        return path if os.path.isfile(path) else None
    return None


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
                return False, "\U0001f6d1 Action bloquée : mode MONITORING actif."
            return False, (
                f"\U0001f6d1 Outil '{tool_name}' non autorisé en mode {self.security_mode}. "
                f"Outils disponibles : {', '.join(sorted(allowed_set)) or 'aucun'}."
            )
        if self.security_mode == SecurityMode.LIMITED_SCOPE and self.current_scope:
            path = params.get('file_path') or params.get('directory', '')
            if path and not str(path).startswith(self.current_scope):
                return False, f"\U0001f6d1 Chemin '{path}' hors du scope autorisé '{self.current_scope}'."
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
        Strippes les blocs <think> avant parsing.
        Retourne (parsed, had_multiple).
        """
        raw = _strip_think_blocks(raw)
        candidates = [m.start() for m in re.finditer(r'\{', raw)]
        had_multiple = False
        first_parsed = None

        for start in candidates:
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
                                    break
                            except json.JSONDecodeError:
                                pass
                            break
                i += 1
            if had_multiple:
                break

        return first_parsed, had_multiple

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
                self._emit(EVT_WARNING, "[FORMAT] Plusieurs JSON détectés — seule la première action est exécutée.")

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
                    obs_for_llm = reason + "\nUtilise 'final_answer' pour expliquer à l'utilisateur."
                    self._emit(EVT_OBSERVE, obs_for_llm[:OBS_UI_MAX])
                    self.chat_history.append({"role": "tool", "content": obs_for_llm})
                    continue

                observation = self.dispatcher.dispatch_call(tool_name=tool_name, **params)
                obs_str = str(observation)

                # Compacter les réponses JSON verbeux (repos GitHub, etc.)
                obs_compact = _compact_github_repos(obs_str)

                # --- P2-002 : détecter un screenshot et émettre l'événement dédié
                if tool_name == "browser_screenshot":
                    img_path = _extract_screenshot_path(obs_compact)
                    if img_path:
                        self._emit(EVT_SCREENSHOT, img_path)

                # --- P2-001 : hints différenciés selon la catégorie de l'outil
                if obs_compact.startswith("[ERREUR]"):
                    obs_for_llm = (
                        obs_compact +
                        "\n→ Cet outil a échoué. Utilise un outil DIFFÉRENT ou appelle "
                        "final_answer pour expliquer l'échec. Ne répète PAS la même commande."
                    )
                elif tool_name in TERMINAL_TOOLS:
                    # L'outil a livré une donnée : l'agent doit lire et répondre
                    obs_for_llm = (
                        obs_compact +
                        "\n→ Donnée reçue. Lis l'observation ci-dessus et appelle "
                        "final_answer avec cette information. N'utilise plus d'outils."
                    )
                elif tool_name in INTERMEDIATE_TOOLS:
                    # L'outil a exécuté une action intermédiaire
                    obs_for_llm = (
                        obs_compact +
                        "\n→ Action exécutée. Si d'autres étapes sont nécessaires, "
                        "continue. Sinon appelle final_answer."
                    )
                else:
                    obs_for_llm = obs_compact

                self._emit(EVT_OBSERVE, obs_for_llm[:OBS_UI_MAX])
                self.chat_history.append({"role": "tool", "content": obs_for_llm[:OBS_LLM_MAX]})

            else:
                self._emit(EVT_WARNING, f"[ERREUR] Action '{action}' inconnue.")
                self.chat_history.append({"role": "tool",
                    "content": f"Action '{action}' inconnue. Utilise 'tool' ou 'final_answer'."})

        self._emit(EVT_WARNING, "[TIMEOUT] Nombre maximum d'étapes atteint.")
        return "[TIMEOUT] L'agent n'a pas pu terminer dans le nombre d'étapes autorisé."
