import json
from typing import Dict, Any, Optional

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


class SecurityMode:
    MONITORING = "MONITORING"
    LIMITED_SCOPE = "LIMITED_SCOPE"
    FULL_CONTROL = "FULL_CONTROL"


class AgentOrchestrator:
    """
    Orchestrateur ReAct réel : Pensée → Action → Observation, jusqu'à réponse finale.
    """
    def __init__(self, llm_provider: LLMProvider, dispatcher: ToolDispatcher):
        self.llm_provider = llm_provider
        self.dispatcher = dispatcher
        self.security_mode: str = SecurityMode.MONITORING
        self.current_scope: Optional[str] = None
        self.chat_history: list = []

    def set_safety_mode(self, mode: str, scope: Optional[str] = None):
        if mode not in [SecurityMode.MONITORING, SecurityMode.LIMITED_SCOPE, SecurityMode.FULL_CONTROL]:
            raise ValueError(f"Mode de sécurité invalide : {mode}")
        self.security_mode = mode
        self.current_scope = scope
        print(f"*** 🛡️ MODE SÉCURITÉ : {mode}{' | Scope: ' + scope if scope else ''} ***")

    def _check_security(self, tool_name: str, params: Dict[str, Any]) -> tuple[bool, str]:
        """Retourne (autorisé, message_erreur)."""
        if self.security_mode == SecurityMode.MONITORING:
            return False, "🛑 Action bloquée : mode MONITORING actif."

        if self.security_mode == SecurityMode.LIMITED_SCOPE and self.current_scope:
            path = params.get('file_path') or params.get('directory', '')
            if path and not str(path).startswith(self.current_scope):
                return False, f"🛑 Action bloquée : le chemin '{path}' est hors du scope '{self.current_scope}'."

        return True, ""

    def _build_system_prompt(self, task: str) -> str:
        history_text = ""
        for msg in self.chat_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "assistant":
                history_text += f"Agent: {content}\n"
            elif role == "tool":
                history_text += f"Observation: {content}\n"

        scope_info = f" | Scope limité : {self.current_scope}" if self.current_scope else ""
        return SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=self.dispatcher.get_tools_description(),
            security_mode=self.security_mode,
            scope_info=scope_info,
            history=history_text or "(aucun historique)",
            task=task
        )

    def _parse_llm_response(self, raw: str) -> Optional[Dict]:
        """Extrait le JSON de la réponse brute du LLM (qui peut contenir du texte autour)."""
        raw = raw.strip()
        # Chercher un bloc JSON entre accolades
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start == -1 or end == 0:
            return None
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            return None

    def run_agentic_cycle(self, initial_prompt: str, max_steps: int = 8) -> Optional[str]:
        """
        Boucle ReAct réelle :
          1. Construire le prompt système avec contexte + historique
          2. Appeler le LLM
          3. Parser la réponse JSON
          4a. Si action=tool : vérifier sécurité, exécuter, ajouter observation à l'historique
          4b. Si action=final_answer : retourner la réponse
          5. Recommencer jusqu'à max_steps
        """
        print("\n#################################################")
        print(f"### AGENT CYCLE | Mode: {self.security_mode} | Max steps: {max_steps} ###")
        print("#################################################\n")

        # Ne pas réinitialiser l'historique pour préserver le contexte entre appels
        task = initial_prompt

        for step in range(max_steps):
            print(f"\n===== ÉTAPE {step + 1}/{max_steps} =====")

            # --- PHASE 1 : Construction du prompt ---
            system_prompt = self._build_system_prompt(task)

            # --- PHASE 2 : Appel LLM (collecte du stream en chaîne complète) ---
            print("🤖 [Pensée] Interrogation du LLM...")
            raw_chunks = []
            for chunk in self.llm_provider.stream_response(system_prompt, []):
                raw_chunks.append(chunk)
            raw_response = "".join(raw_chunks).strip()
            print(f"   Réponse brute : {raw_response[:200]}{'...' if len(raw_response) > 200 else ''}")

            self.chat_history.append({"role": "assistant", "content": raw_response})

            # --- PHASE 3 : Parsing JSON ---
            parsed = self._parse_llm_response(raw_response)

            if not parsed:
                # LLM n'a pas suivi le format JSON — on enregistre et on réessaie
                print("⚠️ Réponse non JSON, nouvelle tentative...")
                self.chat_history.append({
                    "role": "tool",
                    "content": "[FORMAT ERROR] Ta réponse doit être un JSON valide avec 'action' et 'tool'/'answer'."
                })
                continue

            action = parsed.get("action")

            # --- PHASE 4a : Réponse finale ---
            if action == "final_answer":
                answer = parsed.get("answer", "")
                print(f"✅ Réponse finale obtenue après {step + 1} étape(s).")
                return answer

            # --- PHASE 4b : Appel d'outil ---
            elif action == "tool":
                tool_name = parsed.get("tool", "")
                params = parsed.get("params", {})

                print(f"🔧 [Action] Outil demandé : {tool_name} | Params : {params}")

                # Vérification sécurité
                allowed, reason = self._check_security(tool_name, params)
                if not allowed:
                    observation = reason
                    print(f"   {reason}")
                else:
                    observation = self.dispatcher.dispatch_call(tool_name=tool_name, **params)

                print(f"📍 [Observation] {str(observation)[:300]}")
                self.chat_history.append({"role": "tool", "content": str(observation)})

            else:
                # Action inconnue — on informe le LLM
                self.chat_history.append({
                    "role": "tool",
                    "content": f"[ERREUR] Action '{action}' inconnue. Utilise 'tool' ou 'final_answer'."
                })

        # Max steps atteint sans final_answer
        print("⚠️ Nombre maximum d'étapes atteint sans réponse finale.")
        return "[TIMEOUT] L'agent n'a pas pu terminer la tâche dans le nombre d'étapes autorisé."
