import json
from typing import List, Dict, Any, Optional

# Imports corrigés : les providers sont dans providers/, pas dans tools/
from providers.llm_provider_interface import LLMProvider
from providers.openai_provider import OpenAIProvider
from tools.tool_dispatcher import ToolDispatcher

class SecurityMode:
    """Enum pour définir l'état de sécurité du système agentique."""
    MONITORING = "MONITORING"        # Mode par défaut : tout est bloqué, uniquement visualisation des Intentions.
    LIMITED_SCOPE = "LIMITED_SCOPE"  # Délimite les actions à un chemin donné (e.g., /optimisation).
    FULL_CONTROL = "FULL_CONTROL"    # Risque maximal : Autorise toutes les opérations sans confirmation préalable (Usage réservé).

class AgentOrchestrator:
    """
    Central orchestrator for the ReAct loop, maintenant avec un gestionnaire de sécurité contextuel.
    Il contrôle le cycle ReAct en fonction du mode défini par l'utilisateur/le système.
    """
    def __init__(self, llm_provider: LLMProvider, dispatcher: ToolDispatcher):
        self.llm_provider = llm_provider
        self.dispatcher = dispatcher
        # État de sécurité initial : Monitoring (le plus sûr)
        self.security_mode: str = SecurityMode.MONITORING
        self.current_scope: Optional[str] = None # Scope pour LIMITED_SCOPE

    def set_safety_mode(self, mode: str, scope: Optional[str] = None):
        """Change le mode de sécurité et réinitialise les paramètres associés."""
        if mode not in [SecurityMode.MONITORING, SecurityMode.LIMITED_SCOPE, SecurityMode.FULL_CONTROL]:
             raise ValueError("Mode de sécurité invalide.")

        self.security_mode = mode
        self.current_scope = scope
        print(f"*** 🛡️ MODE SÉCURITÉ PASSE À : {mode} {' (Scope: ' + (scope or 'N/A') + ')' if scope else ''} ***")


    def _check_security(self, action: str, tool_name: str, params: Dict[str, Any]) -> bool:
        """Vérifie si l'action est autorisée selon le mode actuel."""
        if self.security_mode == SecurityMode.MONITORING:
            print("🛑 BLOCKER PAR SÉCURITÉ: Le système est en MODE SURVEILLANCE. Aucune action n'est permise.")
            return False

        elif self.security_mode == SecurityMode.LIMITED_SCOPE:
            if 'file_path' in params and not str(params['file_path']).startswith(self.current_scope):
                print(f"🛑 BLOCKER PAR SCOPE: L'opération doit rester dans {self.current_scope}, mais cible un fichier externe.")
                return False

        elif self.security_mode == SecurityMode.FULL_CONTROL:
            return True

        return True


    def run_agentic_cycle(self, initial_prompt: str, max_steps: int = 8) -> Optional[str]:
        """
        Exécute la boucle ReAct avec vérifications de sécurité intégrées.
        """
        print("\n\n#################################################")
        print(f"### STARTING AGENT CYCLE | Mode Actif: {self.security_mode} ###")
        print("#################################################\n")

        self.chat_history = []
        user_prompt_message = {"role": "user", "content": initial_prompt}
        self.chat_history.append(user_prompt_message)

        final_answer = None  # Initialisation pour éviter UnboundLocalError

        for step in range(max_steps):
            print(f"\n===================== ÉTAPE {step + 1}/{max_steps} ===================")

            print("-> 🤖 [Pensée]: Attente de la décision de l'Agent...")

            raw_llm_output = self.llm_provider.stream_response(initial_prompt, [user_prompt_message])
            # ... (Le reste du parsing/simulation doit être ici) ...

            if step == 0:
                action_to_take = {
                    "tool": "find_files",
                    "params": {"pattern": "*.json", "directory": "G:\\optimisation"}
                }

                if not self._check_security("Find Files", "find_files", action_to_take['params']):
                    observation = "[SECURITY BLOCKED] L'action est bloquée par le mode de sécurité."
                else:
                    print(f"   (Simulation): Sécurité OK. Appel au dispatcher...")
                    observation = self.dispatcher.dispatch_call(
                        tool_name=action_to_take["tool"],
                        **action_to_take["params"]
                    )

            else:
                observation = "Continuation du processus..."

            self.chat_history.append({"role": "tool", "content": f"Resultat: {observation}"})

            if step == 1:
                final_answer = "[Agentic Cycle Completed] L'analyse des fichiers JSON montre un plan détaillé et une approche structurée pour l'implémentation."
                break

        print("=======================================")
        return final_answer

# ... (Le reste du bloc if __name__ == '__main__': de démonstration doit être adapté)
