import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox,
    QComboBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat

from providers.llm_provider_interface import LLMProvider
from providers.ollama_provider import OllamaProvider
from providers.lmstudio_provider import LMStudioProvider
from providers.openai_provider import OpenAIProvider
from providers.perplexity_provider import PerplexityProvider
from tools.tool_dispatcher import ToolDispatcher
from tools.agent_orchestrator import (
    AgentOrchestrator,
    EVT_THINKING, EVT_ACTION, EVT_OBSERVE, EVT_ANSWER, EVT_STEP, EVT_WARNING, EVT_SECURITY
)


# Modèles par défaut par provider
PROVIDER_DEFAULTS = {
    "Ollama (local)": "qwen3:8b",
    "LM Studio (local)": "local-model",
    "OpenAI / OpenRouter (cloud)": "gpt-3.5-turbo",
    "Perplexity (Sonar)": "sonar",
}
CLOUD_PROVIDERS = {"OpenAI / OpenRouter (cloud)", "Perplexity (Sonar)"}
API_KEY_HINTS = {
    "OpenAI / OpenRouter (cloud)": "sk-... (ou via OPENAI_API_KEY)",
    "Perplexity (Sonar)": "pplx-... → console.perplexity.ai",
}

# Couleurs par type d'événement ReAct
EVENT_COLORS = {
    EVT_STEP:     "#888888",   # gris  — en-tête d'étape
    EVT_THINKING: "#cccccc",   # blanc cassé — tokens LLM
    EVT_ACTION:   "#4fc3f7",   # bleu ciel — appel d'outil
    EVT_OBSERVE:  "#81c784",   # vert — résultat outil
    EVT_ANSWER:   "#fff176",   # jaune — réponse finale
    EVT_WARNING:  "#ff8a65",   # orange — avertissement
    EVT_SECURITY: "#ef5350",   # rouge — blocage sécurité
}

# Préfixes affichés dans le chat
EVENT_PREFIX = {
    EVT_STEP:     "\n╍╍╍ {data} ╍╍╍",
    EVT_THINKING: "{data}",
    EVT_ACTION:   "\n🔧 Outil : {data}",
    EVT_OBSERVE:  "\n📍 Observation : {data}",
    EVT_ANSWER:   "\n✅ Réponse finale :\n{data}",
    EVT_WARNING:  "\n⚠️ {data}",
    EVT_SECURITY: "\n🛑 {data}",
}


class LLMWorker(QThread):
    # Émet (event_type, data) — data peut être vide pour EVT_THINKING de début
    event_signal    = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, initial_prompt: str, llm_provider: LLMProvider,
                 dispatcher: ToolDispatcher):
        super().__init__()
        self.initial_prompt = initial_prompt
        self.llm_provider   = llm_provider
        self.dispatcher     = dispatcher

    def run(self):
        try:
            orchestrator = AgentOrchestrator(
                llm_provider=self.llm_provider,
                dispatcher=self.dispatcher,
                on_event=lambda evt, data: self.event_signal.emit(evt, data)
            )
            final_answer = orchestrator.run_agentic_cycle(self.initial_prompt)
            if final_answer:
                self.finished_signal.emit(True, "")
            else:
                self.finished_signal.emit(False, "Pas de réponse finale.")
        except Exception as e:
            self.event_signal.emit(EVT_WARNING, f"EXCEPTION : {e}")
            self.finished_signal.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation AI Agent")
        self.setGeometry(100, 100, 1200, 850)
        self.dispatcher = ToolDispatcher()
        self._thinking_started = False  # contrôle saut de ligne avant les tokens
        self._setup_ui()

    # ------------------------------------------------------------------ providers
    def _build_provider(self) -> LLMProvider:
        choice = self.provider_combo.currentText()
        model  = self.model_input.text().strip()
        if choice == "Ollama (local)":
            return OllamaProvider(model=model or "qwen3:8b")
        elif choice == "LM Studio (local)":
            return LMStudioProvider(model=model or "local-model")
        elif choice == "OpenAI / OpenRouter (cloud)":
            api_key = self.apikey_input.text().strip() or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Clé API", "Entrez votre clé OpenAI ou définissez OPENAI_API_KEY.")
            return OpenAIProvider(model=model or "gpt-3.5-turbo", api_key=api_key)
        elif choice == "Perplexity (Sonar)":
            api_key = self.apikey_input.text().strip() or os.environ.get("PERPLEXITY_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Clé API",
                    "Entrez votre clé Perplexity.\nhttps://console.perplexity.ai")
            return PerplexityProvider(model=model or "sonar", api_key=api_key)
        return OllamaProvider(model="qwen3:8b")

    def _on_provider_changed(self, _index: int):
        choice   = self.provider_combo.currentText()
        is_cloud = choice in CLOUD_PROVIDERS
        self.apikey_label.setVisible(is_cloud)
        self.apikey_input.setVisible(is_cloud)
        if is_cloud:
            self.apikey_input.setPlaceholderText(API_KEY_HINTS.get(choice, ""))
            self.apikey_input.clear()
        self.model_input.setPlaceholderText(PROVIDER_DEFAULTS.get(choice, ""))
        self.model_input.clear()

    # ------------------------------------------------------------------ UI
    def _setup_ui(self):
        central = QWidget()
        layout  = QVBoxLayout(central)
        layout.setSpacing(8)

        # --- Config provider ---
        cfg_group  = QGroupBox("Configuration du Provider LLM")
        cfg_layout = QFormLayout()
        cfg_layout.setSpacing(6)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_DEFAULTS.keys()))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        cfg_layout.addRow("Provider :", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("qwen3:8b")
        cfg_layout.addRow("Modèle :", self.model_input)

        self.apikey_label = QLabel("Clé API :")
        self.apikey_input = QLineEdit()
        self.apikey_input.setEchoMode(QLineEdit.EchoMode.Password)
        cfg_layout.addRow(self.apikey_label, self.apikey_input)
        self.apikey_label.setVisible(False)
        self.apikey_input.setVisible(False)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        # --- Chat (fond sombre pour le streaming coloré) ---
        layout.addWidget(QLabel("Conversation & Étapes ReAct :"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color: #1e1e1e; color: #cccccc; "
            "font-family: 'Consolas', monospace; font-size: 13px;"
        )
        self._append_colored(
            "Bienvenue dans l'Optimisation AI Agent.\n"
            "Providers : Ollama · LM Studio · OpenAI · Perplexity (Sonar)",
            "#888888"
        )
        layout.addWidget(self.log_output)

        # --- Barre d'entrée ---
        bar = QHBoxLayout()
        self.prompt_input  = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("▶  Exécuter Agent")
        self.submit_button.clicked.connect(self.start_agent_run)
        self.prompt_input.returnPressed.connect(self.start_agent_run)
        bar.addWidget(self.prompt_input)
        bar.addWidget(self.submit_button)
        layout.addLayout(bar)

        self.setCentralWidget(central)

    # ------------------------------------------------------------------ chat helpers
    def _append_colored(self, text: str, color: str):
        """Ajoute du texte coloré dans le chat sans saut de ligne supplémentaire."""
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _on_event(self, event_type: str, data: str):
        """Reçoit chaque événement de l'orchestrateur et l'affiche dans le chat."""
        color   = EVENT_COLORS.get(event_type, "#cccccc")
        pattern = EVENT_PREFIX.get(event_type, "{data}")

        if event_type == EVT_THINKING:
            if not data:  # signal de début pensée
                if self._thinking_started:
                    return
                self._thinking_started = True
                self._append_colored("\n🤔 Pensée : ", "#888888")
                return
            # chunk token — affichage inline sans saut de ligne
            self._append_colored(data, color)
            return

        # Pour tous les autres événements : on reset le flag thinking
        self._thinking_started = False
        text = pattern.format(data=data)
        self._append_colored(text, color)

    # ------------------------------------------------------------------ agent run
    def start_agent_run(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Entrez une requête.")
            return

        provider      = self._build_provider()
        provider_name = self.provider_combo.currentText()
        model_name    = self.model_input.text().strip() or self.model_input.placeholderText()

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self._thinking_started = False

        self._append_colored(
            f"\n\n🗨️ Vous : {prompt}\n🔌 {provider_name} | {model_name}\n",
            "#aaaaaa"
        )

        self.worker = LLMWorker(prompt, provider, self.dispatcher)
        self.worker.event_signal.connect(self._on_event)
        self.worker.finished_signal.connect(self._run_finished)
        self.worker.start()

    def _run_finished(self, success: bool, error: str):
        self.submit_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.prompt_input.clear()
        self._thinking_started = False

        if success:
            self._append_colored("\n═" * 40 + "\n", "#444444")
        else:
            self._append_colored(f"\n❌ Échec : {error}\n", "#ef5350")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
