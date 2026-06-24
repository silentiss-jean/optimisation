import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox,
    QComboBox, QGroupBox, QFormLayout
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer

from providers.llm_provider_interface import LLMProvider
from providers.ollama_provider import OllamaProvider
from providers.lmstudio_provider import LMStudioProvider
from providers.openai_provider import OpenAIProvider
from providers.perplexity_provider import PerplexityProvider
from tools.tool_dispatcher import ToolDispatcher
from tools.agent_orchestrator import AgentOrchestrator


# Modèles par défaut par provider
PROVIDER_DEFAULTS = {
    "Ollama (local)": "qwen3:8b",
    "LM Studio (local)": "local-model",
    "OpenAI / OpenRouter (cloud)": "gpt-3.5-turbo",
    "Perplexity (Sonar)": "sonar",
}

# Providers nécessitant une clé API
CLOUD_PROVIDERS = {"OpenAI / OpenRouter (cloud)", "Perplexity (Sonar)"}

# Indications clé API par provider
API_KEY_HINTS = {
    "OpenAI / OpenRouter (cloud)": "sk-... (ou via OPENAI_API_KEY)",
    "Perplexity (Sonar)": "pplx-... → https://console.perplexity.ai",
}

# Modèles Perplexity disponibles pour le combobox
PERPLEXITY_MODELS = [
    "sonar",
    "sonar-pro",
    "sonar-reasoning",
    "sonar-reasoning-pro",
    "sonar-deep-research",
]


class LLMWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, initial_prompt: str, llm_provider: LLMProvider, dispatcher: ToolDispatcher):
        super().__init__()
        self.initial_prompt = initial_prompt
        self.llm_provider = llm_provider
        self.dispatcher = dispatcher

    def run(self):
        try:
            orchestrator = AgentOrchestrator(llm_provider=self.llm_provider, dispatcher=self.dispatcher)
            final_answer = orchestrator.run_agentic_cycle(self.initial_prompt)

            if final_answer:
                self.log_signal.emit(f"\U0001f916 Réponse finale :\n\n{final_answer}")
                self.finished_signal.emit(True, "Processus terminé avec succès.")
            else:
                self.log_signal.emit("\U0001f6d1 Le cycle agentique n'a pas pu aboutir à une réponse finale.")
                self.finished_signal.emit(False, "Échec inconnu.")

        except Exception as e:
            error_msg = f"\U0001f6a8 EXCEPTION : {e}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            self.finished_signal.emit(False, str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation AI Agent")
        self.setGeometry(100, 100, 1200, 850)
        self.dispatcher = ToolDispatcher()
        self._setup_ui()

    def _build_provider(self) -> LLMProvider:
        """Instancie le provider sélectionné avec les paramètres saisis dans l'UI."""
        choice = self.provider_combo.currentText()
        model = self.model_input.text().strip()

        if choice == "Ollama (local)":
            return OllamaProvider(model=model or "qwen3:8b")

        elif choice == "LM Studio (local)":
            return LMStudioProvider(model=model or "local-model")

        elif choice == "OpenAI / OpenRouter (cloud)":
            api_key = self.apikey_input.text().strip() or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Clé API manquante",
                    "Entrez votre clé API OpenAI ou définissez OPENAI_API_KEY.")
            return OpenAIProvider(model=model or "gpt-3.5-turbo", api_key=api_key)

        elif choice == "Perplexity (Sonar)":
            api_key = self.apikey_input.text().strip() or os.environ.get("PERPLEXITY_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Clé API manquante",
                    "Entrez votre clé API Perplexity.\n"
                    "Obtenez-la sur https://console.perplexity.ai → API Keys")
            return PerplexityProvider(model=model or "sonar", api_key=api_key)

        # Fallback Ollama
        return OllamaProvider(model="qwen3:8b")

    def _on_provider_changed(self, index: int):
        """Met à jour l'UI selon le provider choisi."""
        choice = self.provider_combo.currentText()
        is_cloud = choice in CLOUD_PROVIDERS

        # Afficher/cacher clé API
        self.apikey_label.setVisible(is_cloud)
        self.apikey_input.setVisible(is_cloud)

        # Hint spécifique à chaque provider cloud
        if is_cloud:
            self.apikey_input.setPlaceholderText(API_KEY_HINTS.get(choice, "Clé API..."))
            self.apikey_input.clear()

        # Pré-remplir le modèle par défaut
        default_model = PROVIDER_DEFAULTS.get(choice, "")
        self.model_input.setPlaceholderText(default_model)
        self.model_input.clear()

        # Si Perplexity : remplacer le champ modèle par un combobox avec les modèles disponibles
        if choice == "Perplexity (Sonar)":
            self.model_input.setPlaceholderText("sonar  (sonar-pro / sonar-reasoning / sonar-deep-research)")

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)

        # --- Groupe configuration provider ---
        config_group = QGroupBox("Configuration du Provider LLM")
        config_layout = QFormLayout()
        config_layout.setSpacing(6)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_DEFAULTS.keys()))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        config_layout.addRow("Provider :", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("qwen3:8b")
        config_layout.addRow("Modèle :", self.model_input)

        self.apikey_label = QLabel("Clé API :")
        self.apikey_input = QLineEdit()
        self.apikey_input.setPlaceholderText("Clé API...")
        self.apikey_input.setEchoMode(QLineEdit.EchoMode.Password)
        config_layout.addRow(self.apikey_label, self.apikey_input)

        # Cacher clé API par défaut (Ollama sélectionné)
        self.apikey_label.setVisible(False)
        self.apikey_input.setVisible(False)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # --- Zone de logs ---
        main_layout.addWidget(QLabel("Historique de Conversation & Logs :"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setText(
            "Bienvenue dans l'Optimisation AI Agent.\n"
            "Providers disponibles : Ollama · LM Studio · OpenAI/OpenRouter · Perplexity (Sonar)\n"
            "Sélectionnez un provider et entrez votre requête."
        )
        main_layout.addWidget(self.log_output)

        # --- Barre d'entrée ---
        input_hbox = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("\u25b6  Exécuter Agent")
        self.submit_button.clicked.connect(self.start_agent_run)
        self.prompt_input.returnPressed.connect(self.start_agent_run)
        input_hbox.addWidget(self.prompt_input)
        input_hbox.addWidget(self.submit_button)
        main_layout.addLayout(input_hbox)

        self.setCentralWidget(central_widget)

    def start_agent_run(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Veuillez entrer une requête avant d'exécuter.")
            return

        provider = self._build_provider()
        provider_name = self.provider_combo.currentText()
        model_name = self.model_input.text().strip() or self.model_input.placeholderText()

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.log_output.append(f"\n\U0001f50c Provider : {provider_name} | Modèle : {model_name}")
        self.log_output.append("... Lancement du cycle agentique ...")

        self.worker = LLMWorker(prompt, provider, self.dispatcher)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.run_finished)
        self.worker.start()

    def update_log(self, message: str):
        self.log_output.append(message + "\n")

    def run_finished(self, success: bool, status_message: str):
        self.submit_button.setEnabled(True)
        self.prompt_input.setEnabled(True)

        if success:
            self.log_output.append("===============================")
            self.log_output.append("\u2705 Cycle agentique terminé avec succès.")
        else:
            self.log_output.setStyleSheet("color: red;")
            self.log_output.append(f"\u274c Échec : {status_message}")
            QTimer.singleShot(3000, lambda: self.log_output.setStyleSheet(""))

        self.log_output.append("===============================\n")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
