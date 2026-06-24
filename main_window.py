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
from tools.tool_dispatcher import ToolDispatcher
from tools.agent_orchestrator import AgentOrchestrator


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
                self.log_signal.emit(f"🤖 Agentic Cycle Terminé.\n\n{final_answer}")
                self.finished_signal.emit(True, "Processus terminé avec succès.")
            else:
                self.log_signal.emit("🛑 Erreur : Le cycle agentique n'a pas pu aboutir à une conclusion finale.")
                self.finished_signal.emit(False, "Échec inconnu.")

        except Exception as e:
            error_msg = f"🚨 EXCEPTION FATALE DANS LE WORKER THREAD : {e}"
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
                    "Entrez votre clé API dans le champ prévu ou définissez OPENAI_API_KEY.")
            return OpenAIProvider(model=model or "gpt-3.5-turbo", api_key=api_key)

        # Fallback Ollama
        return OllamaProvider(model="qwen3:8b")

    def _on_provider_changed(self, index: int):
        """Affiche ou cache le champ API key selon le provider choisi."""
        is_cloud = self.provider_combo.currentText() == "OpenAI / OpenRouter (cloud)"
        self.apikey_label.setVisible(is_cloud)
        self.apikey_input.setVisible(is_cloud)

        # Pré-remplir le modèle par défaut selon le provider
        defaults = {
            "Ollama (local)": "qwen3:8b",
            "LM Studio (local)": "local-model",
            "OpenAI / OpenRouter (cloud)": "gpt-3.5-turbo",
        }
        self.model_input.setPlaceholderText(defaults.get(self.provider_combo.currentText(), ""))

    def _setup_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(8)

        # --- Groupe configuration provider ---
        config_group = QGroupBox("Configuration du Provider LLM")
        config_layout = QFormLayout()
        config_layout.setSpacing(6)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems([
            "Ollama (local)",
            "LM Studio (local)",
            "OpenAI / OpenRouter (cloud)",
        ])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        config_layout.addRow("Provider :", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("qwen3:8b")
        config_layout.addRow("Modèle :", self.model_input)

        self.apikey_label = QLabel("Clé API :")
        self.apikey_input = QLineEdit()
        self.apikey_input.setPlaceholderText("sk-... (ou via OPENAI_API_KEY)")
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
        self.log_output.setText("Bienvenue dans l'Optimisation AI Agent.\nSélectionnez un provider et entrez votre requête.")
        main_layout.addWidget(self.log_output)

        # --- Barre d'entrée ---
        input_hbox = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("▶  Exécuter Agent")
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
        self.log_output.append(f"\n🔌 Provider : {provider_name} | Modèle : {model_name}")
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
            self.log_output.append("✅ Cycle agentique terminé avec succès.")
        else:
            self.log_output.setStyleSheet("color: red;")
            self.log_output.append(f"❌ Échec du processus: {status_message}")
            QTimer.singleShot(3000, lambda: self.log_output.setStyleSheet(""))

        self.log_output.append("===============================\n")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
