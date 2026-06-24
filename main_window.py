import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer

# Imports absolus pour compatibilité PyInstaller
from providers.openai_provider import OpenAIProvider
from tools.tool_dispatcher import ToolDispatcher
from tools.agent_orchestrator import AgentOrchestrator


class LLMWorker(QThread):
    """
    Worker thread responsible for executing long-running and blocking tasks
    (like API calls or tool execution) without freezing the main GUI loop.
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, initial_prompt: str, llm_provider: OpenAIProvider, dispatcher: ToolDispatcher):
        super().__init__()
        self.initial_prompt = initial_prompt
        self.llm_provider = llm_provider
        self.dispatcher = dispatcher

    def run(self):
        """Méthode principale exécutée dans le thread séparé."""
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
    """
    La fenêtre principale de l'application qui gère l'UI et les signaux/slots.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation AI Agent - Assistant LLM Interactif")
        self.setGeometry(100, 100, 1200, 800)

        # Clé API lue depuis la variable d'environnement (ne jamais mettre en dur dans le code)
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            QMessageBox.warning(
                None, "Clé API manquante",
                "La variable d'environnement OPENAI_API_KEY n'est pas définie.\n"
                "Définissez-la avant de lancer l'application :\n"
                "  $env:OPENAI_API_KEY = 'sk-...'"
            )

        self.llm_provider = OpenAIProvider(model="gpt-3.5-turbo", api_key=api_key)
        self.dispatcher = ToolDispatcher()

        self._setup_ui()

    def _setup_ui(self):
        """Initialise le layout de la fenêtre principale."""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setText("Bienvenue dans l'Optimisation AI Agent.\nPrêt à interagir avec LLMs locaux ou cloud.")
        main_layout.addWidget(QLabel("Historique de Conversation & Logs:"))
        main_layout.addWidget(self.log_output)

        input_hbox = QHBoxLayout()
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("Exécuter Agent")

        self.submit_button.clicked.connect(self.start_agent_run)
        self.prompt_input.returnPressed.connect(self.start_agent_run)

        input_hbox.addWidget(self.prompt_input)
        input_hbox.addWidget(self.submit_button)
        main_layout.addLayout(input_hbox)

        self.setCentralWidget(central_widget)

    def start_agent_run(self):
        """Lance le thread de l'AgentOrchestrator dans un QThread."""
        prompt = self.prompt_input.text()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Veuillez entrer une requête avant d'exécuter.")
            return

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self.log_output.append("... Lancement du cycle agentique en cours ...")

        self.worker = LLMWorker(prompt, self.llm_provider, self.dispatcher)
        self.worker.log_signal.connect(self.update_log)
        self.worker.finished_signal.connect(self.run_finished)
        self.worker.start()

    def update_log(self, message: str):
        """Slot appelé par le worker thread pour afficher les messages de log."""
        self.log_output.append(message + "\n\n")

    def run_finished(self, success: bool, status_message: str):
        """Slot appelé quand le worker thread a terminé son travail."""
        self.submit_button.setEnabled(True)
        self.prompt_input.setEnabled(True)

        if success:
            self.log_output.append("===============================")
            self.log_output.append("✅ Cycle agentique terminé avec succès.")
        else:
            self.log_output.setStyleSheet("color: red;")
            self.log_output.append(f"❌ Échec du processus: {status_message}")
            # Correction : réinitialiser le style sur log_output, pas sur self
            QTimer.singleShot(2000, lambda: self.log_output.setStyleSheet(""))

        self.log_output.append("===============================\n")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
