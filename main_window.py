import sys
import os
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QButtonGroup,
    QFileDialog
)
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat

from providers.llm_provider_interface import LLMProvider
from providers.ollama_provider import OllamaProvider
from providers.lmstudio_provider import LMStudioProvider
from providers.openai_provider import OpenAIProvider
from providers.perplexity_provider import PerplexityProvider
from tools.tool_dispatcher import ToolDispatcher
from tools.agent_orchestrator import (
    AgentOrchestrator, SecurityMode,
    EVT_THINKING, EVT_ACTION, EVT_OBSERVE, EVT_ANSWER,
    EVT_STEP, EVT_WARNING, EVT_SECURITY
)

# ─────────────────────────────────────────────────────────── providers
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

# ─────────────────────────────────────────────────────── streaming colors
EVENT_COLORS = {
    EVT_STEP:     "#888888",
    EVT_THINKING: "#cccccc",
    EVT_ACTION:   "#4fc3f7",
    EVT_OBSERVE:  "#81c784",
    EVT_ANSWER:   "#fff176",
    EVT_WARNING:  "#ff8a65",
    EVT_SECURITY: "#ef5350",
}
EVENT_PREFIX = {
    EVT_STEP:     "\n╍╍╍ {data} ╍╍╍",
    EVT_THINKING: "{data}",
    EVT_ACTION:   "\n🔧 Outil : {data}",
    EVT_OBSERVE:  "\n📍 Observation : {data}",
    EVT_ANSWER:   "\n✅ Réponse finale :\n{data}",
    EVT_WARNING:  "\n⚠️ {data}",
    EVT_SECURITY: "\n🛑 {data}",
}

# badge visuel par mode sécurité
SECURITY_BADGE = {
    SecurityMode.MONITORING:    ("MONITORING — lecture seule",    "#1e1e1e", "#888888"),
    SecurityMode.LIMITED_SCOPE: ("LIMITED SCOPE — dossier limité", "#1a2a1a", "#81c784"),
    SecurityMode.FULL_CONTROL:  ("FULL CONTROL — accès total",    "#2a1a1a", "#ef5350"),
}


# ═══════════════════════════════════════════════════════════════ Worker
class LLMWorker(QThread):
    event_signal    = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, prompt: str, provider: LLMProvider,
                 dispatcher: ToolDispatcher,
                 security_mode: str, scope: str):
        super().__init__()
        self.prompt        = prompt
        self.provider      = provider
        self.dispatcher    = dispatcher
        self.security_mode = security_mode
        self.scope         = scope or None

    def run(self):
        try:
            orc = AgentOrchestrator(
                llm_provider=self.provider,
                dispatcher=self.dispatcher,
                on_event=lambda evt, data: self.event_signal.emit(evt, data)
            )
            orc.set_safety_mode(self.security_mode, self.scope)
            answer = orc.run_agentic_cycle(self.prompt)
            self.finished_signal.emit(bool(answer), "")
        except Exception as e:
            self.event_signal.emit(EVT_WARNING, f"EXCEPTION : {e}")
            self.finished_signal.emit(False, str(e))


# ═══════════════════════════════════════════════════════════ MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation AI Agent")
        self.setGeometry(100, 100, 1260, 900)
        self.dispatcher        = ToolDispatcher()
        self._thinking_started = False
        self._setup_ui()

    # ──────────────────────────────────────────────────── build provider
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
                QMessageBox.warning(self, "Clé API", "Définissez OPENAI_API_KEY ou entrez la clé.")
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

    # ─────────────────────────────────────────────────────── security UI
    def _current_security_mode(self) -> str:
        if self.radio_monitoring.isChecked():
            return SecurityMode.MONITORING
        if self.radio_limited.isChecked():
            return SecurityMode.LIMITED_SCOPE
        return SecurityMode.FULL_CONTROL

    def _on_security_changed(self):
        """Met à jour le badge et active/désactive le champ scope."""
        mode = self._current_security_mode()
        label, bg, fg = SECURITY_BADGE[mode]
        self.security_badge.setText(f"  {label}  ")
        self.security_badge.setStyleSheet(
            f"background-color:{bg}; color:{fg}; "
            "font-weight:bold; border-radius:4px; padding:3px 8px;"
        )
        show_scope = (mode == SecurityMode.LIMITED_SCOPE)
        self.scope_label.setVisible(show_scope)
        self.scope_input.setVisible(show_scope)
        self.scope_browse.setVisible(show_scope)

    def _browse_scope(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier scope")
        if folder:
            self.scope_input.setText(folder)

    # ────────────────────────────────────────────────────────── setup UI
    def _setup_ui(self):
        central = QWidget()
        root    = QVBoxLayout(central)
        root.setSpacing(8)

        # ── top row: provider + sécurité côte-à-côte
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        # ---- Provider group
        cfg_group  = QGroupBox("Provider LLM")
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
        top_row.addWidget(cfg_group, stretch=1)

        # ---- Sécurité group
        sec_group  = QGroupBox("🛡️ Mode Sécurité")
        sec_layout = QVBoxLayout()
        sec_layout.setSpacing(6)

        # Badge de statut
        self.security_badge = QLabel()
        self.security_badge.setFixedHeight(28)
        sec_layout.addWidget(self.security_badge)

        # Boutons radio
        self.radio_monitoring = QRadioButton(
            "🔴  MONITORING   — aucune action, réponse directe seulement"
        )
        self.radio_limited    = QRadioButton(
            "🟡  LIMITED SCOPE — accès limité à un dossier"
        )
        self.radio_full       = QRadioButton(
            "🟢  FULL CONTROL  — accès complet système"
        )
        self.radio_monitoring.setChecked(True)

        self.sec_btn_group = QButtonGroup()
        for r in (self.radio_monitoring, self.radio_limited, self.radio_full):
            self.sec_btn_group.addButton(r)
            sec_layout.addWidget(r)
            r.toggled.connect(self._on_security_changed)

        # Scope (dossier)
        scope_row = QHBoxLayout()
        self.scope_label  = QLabel("Dossier scope :")
        self.scope_input  = QLineEdit()
        self.scope_input.setPlaceholderText("C:/mon/dossier")
        self.scope_browse = QPushButton("📂")
        self.scope_browse.setFixedWidth(32)
        self.scope_browse.setToolTip("Parcourir")
        self.scope_browse.clicked.connect(self._browse_scope)
        scope_row.addWidget(self.scope_label)
        scope_row.addWidget(self.scope_input)
        scope_row.addWidget(self.scope_browse)
        sec_layout.addLayout(scope_row)

        sec_group.setLayout(sec_layout)
        top_row.addWidget(sec_group, stretch=1)
        root.addLayout(top_row)

        # init badge
        self._on_security_changed()
        # masquer scope par défaut
        self.scope_label.setVisible(False)
        self.scope_input.setVisible(False)
        self.scope_browse.setVisible(False)

        # ── Chat
        root.addWidget(QLabel("Conversation & Étapes ReAct :"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color:#1e1e1e; color:#cccccc;"
            "font-family:'Consolas',monospace; font-size:13px;"
        )
        self._append_colored(
            "Bienvenue dans l'Optimisation AI Agent.\n"
            "Sélectionnez un provider et un mode sécurité, puis lancez votre requête.",
            "#888888"
        )
        root.addWidget(self.log_output)

        # ── Barre de saisie
        bar = QHBoxLayout()
        self.prompt_input  = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("▶  Exécuter Agent")
        self.submit_button.setFixedHeight(36)
        self.submit_button.clicked.connect(self.start_agent_run)
        self.prompt_input.returnPressed.connect(self.start_agent_run)
        bar.addWidget(self.prompt_input)
        bar.addWidget(self.submit_button)
        root.addLayout(bar)

        self.setCentralWidget(central)

    # ──────────────────────────────────────────────────── chat helpers
    def _append_colored(self, text: str, color: str):
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _on_event(self, event_type: str, data: str):
        color   = EVENT_COLORS.get(event_type, "#cccccc")
        pattern = EVENT_PREFIX.get(event_type, "{data}")

        if event_type == EVT_THINKING:
            if not data:
                if self._thinking_started:
                    return
                self._thinking_started = True
                self._append_colored("\n🤔 Pensée : ", "#888888")
                return
            self._append_colored(data, color)
            return

        self._thinking_started = False
        self._append_colored(pattern.format(data=data), color)

    # ──────────────────────────────────────────────────── run agent
    def start_agent_run(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Entrez une requête.")
            return

        mode  = self._current_security_mode()
        scope = self.scope_input.text().strip() if mode == SecurityMode.LIMITED_SCOPE else ""

        # Confirmation si FULL_CONTROL
        if mode == SecurityMode.FULL_CONTROL:
            reply = QMessageBox.warning(
                self, "🟢 Confirmation FULL CONTROL",
                "L'agent aura accès TOTAL au système (fichiers, commandes OS).\n"
                "Confirmez-vous l'exécution ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        provider      = self._build_provider()
        provider_name = self.provider_combo.currentText()
        model_name    = self.model_input.text().strip() or self.model_input.placeholderText()
        badge_label   = SECURITY_BADGE[mode][0]

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self._thinking_started = False

        self._append_colored(
            f"\n\n🗨️ Vous : {prompt}\n"
            f"🔌 {provider_name} | {model_name}\n"
            f"🛡️  {badge_label}{' | scope: ' + scope if scope else ''}\n",
            "#aaaaaa"
        )

        self.worker = LLMWorker(prompt, provider, self.dispatcher, mode, scope)
        self.worker.event_signal.connect(self._on_event)
        self.worker.finished_signal.connect(self._run_finished)
        self.worker.start()

    def _run_finished(self, success: bool, error: str):
        self.submit_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.prompt_input.clear()
        self._thinking_started = False
        if success:
            self._append_colored("\n" + "═" * 60 + "\n", "#444444")
        else:
            self._append_colored(f"\n❌ Échec : {error}\n", "#ef5350")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
