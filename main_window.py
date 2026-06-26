import sys
import os
import json
import urllib.request
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QButtonGroup,
    QFileDialog, QDialog, QDialogButtonBox, QStackedWidget
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
LOCAL_PROVIDERS = {"Ollama (local)", "LM Studio (local)"}
API_KEY_HINTS = {
    "OpenAI / OpenRouter (cloud)": "sk-... (ou via OPENAI_API_KEY)",
    "Perplexity (Sonar)": "pplx-... \u2192 console.perplexity.ai",
}

# Endpoints pour lister les mod\u00e8les locaux
MODEL_LIST_ENDPOINTS = {
    "Ollama (local)": ("http://localhost:11434/api/tags",
                       lambda d: [m["name"] for m in d.get("models", [])]),
    "LM Studio (local)": ("http://localhost:1234/v1/models",
                           lambda d: [m["id"] for m in d.get("data", [])]),
}

# ─────────────────────────────────────────────────── streaming colors
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
    EVT_STEP:     "\n\u254d\u254d\u254d {data} \u254d\u254d\u254d",
    EVT_THINKING: "{data}",
    EVT_ACTION:   "\n\U0001f527 Outil : {data}",
    EVT_OBSERVE:  "\n\U0001f4cd Observation : {data}",
    EVT_ANSWER:   "\n\u2705 R\u00e9ponse finale :\n{data}",
    EVT_WARNING:  "\n\u26a0\ufe0f {data}",
    EVT_SECURITY: "\n\U0001f6d1 {data}",
}

SECURITY_BADGE = {
    SecurityMode.MONITORING:    ("MONITORING \u2014 lecture seule",    "#1e1e1e", "#888888"),
    SecurityMode.LIMITED_SCOPE: ("LIMITED SCOPE \u2014 dossier limit\u00e9", "#1a2a1a", "#81c784"),
    SecurityMode.FULL_CONTROL:  ("FULL CONTROL \u2014 acc\u00e8s total",    "#2a1a1a", "#ef5350"),
}

# Dur\u00e9es de confiance FULL CONTROL
TRUST_DURATIONS = [
    ("Cette requ\u00eate uniquement",   None),
    ("1 heure",                    timedelta(hours=1)),
    ("Jusqu'\u00e0 demain minuit",      None),
    ("Cette session",              timedelta(days=3650)),
]


# ═══════════════════════════════════════════════════ Dialog de confiance
class TrustDialog(QDialog):
    """Popup FULL CONTROL avec choix de dur\u00e9e."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\U0001f7e2 FULL CONTROL \u2014 Confirmation")
        self.setMinimumWidth(420)
        self.selected_duration = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>L'agent aura acc\u00e8s TOTAL au syst\u00e8me</b><br>"
            "(fichiers, commandes OS, web).<br><br>"
            "Pendant combien de temps accorder l'acc\u00e8s ?"
        ))

        self.btn_group = QButtonGroup(self)
        self.radios = []
        for i, (label, _) in enumerate(TRUST_DURATIONS):
            r = QRadioButton(label)
            if i == 0:
                r.setChecked(True)
            self.btn_group.addButton(r, i)
            layout.addWidget(r)
            self.radios.append(r)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self):
        idx = self.btn_group.checkedId()
        label, delta = TRUST_DURATIONS[idx]
        if label == "Jusqu'\u00e0 demain minuit":
            tomorrow = datetime.now().replace(
                hour=23, minute=59, second=59, microsecond=0
            ) + timedelta(days=1)
            self.selected_duration = tomorrow - datetime.now()
        else:
            self.selected_duration = delta
        self.accept()


# ═══════════════════════════════════════════════════════════ Worker
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
        self._trust_expiry: datetime | None = None
        self._setup_ui()

    # ──────────────────────────────── fetch models from local server
    def _fetch_local_models(self):
        """GET l'API locale du provider s\u00e9lectionn\u00e9 et remplit le combo mod\u00e8les."""
        choice = self.provider_combo.currentText()
        if choice not in MODEL_LIST_ENDPOINTS:
            return
        url, extractor = MODEL_LIST_ENDPOINTS[choice]
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            models = extractor(data)
            if not models:
                QMessageBox.information(self, "Mod\u00e8les", "Aucun mod\u00e8le trouv\u00e9 (liste vide).")
                return
            current = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(models)
            # Restaurer la s\u00e9lection pr\u00e9c\u00e9dente si toujours dans la liste
            if current in models:
                self.model_combo.setCurrentText(current)
        except Exception as e:
            QMessageBox.warning(
                self, "Connexion impossible",
                f"Impossible de joindre {url}\n\n{e}\n\n"
                f"V\u00e9rifiez que {choice} est bien d\u00e9marr\u00e9."
            )

    # ──────────────────────────────── build provider
    def _build_provider(self) -> LLMProvider:
        choice = self.provider_combo.currentText()
        # Mod\u00e8le : combo pour local, texte pour cloud
        if choice in LOCAL_PROVIDERS:
            model = self.model_combo.currentText().strip()
        else:
            model = self.model_input.text().strip()

        if choice == "Ollama (local)":
            return OllamaProvider(model=model or "qwen3:8b")
        elif choice == "LM Studio (local)":
            return LMStudioProvider(model=model or "local-model")
        elif choice == "OpenAI / OpenRouter (cloud)":
            api_key = self.apikey_input.text().strip() or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Cl\u00e9 API", "D\u00e9finissez OPENAI_API_KEY ou entrez la cl\u00e9.")
            return OpenAIProvider(model=model or "gpt-3.5-turbo", api_key=api_key)
        elif choice == "Perplexity (Sonar)":
            api_key = self.apikey_input.text().strip() or os.environ.get("PERPLEXITY_API_KEY", "")
            if not api_key:
                QMessageBox.warning(self, "Cl\u00e9 API",
                    "Entrez votre cl\u00e9 Perplexity.\nhttps://console.perplexity.ai")
            return PerplexityProvider(model=model or "sonar", api_key=api_key)
        return OllamaProvider(model="qwen3:8b")

    def _on_provider_changed(self, _index: int):
        choice   = self.provider_combo.currentText()
        is_cloud = choice in CLOUD_PROVIDERS
        is_local = choice in LOCAL_PROVIDERS

        # Afficher combo ou input selon le type
        self.model_stack.setCurrentIndex(0 if is_local else 1)

        self.apikey_label.setVisible(is_cloud)
        self.apikey_input.setVisible(is_cloud)
        self.refresh_btn.setVisible(is_local)

        if is_cloud:
            self.apikey_input.setPlaceholderText(API_KEY_HINTS.get(choice, ""))
            self.apikey_input.clear()
            self.model_input.setPlaceholderText(PROVIDER_DEFAULTS.get(choice, ""))
            self.model_input.clear()
        else:
            # R\u00e9initialiser le combo avec le default
            self.model_combo.clear()
            self.model_combo.addItem(PROVIDER_DEFAULTS.get(choice, ""))

    # ─────────────────────────────────────────────── security UI
    def _current_security_mode(self) -> str:
        if self.radio_monitoring.isChecked():
            return SecurityMode.MONITORING
        if self.radio_limited.isChecked():
            return SecurityMode.LIMITED_SCOPE
        return SecurityMode.FULL_CONTROL

    def _on_security_changed(self):
        mode = self._current_security_mode()
        if mode != SecurityMode.FULL_CONTROL:
            self._trust_expiry = None
            self._update_trust_badge()
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

    def _is_trust_active(self) -> bool:
        if self._trust_expiry is None:
            return False
        return datetime.now() < self._trust_expiry

    def _update_trust_badge(self):
        if self._is_trust_active():
            expiry_str = self._trust_expiry.strftime("%H:%M")
            self.trust_label.setText(f"  \u2705 Session de confiance active jusqu'\u00e0 {expiry_str}  ")
            self.trust_label.setStyleSheet(
                "background-color:#1a2a1a; color:#81c784; "
                "font-size:11px; border-radius:4px; padding:2px 6px;"
            )
            self.trust_label.setVisible(True)
        else:
            self.trust_label.setVisible(False)

    def _browse_scope(self):
        folder = QFileDialog.getExistingDirectory(self, "Choisir le dossier scope")
        if folder:
            self.scope_input.setText(folder)

    # ─────────────────────────────────────────────────────── setup UI
    def _setup_ui(self):
        central = QWidget()
        root    = QVBoxLayout(central)
        root.setSpacing(8)

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

        # ---- Widget empilement : combo (local) ou QLineEdit (cloud)
        self.model_stack = QStackedWidget()
        self.model_stack.setFixedHeight(32)

        # Page 0 : combo + bouton refresh pour providers locaux
        local_widget = QWidget()
        local_layout = QHBoxLayout(local_widget)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(4)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)          # saisie libre aussi possible
        self.model_combo.addItem("qwen3:8b")        # default Ollama
        self.refresh_btn = QPushButton("\U0001f504")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.setToolTip("Actualiser la liste des mod\u00e8les disponibles")
        self.refresh_btn.clicked.connect(self._fetch_local_models)
        local_layout.addWidget(self.model_combo)
        local_layout.addWidget(self.refresh_btn)
        self.model_stack.addWidget(local_widget)    # index 0

        # Page 1 : QLineEdit pour cloud
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("gpt-3.5-turbo")
        self.model_stack.addWidget(self.model_input)  # index 1

        self.model_stack.setCurrentIndex(0)           # Ollama par d\u00e9faut
        cfg_layout.addRow("Mod\u00e8le :", self.model_stack)

        self.apikey_label = QLabel("Cl\u00e9 API :")
        self.apikey_input = QLineEdit()
        self.apikey_input.setEchoMode(QLineEdit.EchoMode.Password)
        cfg_layout.addRow(self.apikey_label, self.apikey_input)
        self.apikey_label.setVisible(False)
        self.apikey_input.setVisible(False)

        cfg_group.setLayout(cfg_layout)
        top_row.addWidget(cfg_group, stretch=1)

        # ---- S\u00e9curit\u00e9 group
        sec_group  = QGroupBox("\U0001f6e1\ufe0f Mode S\u00e9curit\u00e9")
        sec_layout = QVBoxLayout()
        sec_layout.setSpacing(6)

        self.security_badge = QLabel()
        self.security_badge.setFixedHeight(28)
        sec_layout.addWidget(self.security_badge)

        self.trust_label = QLabel()
        self.trust_label.setFixedHeight(22)
        self.trust_label.setVisible(False)
        sec_layout.addWidget(self.trust_label)

        self.radio_monitoring = QRadioButton(
            "\U0001f534  MONITORING   \u2014 aucune action, r\u00e9ponse directe seulement"
        )
        self.radio_limited    = QRadioButton(
            "\U0001f7e1  LIMITED SCOPE \u2014 acc\u00e8s limit\u00e9 \u00e0 un dossier"
        )
        self.radio_full       = QRadioButton(
            "\U0001f7e2  FULL CONTROL  \u2014 acc\u00e8s complet syst\u00e8me"
        )
        self.radio_monitoring.setChecked(True)

        self.sec_btn_group = QButtonGroup()
        for r in (self.radio_monitoring, self.radio_limited, self.radio_full):
            self.sec_btn_group.addButton(r)
            sec_layout.addWidget(r)
            r.toggled.connect(self._on_security_changed)

        scope_row = QHBoxLayout()
        self.scope_label  = QLabel("Dossier scope :")
        self.scope_input  = QLineEdit()
        self.scope_input.setPlaceholderText("C:/mon/dossier")
        self.scope_browse = QPushButton("\U0001f4c2")
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

        self._on_security_changed()
        self.scope_label.setVisible(False)
        self.scope_input.setVisible(False)
        self.scope_browse.setVisible(False)

        # ── Chat
        root.addWidget(QLabel("Conversation & \u00c9tapes ReAct :"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color:#1e1e1e; color:#cccccc;"
            "font-family:'Consolas',monospace; font-size:13px;"
        )
        self._append_colored(
            "Bienvenue dans l'Optimisation AI Agent.\n"
            "S\u00e9lectionnez un provider et un mode s\u00e9curit\u00e9, puis lancez votre requ\u00eate.\n"
            "Cliquez \U0001f504 pour charger les mod\u00e8les disponibles du provider local.",
            "#888888"
        )
        root.addWidget(self.log_output)

        bar = QHBoxLayout()
        self.prompt_input  = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requ\u00eate ici...")
        self.submit_button = QPushButton("\u25b6  Ex\u00e9cuter Agent")
        self.submit_button.setFixedHeight(36)
        self.submit_button.clicked.connect(self.start_agent_run)
        self.prompt_input.returnPressed.connect(self.start_agent_run)
        bar.addWidget(self.prompt_input)
        bar.addWidget(self.submit_button)
        root.addLayout(bar)

        self.setCentralWidget(central)

    # ──────────────────────────────────────────────── chat helpers
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
                self._append_colored("\n\U0001f914 Pens\u00e9e : ", "#888888")
                return
            self._append_colored(data, color)
            return

        self._thinking_started = False
        self._append_colored(pattern.format(data=data), color)

    # ──────────────────────────────────────────────── run agent
    def start_agent_run(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Entrez une requ\u00eate.")
            return

        mode  = self._current_security_mode()
        scope = self.scope_input.text().strip() if mode == SecurityMode.LIMITED_SCOPE else ""

        if mode == SecurityMode.FULL_CONTROL and not self._is_trust_active():
            dlg = TrustDialog(self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            duration = dlg.selected_duration
            if duration is not None:
                self._trust_expiry = datetime.now() + duration
                self._update_trust_badge()
            else:
                self._trust_expiry = None

        provider      = self._build_provider()
        provider_name = self.provider_combo.currentText()
        # Nom du mod\u00e8le affich\u00e9 selon type
        choice = self.provider_combo.currentText()
        if choice in LOCAL_PROVIDERS:
            model_name = self.model_combo.currentText().strip()
        else:
            model_name = self.model_input.text().strip() or self.model_input.placeholderText()
        badge_label = SECURITY_BADGE[mode][0]

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self._thinking_started = False

        self._append_colored(
            f"\n\n\U0001f4ac Vous : {prompt}\n"
            f"\U0001f50c {provider_name} | {model_name}\n"
            f"\U0001f6e1\ufe0f  {badge_label}{' | scope: ' + scope if scope else ''}\n",
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
            self._append_colored("\n" + "\u2550" * 60 + "\n", "#444444")
        else:
            self._append_colored(f"\n\u274c \u00c9chec : {error}\n", "#ef5350")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
