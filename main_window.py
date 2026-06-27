import sys
import os
import json
import re
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QMessageBox,
    QComboBox, QGroupBox, QFormLayout, QRadioButton, QButtonGroup,
    QFileDialog, QDialog, QDialogButtonBox, QStackedWidget,
    QListWidget, QListWidgetItem, QSplitter, QProgressBar
)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QPixmap, QTextImageFormat

from providers.llm_provider_interface import LLMProvider
from providers.ollama_provider import OllamaProvider
from providers.lmstudio_provider import LMStudioProvider
from providers.openai_provider import OpenAIProvider
from providers.perplexity_provider import PerplexityProvider
from tools.tool_dispatcher import ToolDispatcher
from tools.memory_store import MemoryStore
from tools.agent_orchestrator import (
    AgentOrchestrator, SecurityMode,
    EVT_THINKING, EVT_ACTION, EVT_OBSERVE, EVT_ANSWER,
    EVT_STEP, EVT_WARNING, EVT_SECURITY, EVT_SCREENSHOT
)

# ─────────────────────────────────────────────────────────────────── providers
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
    "Perplexity (Sonar)": "pplx-... → console.perplexity.ai",
}

MODEL_LIST_ENDPOINTS = {
    "Ollama (local)": ("http://localhost:11434/api/tags",
                       lambda d: [m["name"] for m in d.get("models", [])]),
    "LM Studio (local)": ("http://localhost:1234/v1/models",
                           lambda d: [m["id"] for m in d.get("data", [])]),
}

START_COMMANDS = {
    "Ollama (local)": "ollama serve",
    "LM Studio (local)": "Lancez LM Studio et activez \"Local Server\" dans l'interface.",
}

REACT_INCOMPATIBLE_PATTERNS = [
    "llama3.1", "llama3:8b", "llama3.1:8b", "llama3.1:70b",
    "llama2", "mistral:7b", "mistral:latest",
]

SCREENSHOT_MAX_WIDTH = 780

# ────────────────────────────────────────────────────── streaming colors
EVENT_COLORS = {
    EVT_STEP:       "#888888",
    EVT_THINKING:   "#cccccc",
    EVT_ACTION:     "#4fc3f7",
    EVT_OBSERVE:    "#81c784",
    EVT_ANSWER:     "#fff176",
    EVT_WARNING:    "#ff8a65",
    EVT_SECURITY:   "#ef5350",
    EVT_SCREENSHOT: "#b39ddb",
}
EVENT_PREFIX = {
    EVT_STEP:     "\n\u254d\u254d\u254d {data} \u254d\u254d\u254d",
    EVT_THINKING: "{data}",
    EVT_ACTION:   "\n\U0001f527 Outil : {data}",
    EVT_OBSERVE:  "\n\U0001f4cd Observation : {data}",
    EVT_ANSWER:   "\n\u2705 Réponse finale :\n{data}",
    EVT_WARNING:  "\n\u26a0\ufe0f {data}",
    EVT_SECURITY: "\n\U0001f6d1 {data}",
}

SECURITY_BADGE = {
    SecurityMode.MONITORING:    ("MONITORING \u2014 lecture seule",    "#1e1e1e", "#888888"),
    SecurityMode.LIMITED_SCOPE: ("LIMITED SCOPE \u2014 dossier limité", "#1a2a1a", "#81c784"),
    SecurityMode.FULL_CONTROL:  ("FULL CONTROL \u2014 accès total",    "#2a1a1a", "#ef5350"),
}

TRUST_DURATIONS = [
    ("Cette requête uniquement",   None),
    ("1 heure",                    timedelta(hours=1)),
    ("Jusqu'à demain minuit",      None),
    ("Cette session",              timedelta(days=3650)),
]


def _local_server_error_msg(provider_name: str, url: str, exc: Exception) -> str:
    cmd = START_COMMANDS.get(provider_name, f"Démarrez {provider_name}.")
    reason = str(exc)
    if "Connection refused" in reason or "111" in reason:
        detail = "Le serveur n'est pas démarré (connexion refusée)."
    elif "timed out" in reason.lower():
        detail = "Le serveur ne répond pas (timeout 3 s)."
    else:
        detail = f"Erreur réseau : {reason}"
    return (
        f"\u26a0\ufe0f  Impossible de joindre {provider_name}\n"
        f"URL : {url}\n\n"
        f"{detail}\n\n"
        f"Pour démarrer {provider_name} :\n"
        f"  {cmd}\n\n"
        f"Le modèle par défaut reste sélectionné \u2014 vous pouvez continuer à le taper manuellement."
    )


def _is_react_incompatible(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(pattern in name_lower for pattern in REACT_INCOMPATIBLE_PATTERNS)


# ═══════════════════════════════════════════════════ Dialog de confiance
class TrustDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("\U0001f7e2 FULL CONTROL \u2014 Confirmation")
        self.setMinimumWidth(420)
        self.selected_duration = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "<b>L'agent aura accès TOTAL au système</b><br>"
            "(fichiers, commandes OS, web).<br><br>"
            "Pendant combien de temps accorder l'accès ?"
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
        if label == "Jusqu'à demain minuit":
            tomorrow = datetime.now().replace(
                hour=23, minute=59, second=59, microsecond=0
            ) + timedelta(days=1)
            self.selected_duration = tomorrow - datetime.now()
        else:
            self.selected_duration = delta
        self.accept()


# ═══════════════════════════════════════════════════════════════════ Worker
class LLMWorker(QThread):
    event_signal    = pyqtSignal(str, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, prompt: str, provider: LLMProvider,
                 dispatcher: ToolDispatcher,
                 security_mode: str, scope: str,
                 provider_name: str = "",
                 memory_store: MemoryStore | None = None,
                 chat_history: list | None = None):
        super().__init__()
        self.prompt        = prompt
        self.provider      = provider
        self.dispatcher    = dispatcher
        self.security_mode = security_mode
        self.scope         = scope or None
        self.provider_name = provider_name
        self.memory_store  = memory_store
        self.chat_history  = list(chat_history) if chat_history else []

    def run(self):
        try:
            orc = AgentOrchestrator(
                llm_provider=self.provider,
                dispatcher=self.dispatcher,
                on_event=lambda evt, data: self.event_signal.emit(evt, data),
                memory_store=self.memory_store,
            )
            orc.chat_history = self.chat_history
            orc.set_safety_mode(self.security_mode, self.scope)
            answer = orc.run_agentic_cycle(self.prompt)
            self.finished_signal.emit(bool(answer), "")
        except Exception as e:
            err = str(e)
            if self.provider_name in LOCAL_PROVIDERS and (
                "Connection refused" in err or "111" in err or "timed out" in err.lower()
                or "ConnectionRefusedError" in err or "RemoteDisconnected" in err
            ):
                cmd = START_COMMANDS.get(self.provider_name, "")
                friendly = (
                    f"{self.provider_name} ne répond pas.\n"
                    f"Démarrez-le d'abord :\n  {cmd}"
                )
                self.event_signal.emit(EVT_WARNING, friendly)
                self.finished_signal.emit(False, friendly)
            else:
                self.event_signal.emit(EVT_WARNING, f"EXCEPTION : {err}")
                self.finished_signal.emit(False, err)


# ═══════════════════════════════════════════════════════════════════ MainWindow
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Optimisation AI Agent")
        self.setGeometry(100, 100, 1260, 900)
        self.dispatcher        = ToolDispatcher()
        self._thinking_started = False
        self._trust_expiry: datetime | None = None
        self.memory_store      = MemoryStore()
        self._setup_ui()

    # ────────────────────────────────── fetch models from local server
    def _fetch_local_models(self):
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
                QMessageBox.information(self, "Modèles",
                    "Aucun modèle trouvé.\nTéléchargez un modèle, ex : ollama pull qwen3:8b")
                return
            current = self.model_combo.currentText()
            self.model_combo.clear()
            self.model_combo.addItems(models)
            if current in models:
                self.model_combo.setCurrentText(current)
        except Exception as e:
            msg = _local_server_error_msg(choice, url, e)
            QMessageBox.warning(self, f"{choice} injoignable", msg)

    # ────────────────────────────────── build provider
    def _build_provider(self) -> LLMProvider:
        choice = self.provider_combo.currentText()
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
        is_local = choice in LOCAL_PROVIDERS
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
            self.model_combo.clear()
            self.model_combo.addItem(PROVIDER_DEFAULTS.get(choice, ""))

    def _on_model_changed(self, _index: int):
        model = self.model_combo.currentText().strip()
        if model and _is_react_incompatible(model):
            self._append_colored(
                f"\n\u26a0\ufe0f  {model} est peu compatible avec l'architecture ReAct de cet agent.\n"
                "   Les résultats peuvent être imprévisibles. Préférez qwen3:8b ou phi4-mini.\n",
                "#ff8a65"
            )

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
            self.trust_label.setText(f"  \u2705 Session de confiance active jusqu'à {expiry_str}  ")
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

    # ─────────────────────────────────────────────────────── memory helpers
    def _clear_memory(self):
        reply = QMessageBox.question(
            self, "Effacer la mémoire",
            "Effacer toute la mémoire (historique de session + faits longue durée) ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.memory_store.clear_all()
            self.lt_list.clear()
            self._append_colored("\n🗑️  Mémoire effacée.\n", "#888888")

    def _add_long_term_fact(self):
        fact = self.lt_input.text().strip()
        if not fact:
            return
        self.memory_store.add_long_term(fact)
        self.lt_input.clear()
        self._refresh_lt_list()
        self._append_colored(f"\n🧠 Fait mémorisé : {fact}\n", "#b39ddb")

    def _remove_long_term_fact(self):
        item = self.lt_list.currentItem()
        if not item:
            return
        fact = item.text()
        self.memory_store.remove_long_term(fact)
        self._refresh_lt_list()
        self._append_colored(f"\n🗑️  Fait supprimé : {fact}\n", "#888888")

    def _refresh_lt_list(self):
        self.lt_list.clear()
        for fact in self.memory_store.long_term:
            self.lt_list.addItem(QListWidgetItem(fact))

    # ─────────────────────────────────────────────────────── progress helper
    def _set_progress(self, current: int, total: int, text: str = ""):
        total   = max(total, 1)
        current = max(0, min(current, total))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)
        self.progress_label.setText(text or f"Étape {current}/{total}")

    # ─────────────────────────────────────────────────────── setup UI
    def _setup_ui(self):
        central = QWidget()
        root    = QVBoxLayout(central)
        root.setSpacing(8)
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        cfg_group  = QGroupBox("Provider LLM")
        cfg_layout = QFormLayout()
        cfg_layout.setSpacing(6)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(list(PROVIDER_DEFAULTS.keys()))
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        cfg_layout.addRow("Provider :", self.provider_combo)

        self.model_stack = QStackedWidget()
        self.model_stack.setFixedHeight(32)

        local_widget = QWidget()
        local_layout = QHBoxLayout(local_widget)
        local_layout.setContentsMargins(0, 0, 0, 0)
        local_layout.setSpacing(4)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItem("qwen3:8b")
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.model_combo.lineEdit().editingFinished.connect(
            lambda: self._on_model_changed(self.model_combo.currentIndex())
        )
        self.refresh_btn = QPushButton("\U0001f504")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.setToolTip("Charger la liste des modèles disponibles (serveur local requis)")
        self.refresh_btn.clicked.connect(self._fetch_local_models)
        local_layout.addWidget(self.model_combo)
        local_layout.addWidget(self.refresh_btn)
        self.model_stack.addWidget(local_widget)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("gpt-3.5-turbo")
        self.model_stack.addWidget(self.model_input)

        self.model_stack.setCurrentIndex(0)
        cfg_layout.addRow("Modèle :", self.model_stack)

        self.apikey_label = QLabel("Clé API :")
        self.apikey_input = QLineEdit()
        self.apikey_input.setEchoMode(QLineEdit.EchoMode.Password)
        cfg_layout.addRow(self.apikey_label, self.apikey_input)
        self.apikey_label.setVisible(False)
        self.apikey_input.setVisible(False)

        cfg_group.setLayout(cfg_layout)
        top_row.addWidget(cfg_group, stretch=1)

        sec_group  = QGroupBox("\U0001f6e1\ufe0f Mode Sécurité")
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
            "\U0001f534  MONITORING   \u2014 aucune action, réponse directe seulement")
        self.radio_limited    = QRadioButton(
            "\U0001f7e1  LIMITED SCOPE \u2014 accès limité à un dossier")
        self.radio_full       = QRadioButton(
            "\U0001f7e2  FULL CONTROL  \u2014 accès complet système")
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

        # ══ Panneau mémoire longue durée ══════════════════════════════════════
        mem_group  = QGroupBox("🧠 Mémoire longue durée")
        mem_layout = QVBoxLayout()
        mem_layout.setSpacing(4)
        mem_path_label = QLabel(f"💾 {self.memory_store.path}")
        mem_path_label.setStyleSheet("color:#555555; font-size:10px;")
        mem_layout.addWidget(mem_path_label)
        self.lt_list = QListWidget()
        self.lt_list.setStyleSheet(
            "background:#1e1e1e; color:#cccccc; font-size:12px; border:1px solid #333;"
        )
        self.lt_list.setMaximumHeight(90)
        self.lt_list.setToolTip("Cliquez sur un fait pour le sélectionner, puis ➖ pour le supprimer")
        self._refresh_lt_list()
        mem_layout.addWidget(self.lt_list)
        lt_input_row = QHBoxLayout()
        lt_input_row.setSpacing(4)
        self.lt_input = QLineEdit()
        self.lt_input.setPlaceholderText("Nouveau fait… (ex: préfère qwen3:14b)")
        self.lt_input.setFixedHeight(28)
        self.lt_input.returnPressed.connect(self._add_long_term_fact)
        lt_add_btn = QPushButton("➕")
        lt_add_btn.setFixedSize(28, 28)
        lt_add_btn.setToolTip("Ajouter ce fait à la mémoire longue durée")
        lt_add_btn.clicked.connect(self._add_long_term_fact)
        lt_del_btn = QPushButton("➖")
        lt_del_btn.setFixedSize(28, 28)
        lt_del_btn.setToolTip("Supprimer le fait sélectionné")
        lt_del_btn.clicked.connect(self._remove_long_term_fact)
        self.clear_mem_btn = QPushButton("🗑️")
        self.clear_mem_btn.setFixedSize(28, 28)
        self.clear_mem_btn.setToolTip("Effacer TOUTE la mémoire (session + longue durée)")
        self.clear_mem_btn.clicked.connect(self._clear_memory)
        lt_input_row.addWidget(self.lt_input)
        lt_input_row.addWidget(lt_add_btn)
        lt_input_row.addWidget(lt_del_btn)
        lt_input_row.addWidget(self.clear_mem_btn)
        mem_layout.addLayout(lt_input_row)
        mem_group.setLayout(mem_layout)
        top_row.addWidget(mem_group, stretch=1)

        root.addLayout(top_row)
        self._on_security_changed()
        self.scope_label.setVisible(False)
        self.scope_input.setVisible(False)
        self.scope_browse.setVisible(False)

        # ── Titre log
        log_header = QHBoxLayout()
        log_header.addWidget(QLabel("Conversation & Étapes ReAct :"))
        root.addLayout(log_header)

        # ── Barre de progression (P3-003) ─────────────────────────────────────
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)
        progress_title = QLabel("Progression :")
        progress_title.setStyleSheet("color:#aaaaaa;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 8)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v/%m")
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background:#1e1e1e;
                border:1px solid #333333;
                border-radius:4px;
                color:#cccccc;
                height:18px;
            }
            QProgressBar::chunk {
                background-color:#4fc3f7;
                border-radius:3px;
            }
        """)
        self.progress_label = QLabel("En attente")
        self.progress_label.setStyleSheet("color:#888888; font-size:12px;")
        progress_row.addWidget(progress_title)
        progress_row.addWidget(self.progress_bar, 1)
        progress_row.addWidget(self.progress_label)
        root.addLayout(progress_row)
        # ─────────────────────────────────────────────────────────────────────

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet(
            "background-color:#1e1e1e; color:#cccccc;"
            "font-family:'Consolas',monospace; font-size:13px;"
        )
        hist_count = len(self.memory_store.session_history)
        lt_count   = len(self.memory_store.long_term)
        self._append_colored(
            "Bienvenue dans l'Optimisation AI Agent.\n"
            "Sélectionnez un provider et un mode sécurité, puis lancez votre requête.\n"
            "Cliquez \U0001f504 pour charger les modèles disponibles du provider local.\n",
            "#888888"
        )
        if hist_count or lt_count:
            self._append_colored(
                f"💾 Mémoire chargée : {hist_count} messages de session, "
                f"{lt_count} fait(s) longue durée.\n",
                "#4fc3f7"
            )
        root.addWidget(self.log_output)

        bar = QHBoxLayout()
        self.prompt_input  = QLineEdit()
        self.prompt_input.setPlaceholderText("Entrez votre requête ici...")
        self.submit_button = QPushButton("\u25b6  Exécuter Agent")
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

    def _insert_screenshot(self, img_path: str):
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self._append_colored(f"\n[screenshot non lisible : {img_path}]\n", "#ff8a65")
            return
        if pixmap.width() > SCREENSHOT_MAX_WIDTH:
            pixmap = pixmap.scaledToWidth(
                SCREENSHOT_MAX_WIDTH,
                Qt.TransformationMode.SmoothTransformation
            )
        doc = self.log_output.document()
        img_name = f"screenshot_{id(img_path)}"
        doc.addResource(
            doc.ResourceType.ImageResource,
            __import__('PyQt6.QtCore', fromlist=['QUrl']).QUrl(img_name),
            pixmap
        )
        cursor = self.log_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt_text = QTextCharFormat()
        fmt_text.setForeground(QColor(EVENT_COLORS[EVT_SCREENSHOT]))
        cursor.setCharFormat(fmt_text)
        cursor.insertText(f"\n\U0001f4f8 Screenshot ({pixmap.width()}x{pixmap.height()}px) :\n")
        img_fmt = QTextImageFormat()
        img_fmt.setName(img_name)
        img_fmt.setWidth(pixmap.width())
        img_fmt.setHeight(pixmap.height())
        cursor.insertImage(img_fmt)
        cursor.insertText("\n")
        self.log_output.setTextCursor(cursor)
        self.log_output.ensureCursorVisible()

    def _on_event(self, event_type: str, data: str):
        # ── Progression : parse "Étape N/M" depuis EVT_STEP ──────────────────
        if event_type == EVT_STEP:
            m = re.search(r"[EÉ]tape\s+(\d+)/(\d+)", data)
            if m:
                self._set_progress(int(m.group(1)), int(m.group(2)), data)
            else:
                # Pas de N/M → avance juste le label
                self.progress_label.setText(data[:60] if data else "…")
            # On laisse aussi l'affichage texte se faire (pas de return ici)

        if event_type == EVT_SCREENSHOT:
            self._insert_screenshot(data)
            return

        color   = EVENT_COLORS.get(event_type, "#cccccc")
        pattern = EVENT_PREFIX.get(event_type, "{data}")
        if event_type == EVT_THINKING:
            if not data:
                if self._thinking_started:
                    return
                self._thinking_started = True
                self._append_colored("\n\U0001f914 Pensée : ", "#888888")
                return
            self._append_colored(data, color)
            return
        self._thinking_started = False
        self._append_colored(pattern.format(data=data), color)

    # ──────────────────────────────────────────────── run agent
    def start_agent_run(self):
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Attention", "Entrez une requête.")
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
        choice = self.provider_combo.currentText()
        if choice in LOCAL_PROVIDERS:
            model_name = self.model_combo.currentText().strip()
        else:
            model_name = self.model_input.text().strip() or self.model_input.placeholderText()
        badge_label = SECURITY_BADGE[mode][0]

        self.submit_button.setEnabled(False)
        self.prompt_input.setEnabled(False)
        self._thinking_started = False
        self._set_progress(0, 8, "Démarrage…")          # ← P3-003
        self._append_colored(
            f"\n\n\U0001f4ac Vous : {prompt}\n"
            f"\U0001f50c {provider_name} | {model_name}\n"
            f"\U0001f6e1\ufe0f  {badge_label}{' | scope: ' + scope if scope else ''}\n",
            "#aaaaaa"
        )
        current_history = list(self.memory_store.session_history)
        self.worker = LLMWorker(
            prompt, provider, self.dispatcher, mode, scope,
            provider_name=provider_name,
            memory_store=self.memory_store,
            chat_history=current_history,
        )
        self.worker.event_signal.connect(self._on_event)
        self.worker.finished_signal.connect(self._run_finished)
        self.worker.start()

    def _run_finished(self, success: bool, error: str):
        # ── Finalise la barre de progression ─────────────────────────────────
        maximum = self.progress_bar.maximum() or 1
        self.progress_bar.setValue(maximum)
        self.progress_label.setText("✅ Terminé" if success else "❌ Échec")
        # ─────────────────────────────────────────────────────────────────────
        self.submit_button.setEnabled(True)
        self.prompt_input.setEnabled(True)
        self.prompt_input.clear()
        self._thinking_started = False
        if success:
            self._append_colored("\n" + "\u2550" * 60 + "\n", "#444444")
        else:
            self._append_colored(f"\n\u274c Échec : {error}\n", "#ef5350")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
