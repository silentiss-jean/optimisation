# Optimisation AI Agent

Agent IA autonome avec interface PyQt5, support multi-providers LLM et outils navigateur/système.

---

## Architecture

```
optimisation/
├── main.py                        # Point d'entrée PyQt5
├── providers/
│   ├── llm_provider_interface.py   # Interface commune LLM
│   ├── ollama_provider.py          # Ollama (local)
│   └── openai_provider.py          # OpenAI / API compatible
├── tools/
│   ├── base_tool.py               # Classe de base Tool
│   ├── tool_dispatcher.py         # Registre + dispatch des outils
│   ├── agent_orchestrator.py      # Boucle agentique ReAct
│   ├── filesystem_tool.py         # read_file, write_file, find_files
│   ├── command_line_tool.py       # command_line_execute
│   ├── open_url_tool.py           # open_url (webbrowser système)
│   ├── web_scraper_tool.py        # web_scrape (requests + BeautifulSoup)
│   └── playwright_browser_tool.py # Contrôle navigateur Playwright
└── backlog.json                   # Tâches restantes classées par priorité
```

---

## Outils disponibles

### Fichiers
| Outil | Description |
|---|---|
| `read_file(file_path)` | Lire un fichier texte |
| `write_file(file_path, content)` | Écrire dans un fichier |
| `find_files(directory, pattern)` | Chercher des fichiers par pattern |

### Système
| Outil | Description |
|---|---|
| `command_line_execute(command)` | Exécuter une commande shell |

### Navigateur
| Outil | Description |
|---|---|
| `open_url(url)` | Ouvrir une URL via le navigateur système (fallback) |
| `browser_navigate(url)` | Naviguer dans l'onglet actif (Playwright) |
| `browser_new_tab(url)` | Ouvrir un nouvel onglet dans la même fenêtre |
| `browser_click(selector)` | Cliquer sur un élément CSS |
| `browser_fill(selector, text)` | Remplir un champ de saisie |
| `browser_get_text(selector?)` | Lire le texte visible de la page ou d'un élément |
| `browser_scroll(direction, amount)` | Scroller la page |
| `browser_wait_for(selector, timeout)` | Attendre qu'un élément soit visible |
| `browser_screenshot(save_path?)` | Capturer la page |

### Web
| Outil | Description |
|---|---|
| `web_scrape(url)` | Récupérer le texte d'une page sans navigateur |

---

## Modes de sécurité

| Mode | Outils autorisés |
|---|---|
| `MONITORING` | Aucun — lecture seule, réponse directe |
| `LIMITED_SCOPE` | Fichiers uniquement (dans un scope de dossier) |
| `FULL_CONTROL` | Tous les outils |

---

## Session Playwright

La session Playwright est liée au **thread courant** (`threading.get_ident()`).  
Elle est automatiquement recréée si :
- Le thread change (nouvelle conversation PyQt)
- La fenêtre a été fermée manuellement (`_is_alive()` détecte le crash)

Un `BrowserContext` partagé permet d'ouvrir plusieurs onglets dans la même fenêtre.

---

## Problèmes connus

### Double-JSON (qwen3:8b)
Le modèle envoie parfois 2 objets JSON sur la même ligne quand la tâche demande 2 actions.  
Le parser détecte le double et exécute seulement le premier.  
Mais `DONE_TRIGGERS` arrête le cycle dès le 1er `[DONE]` — la 2e action n'est jamais exécutée.  
**Fix prévu** : voir `backlog.json` → priorité 1.

### Onglet vs fenêtre
Si deux conversations utilisent Playwright dans deux threads différents, chacune ouvre sa propre fenêtre.  
Dans la même conversation, `browser_new_tab` ouvre bien un onglet dans la fenêtre existante.

---

## Installation

```bash
pip install PyQt5 playwright requests beautifulsoup4
playwright install chromium
```

## Lancement

```bash
python main.py
```
