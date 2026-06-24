# 🛠️ Guide de Configuration et Prérequis

Ce guide détaille tout ce qui est nécessaire pour faire tourner l'Optimisation AI Agent. Suivre ces étapes dans l'ordre est **CRITIQUE** pour que l'application démarre correctement.

## 1. Environnement Python
Nous recommandons d'utiliser un environnement virtuel (`venv`).

```powershell
# Créer le venv et activer (sur PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
```

## 2. Dépendances du Projet
Installez toutes les dépendances nécessaires en utilisant `pip`. Les librairies sont regroupées par leur fonction pour faciliter la maintenance.

**Installation des Core Dependencies :**
```powershell
# Composants de base (GUI, JSON handling)
pip install PyQt6 requests
```

**Dépendances LLM Providers (Selon votre usage):**
*   **Local Ollama:** Pas d'installation pip requise si le service est déjà lancé.
*   **LM Studio/OpenAI (Cloud):** Nécessite `httpx` pour les appels API externes, même si nous utilisons `requests` en interne dans l'implémentation actuelle.

```powershell
pip install requests PyQt6
```

**Dépendances Web Automation:**
Si vous prévoyez d'utiliser la navigation web complète (Playwright/Selenium), installez les outils nécessaires **avant** de lancer le premier test :
```powershell
# Installer les binaires Playwright requis par l'application.
# REMARQUE: Cette commande doit être exécutée dans un environnement propre et avec droits élevés.
pip install playwright
playwright install
```

## 💡 Étapes Cruciales Avant Lancement
1.  **Services LLM:** Assurez-vous que votre service Ollama ou LM Studio est **en cours d'exécution** sur `http://localhost:11434` (ou autre) avant de lancer le GUI.
2.  **Clés API :** Si vous utilisez l'API OpenAI, configurez une variable d'environnement (`OPENAI_API_KEY`) au niveau du système pour que le fournisseur Cloud puisse s'y connecter.

---
*Ce fichier doit être mis à jour à chaque fois qu'une nouvelle dépendance est ajoutée.*