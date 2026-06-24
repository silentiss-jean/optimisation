# 🚀 Guide d'Utilisation de l'Agent Optimisation

Ce guide explique comment interagir avec l'Optimisation AI Agent, en gardant toujours la sécurité et le contexte en tête.

## 1. Lancement du Programme
Lancez l'application via `python main_window.py` dans votre environnement virtuel activé. Une fenêtre graphique de gestion des tâches apparaîtra.

## 2. Le Cycle d'Interaction (Le Flux ReAct)
L'interaction se fait en trois étapes :
1.  **Entrée Utilisateur:** Vous entrez une requête générale dans le champ et cliquez sur "Exécuter Agent".
2.  **Observation de l'Agent:** L'agent raisonne en arrière-plan. Regardez la zone de log pour suivre sa pensée (`[Pensée]:...`) lorsqu'il décide d'utiliser un outil (ex: `find_files`).
3.  **Interaction Manuelle Requise:** Si le système est en mode **MONITORING**, il *bloquera* l'exécution et vous demandera explicitement de changer le mode de sécurité avant de procéder.

## 3. Gestion du Niveau de Sécurité (CRITIQUE)
L'état de sécurité détermine si l'Agent peut agir. Vous DEVEZ contrôler cet état via les options disponibles dans la GUI (ou en modifiant le code).

| Mode | Objectif Principal | Autorisations / Actions Permises | Quand l'utiliser? |
| :--- | :--- | :--- | :--- |
| **🔵 MONITORING (Par Défaut)** | Audit et Planification. | Aucune exécution réelle. Seule la *simulation* des actions est possible. | Lors de la découverte du système, ou si vous avez un doute sur le comportement attendu. |
| **🟡 LIMITED_SCOPE** | Tâches ciblées et contrôlées. | Permet d'interagir avec les outils, mais seulement pour les chemins spécifiés (`G:\optimisation\`). | Quand on sait que l'action doit rester dans une zone de travail précise (recommandé). |
| **🔴 FULL_CONTROL** | Test en conditions réelles. | Toutes les commandes et accès fichiers sont autorisés. **UTILISATION EXTRÊMEMENT DÉCONSEILLÉE.** | UNIQUEMENT lors des tests de validation finaux après revue de code approfondie. |

## 4. Workflow d'Exemple : Recherche & Analyse
1.  Démarrez en mode `MONITORING`. Le système vous dira qu'il bloque les actions (Attendu).
2.  Changez le mode à `LIMITED_SCOPE` avec le scope `G:\optimisation`.
3.  Entrez une requête : "Liste-moi tous les fichiers JSON et dis-moi ce que cela signifie."
4.  L'agent exécutera `find_files(...)`, puis pourra lire chaque fichier trouvé séquentiellement en respectant le mode de sécurité.

---
*N'oubliez pas de consulter également [README.md](#) pour la structure du projet.*