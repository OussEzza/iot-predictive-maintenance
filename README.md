# Maintenance Prédictive de Capteurs IoT (Intel Berkeley Lab)

**Projet de Data Mining — Master IAII (Intelligence Artificielle et Informatique Industrielle)**  
**Auteurs :** EZZAHRI Oussama & EZZHAR Mohammed  
**Encadrante :** Pr. BOUZAACHANE Khadija  

---

## 1. Description du projet

Ce projet porte sur la **maintenance prédictive** appliquée aux réseaux de capteurs sans fil industriels et environnementaux. L'étude exploite le jeu de données historique du laboratoire **Intel Berkeley Research Lab** (2004), composé de 54 nœuds de capteurs *Mica2Dot* mesurant la température, l'humidité relative, la luminosité et la tension d'alimentation.

L'objectif méthodologique est d'anticiper la défaillance matérielle d'un nœud capteur avec un horizon de prédiction fixé à **24 heures** ($H=24\text{ h}$), dans un cadre d'évaluation strict sans fuite temporelle ou transversale.

### Faits saillants et conclusions scientifiques :
- **Cause physique de panne :** Épuisement progressif de la pile bouton d'alimentation.
- **Protocole d'évaluation :** Séparation stricte au niveau nœud en 41 nœuds de développement (évaluation par validation croisée *leave-nodes-out* / prédictions hors-échantillon OOF) et 10 nœuds de test verrouillés réservés à l'inférence finale (3 nœuds sans données exploitables : 5, 15, 28).
- **Reproductibilité :** Graine aléatoire fixée à **42** (`SEED = 42`) sur l'ensemble de la chaîne de traitement, des partitions et des modèles.
- **Résultats contrôlés :**
  - Nœuds de développement : **38 / 41 détectés (92,7 %)**, délai d'alerte médian de **65,6 h**, **0,019** fausse alerte par nœud-jour. Nœuds manqués : 16, 20 (silence), 50 (silence).
  - Nœuds de test verrouillés : **9 / 10 détectés (90,0 %)**, délai d'alerte médian de **72,7 h**, **0,015** fausse alerte par nœud-jour. Nœud manqué : 8 (silence).
  - Niveau ligne (minute) sur test : **PR-AUC 0,686**, **ROC-AUC 0,987**, taux de base 6,0 %.
- **Modèle sélectionné :** Régression Logistique avec standardisation (`StandardScaler`) et pondération des classes (`class_weight='balanced'`).

---

## 2. Structure du projet

```text
.
├── intel_lab/                      # Carnets Jupyter de l'étude Intel Berkeley
│   ├── 01_data_understanding.ipynb # Exploration, nettoyage et définition formelle des pannes
│   ├── 02_features_and_dataset.ipynb# Construction de la grille 10 min et des 48 features temporelles
│   ├── 03_modeling_intel.ipynb     # Entraînement, comparaison (LR, DT, RF, XGB) et optimisation CV
│   └── 04_evaluation_intel.ipynb   # Évaluation stricte au niveau nœud (délai, budget fausses alertes)
├── metropt3/                       # Étude préliminaire MetroPT-3 (Air Compressor Dataset)
│   └── README.md                   # Carnets d'étude préliminaire (absents du dépôt)
├── demo/                           # Démonstrateur scientifique et API de service
│   ├── app_ui.py                   # Interface interactive Streamlit (Modes A et B, Flotte, Limites)
│   ├── app_api.py                  # API d'inférence REST FastAPI
│   ├── run_dashboard.py            # Script lanceur conjoint (FastAPI + Streamlit)
│   └── .streamlit/                 # Configuration Streamlit du démonstrateur
├── models/                         # Modèles finaux sérialisés
│   └── best_model_intel.joblib     # Pipeline scikit-learn standardisé final
├── reports/
│   ├── figures/                    # 14 graphiques haute résolution générés par l'étude
│   └── results/                    # Résultats tabulés JSON, CSV et prédictions OOF
├── data/
│   ├── raw/                        # Répertoire pour les données brutes (non versionnées)
│   │   └── .gitkeep
│   └── processed/                  # Répertoire pour les données traitées (non versionnées)
│       └── .gitkeep
├── requirements.txt                # Dépendances Python avec versions exactes
├── LICENSE                         # Licence MIT
└── README.md                       # Présente documentation
```

---

## 3. Données brutes et téléchargement

Les fichiers de données brutes volumineux ne sont pas versionnés dans le dépôt Git.

### Liens de téléchargement :
1. **Intel Berkeley Research Lab :**
   - **URL officielle :** [https://db.csail.mit.edu/labdata/labdata.html](https://db.csail.mit.edu/labdata/labdata.html)
   - **Fichier :** `data.txt.gz` (archive compressée d'environ 33 Mo, contenant 2,3 millions de lectures).
   - **Emplacement requis :** Déposer le fichier directement dans :
     ```text
     data/raw/data.txt.gz
     ```
2. **MetroPT-3 (Étude préliminaire) :**
   - **URL officielle :** [UCI Machine Learning Repository — MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)

---

## 4. Ordre d'exécution des carnets (Notebooks)

Pour reproduire l'intégralité du pipeline depuis les données brutes jusqu'aux résultats finaux, exécuter les carnets du dossier `intel_lab/` dans l'ordre suivant :

1. **`intel_lab/01_data_understanding.ipynb`** :
   - Lecture de `data/raw/data.txt.gz`, nettoyage des horodatages et valeurs aberrantes.
   - Formalisation de la condition de panne (au moins 50 % de lectures hors normes sur 60 mesures consécutives ou arrêt d'émission).
   - Génère `data/processed/intel_clean.parquet` et `data/processed/failure_definition.json`.

2. **`intel_lab/02_features_and_dataset.ipynb`** :
   - Ré-échantillonnage sur une grille temporelle régulière de 10 minutes.
   - Ingénierie des 48 descripteurs temporels (moyennes, minima, maxima, écarts-types, variations sur fenêtres de 1 h, 6 h, 24 h et 48 h).
   - Détermination des instants de panne par capteur (`mote_onsets.csv`).
   - Génère `data/processed/features_intel.parquet` et les métadonnées de variables.

3. **`intel_lab/03_modeling_intel.ipynb`** :
   - Partitionnement déterministe avec graine fixe `42` (`split_nodes.json` : 41 dev / 10 test).
   - Validation croisée groupée par nœud (*GroupKFold*) sur les nœuds de développement.
   - Benchmark comparatif (Régression Logistique, Arbre de décision, Forêt aléatoire, XGBoost).
   - Sélection du modèle final et optimisation des seuils de décision sous contrainte de fausses alertes.
   - Sauvegarde du modèle dans `models/best_model_intel.joblib` et des prédictions OOF dans `reports/results/`.

4. **`intel_lab/04_evaluation_intel.ipynb`** :
   - Déverrouillage unique et évaluation sur les 10 nœuds de test réservés.
   - Calcul des délais d'alerte, des taux d'anticipation et analyse des cas limites (pannes silencieuses).
   - Analyse d'importance des variables par permutation et export des bilans finaux (`reports/results/intel_final_results.json`).

> **Note de reproductibilité :** Tous les calculs s'exécutent avec `SEED = 42`.

---

## 5. Démarrage de la démonstration

### Prérequis d'installation :
```bash
pip install -r requirements.txt
```

### Lancement de l'interface Streamlit :
Depuis la racine du projet :
```bash
streamlit run demo/app_ui.py
```
L'application s'ouvre automatiquement dans le navigateur à l'adresse : [http://localhost:8501](http://localhost:8501).

### Lancement du service API FastAPI (optionnel) :
```bash
uvicorn demo.app_api:app --host 127.0.0.1 --port 8000
```
Documentation interactive OpenAPI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### Lancement conjoint (Backend + Frontend) :
```bash
python demo/run_dashboard.py
```

---

## 6. Fonctionnalités du démonstrateur

- **Replay historique (Mode B) :** Rejoue la chronologie exacte de chaque capteur pas à pas sans fuite d'information future. L'évaluation rétrospective (statut de détection, délai d'alerte, ligne de panne et fenêtre de 24 h) est masquée par défaut et activable par case à cocher.
- **Simulation manuelle (Mode A) :** Permet de modifier localement des variables physiques (tension, température, humidité) à partir d'un vecteur d'observation réel, avec notice d'avertissement explicite sur le caractère hypothétique du scénario.
- **Tableau de la flotte :** Synthèse exhaustive des performances sur les 54 nœuds (41 développement, 10 test, 3 sans données), métriques agrégées et taux de fausses alertes par nœud-jour.
- **Limites de l'étude :** Analyse critique transparente des hypothèses méthodologiques (taille d'échantillon test, pannes silencieuses, comparaison avec la règle heuristique de sous-tension).

---

## 7. Licence

Ce projet est sous licence MIT. Consulter le fichier [LICENSE](LICENSE) pour plus de détails.
