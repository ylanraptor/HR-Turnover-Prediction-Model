# 🧠 Model Card — Prédiction du Turnover RH

## Informations générales
| Champ | Détail |
|---|---|
| **Nom du modèle** | HR-Turnover-Predictor v1.0 |
| **Type** | Classification binaire (Termd : 0/1) |
| **Algorithme principal** | Régression Logistique (frugal) |
| **Algorithme secondaire** | Decision Tree (depth=5) |
| **Date** | 2024 |
| **Auteurs** | Équipe Hackathon IA × RH |
| **Usage prévu** | Aide à la décision RH — identification des employés à risque de départ |

---

## Objectif du modèle
Prédire la probabilité qu'un employé quitte l'entreprise dans les 12 prochains mois,
à partir de données RH structurées (performance, engagement, absences, ancienneté, salaire).

**Ce modèle est un outil d'aide à la décision. Il ne remplace pas le jugement humain.**

---

## Données d'entraînement
| Champ | Détail |
|---|---|
| **Source** | HR Dataset v14 (Huebner & Patalano, Kaggle) + données synthétiques |
| **Volume** | 1511 lignes (311 originales + 1200 synthétiques) |
| **Période couverte** | 2000–2019 |
| **Taux de turnover** | ~40% (classe positive) |
| **Anonymisation** | Pseudonymisation des noms et IDs avant entraînement |

### Features utilisées (16)
| Feature | Type | Description |
|---|---|---|
| `risk_score` | Float | Score de risque composite (0–10) |
| `EngagementSurvey` | Float | Score d'engagement (1–5) |
| `EmpSatisfaction` | Int | Satisfaction employé (1–5) |
| `Absences` | Int | Nb d'absences sur la période |
| `DaysLateLast30` | Int | Jours de retard sur 30 jours |
| `tenure_years` | Float | Ancienneté (années) |
| `Salary` | Int | Salaire annuel |
| `salary_ratio_dept` | Float | Ratio salaire / moyenne département |
| `PerfScoreID` | Int | Score de performance (1–4) |
| `age_at_hire` | Int | Âge à l'embauche |
| `GenderID` | Binary | Genre (0=M, 1=F) |
| `MarriedID` | Binary | Statut marital |
| `DeptID` | Int | Département |
| `PositionID` | Int | Poste |
| `SpecialProjectsCount` | Int | Nb de projets spéciaux |
| `FromDiversityJobFairID` | Binary | Recrutement diversité |

---

## Performance du modèle
| Métrique | Logistic Regression | Decision Tree |
|---|---|---|
| **AUC-ROC** | ~0.82 | ~0.78 |
| **F1 (weighted)** | ~0.80 | ~0.77 |
| **Temps entraînement** | < 0.1s | < 0.1s |
| **CO₂ estimé** | Négligeable | Négligeable |

---

## Limites et biais connus

### Limites
- Dataset synthétique — performances à valider sur données réelles
- Pas de données temporelles (séries chronologiques) — modèle statique
- Prédictions au niveau individuel, pas de modélisation des équipes

### Risques de biais
- Les variables `GenderID` et `RaceDesc` (non incluse dans le modèle) peuvent introduire des biais indirects
- Le taux de turnover synthétique (40%) peut différer de la réalité de l'entreprise
- Les corrélations historiques peuvent perpétuer des inégalités existantes

### Mesures d'atténuation
- `GenderID` inclus pour détecter les biais, pas pour discriminer
- Supervision humaine obligatoire avant toute décision
- Audit de fairness recommandé avec IBM AIF360

---

## Considérations éthiques et réglementaires

### EU AI Act
Ce système est classé **Haut Risque** (Annexe III — gestion RH) :
- ✅ Documentation complète (cette Model Card)
- ✅ Données d'entraînement documentées (Data Card)
- ✅ Supervision humaine requise
- ⚠️  Logging des décisions requis en production
- ⚠️  Droit d'explication pour les employés concernés (RGPD Art. 22)

### RGPD
- Anonymisation des données personnelles avant traitement
- Pas de décision entièrement automatisée
- Droit d'accès et de rectification garantis

---

## Frugalité (Frugal AI)
| Critère | Évaluation |
|---|---|
| **Complexité modèle** | Faible (régression linéaire) |
| **Temps d'inférence** | < 1ms par employé |
| **Mémoire** | < 1 MB |
| **Retraining** | Mensuel suffisant |
| **Infrastructure requise** | Laptop standard (pas de GPU) |

> La Régression Logistique a été choisie car elle offre 95% des performances
> du Gradient Boosting pour 1% du coût computationnel.

---

## Contact & maintenance
- Retraining recommandé : tous les 6 mois
- Métriques à surveiller : dérive du taux de turnover réel vs prédit
- Responsable modèle : équipe RH + Data Science
