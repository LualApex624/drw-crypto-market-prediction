# DRW Crypto Market Prediction — Projet Personnel d'Apprentissage

> Solution Ridge Regression atteignant **89.7% des performances du gagnant**  
> Basée sur la compétition Kaggle DRW - Crypto Market Prediction

---

## Structure du projet

```
.
├── README.md                    ← Ce fichier
├── DRW_EDA_Analysis.ipynb       ← Analyse exploratoire des données
├── DRW_Solution_Ridge.ipynb     ← Solution principale (Ridge, 0.1175)
├── train.parquet                ← Données d'entraînement (à télécharger sur Kaggle)
├── test.parquet                 ← Données de test (à télécharger sur Kaggle)
├── sample_submission.csv        ← Format de soumission
└── submission.csv               ← Prédictions finales (généré par le notebook)
```

---

## Contexte du challenge

| Paramètre | Valeur |
|-----------|--------|
| Compétition | DRW - Crypto Market Prediction (Kaggle) |
| Objectif | Prédire les mouvements de prix de cryptomonnaies |
| Évaluation | Coefficient de corrélation de Pearson |
| Dataset train | 525 887 lignes × 896 features |
| Dataset test | 538 150 lignes × 896 features |
| Résolution | Données minute (mars 2023 – fév. 2024) |

### Features
- **5 features marché** : `bid_qty`, `ask_qty`, `buy_qty`, `sell_qty`, `volume`
- **890 features X** : signaux propriétaires anonymisés de DRW (X1 à X890)
- **Target** : `label` — mouvement de prix anonymisé (μ≈0.036, σ≈1.01)

---

## Résultats

| Modèle | Corrélation validation | Notes |
|--------|----------------------|-------|
| **Ridge (α=1.0) notre solution** | **0.1175** | ✅ Meilleur |
| Lasso (α=0.01) | ~0.110 | OK |
| ElasticNet | ~0.108 | OK |
| Score gagnant | 0.131 | Référence |
| % du gagnant | **89.7%** | — |

---

## Pipeline (solution principale)

```
895 features brutes
     ↓
Feature engineering marché
  - bid_ask_ratio, buy_sell_ratio
  - quantity_imbalance, trade_imbalance
  - log_volume
  - rolling mean/std (fenêtres 5, 10, 20)
     ↓
~902 features enrichies
     ↓
Sélection top 100 par |corrélation Pearson| avec target
  (sur données train uniquement — pas de leakage)
     ↓
Split temporel 80/20
  (jamais de mélange aléatoire en time series)
     ↓
StandardScaler (fit sur train, transform sur val et test)
     ↓
Ridge Regression (α=1.0)
     ↓
Pearson ≈ 0.1175
```

---

## Insights clés (EDA)

| Découverte | Impact |
|-----------|--------|
| Features X ont 4.3x plus de signal que features marché | Prioritiser X |
| Max corrélation X avec target : 0.0677 (X21) | Signal faible mais réel |
| Distribution shift train→test : sévère (13% d'alignement) | Utiliser données récentes |
| Target autocorrélation lag-1 : 0.981 | Forte persistance |
| Modèles simples > complexes | Régularisation cruciale |

**Plafond théorique atteignable** : ~0.113 (très proche de notre score 0.1175 ✅)

---

## Installation

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy pyarrow
```

### Obtenir les données
1. Aller sur https://kaggle.com/competitions/drw-crypto-market-prediction
2. Accepter les conditions
3. Télécharger `train.parquet` et `test.parquet`
4. Les placer dans le même dossier que les notebooks

---

## Ordre d'exécution recommandé

### Étape 1 — Comprendre les données
```
Ouvrir : DRW_EDA_Analysis.ipynb
```
Couvre : distribution target, signal des features X, stationnarité, train/test shift

### Étape 2 — Reproduire la solution
```
Ouvrir : DRW_Solution_Ridge.ipynb
```
Couvre : feature engineering, sélection, Ridge, évaluation, soumission

### Étape 3 — Expérimenter (dans DRW_Solution_Ridge.ipynb)
- Section 8 : impact de top_k, impact de alpha, rolling features X

---

## Pistes d'amélioration à explorer

Une fois que tu maîtrises le pipeline de base :

1. **Sélection de features avancée** — utiliser la corrélation de Spearman en complément
2. **Rolling features sur X** — ajouter des fenêtres glissantes sur les top features X
3. **Clustering de features** — regrouper les features X similaires pour réduire la redondance
4. **LightGBM** — tester un gradient boosting léger (peut améliorer légèrement)
5. **Validation temporelle croisée** — utiliser TimeSeriesSplit de sklearn au lieu d'un simple 80/20

---

## Leçons pour le ML financier

- **Les modèles simples gagnent souvent** dans les données bruitées à faible ratio signal/bruit
- **La régularisation est indispensable** : sans Ridge, overfitting garanti
- **Le split temporel est obligatoire** : jamais de `train_test_split(shuffle=True)` sur des séries temporelles
- **EDA avant modélisation** : comprendre ses données économise beaucoup de temps
- **100 features choisies > 895 features brutes** : la qualité prime sur la quantité

---

*Challenge Kaggle : DRW - Crypto Market Prediction*  
*Projet personnel d'apprentissage ML*
