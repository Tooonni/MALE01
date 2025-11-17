# Credit Score Classification – Klassische vs. DeepTabular Modelle

Dieses Projekt untersucht, ob klassische Machine-Learning-Modelle im Kredit-Scoring mit tabellarischen Daten gegenüber modernen DeepTabular-Ansätzen konkurrenzfähig bleiben. Die Arbeit basiert auf dem Kaggle-Datensatz **Credit Score Classification** und prüft zwei Hypothesen:

- **H1:** Klassische Modelle (Logistic Regression, Random Forest, HistGradientBoosting) erreichen auf strukturierten Daten mindestens vergleichbare Scores wie generative DeepTabular-Modelle.
- **H2:** Der Ressourcenbedarf von DeepTabular-Modellen (Trainingszeit, Hardware) steht nicht im Verhältnis zu ihrem Mehrwert im Kredit-Scoring-Kontext.

## Wichtigste Erkenntnisse
- Die klassische Pipeline mit sauberem Preprocessing liefert stabile Balanced-Accuracy/AUC-Werte bei Sekunden-Laufzeiten → H1 bestätigt.
- DeepTabular-Modelle (MAM, FTTransformer) mussten auf `max_epochs = 1` limitiert werden; ein einziger Fit dauert länger als die komplette klassische Grid-Suche → H2 bestätigt.
- Merkmale wie `Outstanding_Debt`, `Credit_Utilization_Ratio`, `Num_of_Loan` sowie Zahlungsverhalten (`Payment_of_Min_Amount`, `Payment_Behaviour`) treiben die Trennschärfe und wurden bei der Feature-Priorisierung berücksichtigt.
- Datenbereinigung entfernte Identifikatoren, glättete Ausreißer und synchronisierte Train/Test-Schemata, sodass Modelle unmittelbar auf `data/train/train_fe.csv` und `data/test/test_fe.csv` trainiert werden können.

## Datenbasis
- **Quelle:** [Kaggle – Credit Score Classification](https://www.kaggle.com/datasets/parisrohan/credit-score-classification)
- **Zielvariable:** `Credit_Score` (Poor, Standard, Good)
- **Train/Test nach Feature-Engineering:** 72 878 Zeilen (train) / 50 000 Zeilen (test) mit 59 Merkmalen + Ziel
- **Eigenschaften:** numerische und kategoriale Features, Ausreißer bei Einkommen & Verschuldung, moderate Klassen-Ungleichgewichte

## Workflow & Notebooks
| Notebook | Fokus | Wichtigste Ergebnisse |
| --- | --- | --- |
| `01_Datenverständnis.ipynb` | Zieldefinition, Hypothesen, Datensatzwahl | Formuliert H1/H2 und Motiv für Klassisch vs. DeepTabular |
| `02_Datenbereinigung.ipynb` | Säuberung von Train/Test | Entfernt IDs, Textlisten, behandelt Missing Values, harmonisiert Präfixe |
| `03_EDA.ipynb` | Explorative Analyse | Identifiziert dominante Features (Schulden, Nutzung, Zahlungsverhalten) und verweist auf Ausreißer-Bereinigung |
| `04_Feature_Engineering.ipynb` | Pipeline-Bau, Exports | Erzeugt konsistente Feature-Sets `train_fe.csv` & `test_fe.csv` |
| `05_Modelling.ipynb` | Klassische Modelle | ColumnTransformer + GridSearch für LR, RF, HistGradientBoosting; Evaluation mit Balanced Accuracy, F1, AUC & Laufzeiten |
| `06_DeepTabularHyperparameterTuning.ipynb` | DeepTabular Tuning | RandomizedSearch auf Mambular (DeepTabular) → Laufzeitnachweis & Submission-Vorbereitung |

## Datenaufbereitung & Feature Engineering
- Entfernte Identifikatoren (`ID`, `Customer_ID`, `Name`, `SSN`) sowie Textlisten (`Type_of_Loan`) bzw. wandelte sie in zählbare Merkmale um.
- Imputete numerische Features per Median, kategoriale per häufigstem Wert; skalierte numerische Spalten und codierte Kategorien mittels One-Hot-Encoding.
- Bereinigte Ausreißer nach IQR-Regel und erzeugte Zusatzfeatures zu Kredit-Historie, Belastungsquoten, Zahlungsverhalten und saisonalen Effekten.
- Exportierte konsistente, modellfertige CSVs unter `data/train/train_fe.csv` und `data/test/test_fe.csv`.

## Modellierung & Evaluation
**Klassische Pipelines (Notebook 05)**
- Modelle: Logistic Regression, Random Forest, HistGradientBoostingClassifier.
- Einheitliche `ColumnTransformer`-Pipeline (Skalierung + One-Hot) mit GridSearchCV.
- Bewertet via Balanced Accuracy, Macro-F1, ROC AUC sowie Trainings-/Prädiktionszeit.
- Ergebnis: Klassische Modelle liefern robuste Scores bei sehr kurzer Laufzeit → bevorzugte Wahl für produktive Kredit-Scoring-Usecases.

**DeepTabular (Notebook 05 & 06)**
- Modelle: `MambularClassifier` und `FTTransformerClassifier` aus [OpenTabular/DeepTabular](https://github.com/OpenTabular/DeepTabular).
- Eigene Preprocessing-Pipeline mit Embeddings; RandomizedSearchCV (`n_iter=2`, `cv=2`) auf `max_epochs=1`.
- Beobachtung: Schon Minimal-Search benötigt GPU-Unterstützung (MPS) und übertrifft die Laufzeit der kompletten klassischen Experimente. Iterative Hyperparameter-Optimierung ist ohne zusätzliche Hardware nicht praktikabel.
- Fazit: DeepTabular bestätigt Flexibilität, aber der Ressourcenbedarf verdrängt die praktischen Vorteile bei diesem Datensatz.

## Reproduktion
1. **Python & Abhängigkeiten:** Projekt nutzt `uv` und Python ≥ 3.13. Installation:
   ```bash
   uv sync
   ```
2. **Datenstruktur:** Platzieren der Kaggle-Daten in `data/raw`, anschließend die Notebooks 02–04 ausführen, um `data/train/train_fe.csv` und `data/test/test_fe.csv` zu regenerieren.
3. **Experimente:** Notebooks der Reihe nach öffnen (01 → 06). Für DeepTabular-Modelle ist eine GPU oder Apple Silicon (MPS) empfehlenswert.

## Weiterführende Ideen
- Klassische Modelle: weitere Gradient-Boosting-Varianten (CatBoost, LightGBM) und Kalibrierung für Probability Scores.
- DeepTabular: Early-Stopping, reduzierte Feature-Dimensionen oder vortrainierte Checkpoints, um Laufzeiten zu senken.
- Explainability: SHAP/Permutation-Importances auf dem besten klassischen Modell zur Kommunikation gegenüber Kredit-Sachbearbeitern.