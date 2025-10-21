Dataset: https://www.kaggle.com/datasets/parisrohan/credit-score-classification?resource=download

Problem Statement
You are working as a data scientist in a global finance company. Over the years, the company has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts.

Task
Given a person’s credit-related information, build a machine learning model that can classify the credit score.

- Vergleich komplexe generative KI-Modelle und klassiche supervised Modelle bei Klassifikationsaufgabe
- https://github.com/OpenTabular/DeepTabular
🧾 1️⃣ Datensatzüberblick

📂 Kaggle: Credit Score Classification Dataset
Eigenschaft
Beschreibung
Zielvariable
Credit_Score (typisch: Poor, Standard, Good)
Aufgabentyp
Multiclass-Klassifikation
Anzahl Zeilen
ca. 28 000
Anzahl Features
~18–25 (je nach Preprocessing)
Featuretypen
numerisch (z. B. Annual_Income, Monthly_Inhand_Salary) + kategorial (z. B. Occupation, Credit_Mix)
Schwierigkeit
Mittel — unbalancierte Klassen, Ausreißer, Missing Values
Reale Anwendung
Kreditwürdigkeitsprüfung → Klassisches tabellarisches Problem

🧰 2️⃣ Eignung der Modelle
Modell
Unterstützt numerisch/kategorial
GPU empfohlen?
Erwartung
Logistic Regression (LR)
✅ numerisch (nach Encoding)
❌
Baseline
Random Forest (RF)
✅ gemischt
❌
Solide, interpretiert
XGBoost (XGB)
✅ gemischt
optional
Sehr stark, oft SOTA klassisch
MambularClassifier (MAM)
✅ (integrierte Embeddings)
✅
DeepTabular-Vergleich
FTTransformerClassifier (FTT)
✅ (Embeddings + Attention)
✅
DeepTabular-SOTA

## Setup
- UV installieren
    - `uv sync` um alle Pakete für das Projekt zu laden