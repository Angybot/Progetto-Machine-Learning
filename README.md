# Analisi del Dropout e del Successo Accademico degli Studenti con Tecniche di Machine Learning

Questo progetto nasce con l'obiettivo di applicare modelli predittivi di Machine Learning per identificare precocemente gli studenti a rischio di abbandono scolastico (dropout) e prevedere il loro successo accademico. 
L'analisi si basa su fattori socio-economici, dati demografici e performance macro-accademiche raccolte al momento dell'iscrizione e durante il percorso di studi.

## Obiettivi del Progetto
- **Predizione Multiclasse/Binaria:** Classificare lo stato finale dello studente (Dropout, Enrolled, Graduate).
- **Feature Importance:** Identificare i fattori chiave (es. performance del primo/secondo semestre, background socio-economico) che influenzano maggiormente la ritenzione studentesca.
- **Ottimizzazione dei Modelli:** Costruire e ottimizzare pipeline di Machine Learning per massimizzare metriche quali Accuracy, F1-Score e ridurre i falsi negativi (studenti a rischio non identificati).

## Tecnologie e Librerie Utilizzate
- **Linguaggio:** Python
- **Analisi dei Dati & Preprocessing:** Pandas, NumPy
- **Visualizzazione Dati:** Matplotlib, Seaborn
- **Machine Learning & Pipeline:** Scikit-Learn
- **Modelli Avanzati:** [Inserisci qui eventuali modelli usati, es. XGBoost, LightGBM, Random Forest]

## Pipeline del Progetto e Risultati

1. **Esplorazione dei Dati (EDA):** Analisi delle distribuzioni, gestione degli sbilanciamenti delle classi e studio delle correlazioni tra le feature.
2. **Preprocessing & Feature Engineering:** 
   - Codifica delle variabili categoriali.
   - Scaling delle feature numeriche.
   - Gestione del data leakage e separazione rigorosa tra Train e Test set.
3. **Addestramento dei Modelli (Model Training):** Validazione incrociata (Cross-Validation) su diversi algoritmi di classificazione.
4. **Tuning dei Iperparametri (Hyperparameter Tuning):** Ottimizzazione dei modelli tramite tecniche di Grid Search / Randomized Search.
5. **Valutazione (Evaluation):** Analisi approfondita dei risultati tramite matrici di confusione e report di classificazione.
