import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score

# Caricamento (usa il punto e virgola come indicato nel tuo file)
df = pd.read_csv('dataset.csv', sep=';')

# Lista delle colonne che vogliamo TENERE (le più significative)
# Teniamo dati anagrafici base e soprattutto l'andamento accademico
cols_to_keep = [
    'Tuition fees up to date', 'Scholarship holder', 'Debtor', 'Age at enrollment', 'Gender',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Target'
]

# Tagliamo il dataset
df_clean = df[cols_to_keep]

# Verifica immediata
print(df_clean.info())
print(df_clean.head())

# Encoding del Target
le = LabelEncoder()
df_clean['Target'] = le.fit_transform(df_clean['Target'])
# Nota: ora Dropout/Enrolled/Graduate sono 0, 1, 2

# Divisione Feature (X) e Target (y)
X = df_clean.drop('Target', axis=1)
y = df_clean['Target']

# Split Training e Test (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling Feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CONTROLLO MISSING VALUES
print("Valori mancanti per colonna:")
print(df_clean.isnull().sum())

# ANALISI OUTLIERS (Esempio su Età)
plt.figure(figsize=(8,4))
sns.boxplot(x=df_clean['Age at enrollment'])
plt.title('Distribuzione Età all\'iscrizione (Analisi Outlier)')
plt.show()

#  ANALISI SBILANCIAMENTO CLASSI
print("\nDistribuzione Target:")
print(df_clean['Target'].value_counts(normalize=True) * 100) # Percentuali
sns.countplot(x='Target', data=df_clean)
plt.title('Distribuzione delle Classi (Sbilanciamento)')
plt.show()

print("Pre-processing completato. Dati pronti per il modello.")

# Creazione del Modello
# class_weight='balanced' per gestire lo sbilanciamento analizzato prima
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Cross-Validation (Valutazione di robustezza)
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5)
print(f"Accuratezza media Cross-Val: {cv_scores.mean():.2f}")

# Addestramento
rf_model.fit(X_train_scaled, y_train)

# Predizione sul Test Set (Dati mai visti)
y_pred = rf_model.predict(X_test_scaled)

print("\nReport di Classificazione:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# GRAFICO 1: MATRICE DI CONFUSIONE
#plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title('Matrice di Confusione - Random Forest')
plt.tight_layout()
plt.show()


# GRAFICO 2: FEATURE IMPORTANCE
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(
    x='Importance',
    y='Feature',
    data=feature_importance_df,
    hue='Feature',
    palette='viridis',
    legend=False
)
plt.title('Importanza delle Feature nel Modello')
plt.xlabel('Grado di Importanza')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
plt.tight_layout()