# ============================================================
#  MLP — Predicción de Estado de Salud de Plantas
#  Clasificación: Bueno / Regular / Malo
#  Tesis de grado — Perceptrón Multicapa (MLPClassifier)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MLP — Clasificación de Estado de Salud de Plantas")
print("  Clases: Bueno | Regular | Malo")
print("=" * 60)

# ── PASO 1: Cargar dataset ─────────────────────────────────
# Leer solo la hoja 'agriculture' del Excel
df = pd.read_excel(
    r"C:\Users\eduar\Downloads\data_10k.xlsx",
    sheet_name="agriculture_10k"
)
print(f"\n[1] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ── PASO 2: Selección de columnas ─────────────────────────
# FEATURES: solo variables observables originales
# (los scores e ISP son derivados de la variable objetivo — no entran)
FEATURES = [
    "Temperatura",
    "Humedad",
    "pH_Suelo",
    "Luz_Solar",
    "Precipitacion",
    "Altitud",
    "Tipo_Suelo",
    "Tipo_Irrigacion",
    "Uso_Fertilizantes",
    "Presencia_Plagas_Enfermedades",
    "Tipo_Producto",
]
TARGET = "Estado"

df = df[FEATURES + [TARGET]].copy()
print(f"[2] Features seleccionadas: {len(FEATURES)}")
print(f"    Variable objetivo: {TARGET}")
print(f"    Distribución de clases:")
print(df[TARGET].value_counts().to_string(index=True))

# ── PASO 3: Preprocesamiento ───────────────────────────────
# Fecha_Siembra ya no está en FEATURES — se descartó correctamente.
# Convertir variables categóricas con LabelEncoder
CATEGORICAS = [
    "Tipo_Suelo",
    "Tipo_Irrigacion",
    "Uso_Fertilizantes",
    "Presencia_Plagas_Enfermedades",
    "Tipo_Producto",
]

le = LabelEncoder()
for col in CATEGORICAS:
    df[col] = le.fit_transform(df[col].astype(str).str.strip())

# Codificar variable objetivo
le_target = LabelEncoder()
df[TARGET] = le_target.fit_transform(df[TARGET])
# Guardar mapeo para interpretación
clases = le_target.classes_          # ['Bueno', 'Malo', 'Regular'] — orden alfabético
print(f"\n[3] Codificación de clases: {dict(enumerate(clases))}")

X = df[FEATURES].values
y = df[TARGET].values
print(f"    Preprocesamiento completo — Shape X: {X.shape}")

# ── PASO 4: División entrenamiento / prueba ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # mantiene proporción de clases en ambos splits
)
print(f"\n[4] División estratificada:")
print(f"    Entrenamiento: {len(X_train)} muestras")
print(f"    Prueba:        {len(X_test)} muestras")

# ── PASO 5: Escalado de features ──────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"[5] Escalado StandardScaler aplicado")

# ── PASO 6: Construcción y entrenamiento del MLP ───────────
# Arquitectura: 3 capas ocultas (128 → 64 → 32 neuronas)
# Activación ReLU + optimizador Adam + early stopping
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=50,
    verbose=False
)

print("\n[6] Entrenando MLP (128 → 64 → 32)...")
mlp.fit(X_train_scaled, y_train)
print(f"    Épocas ejecutadas:  {mlp.n_iter_}")
print(f"    Pérdida final:      {mlp.loss_:.4f}")

# ── PASO 7: Evaluación ────────────────────────────────────
y_pred      = mlp.predict(X_test_scaled)
y_pred_prob = mlp.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print(f"\n[7] Métricas de evaluación:")
print(f"    Accuracy global: {accuracy:.4f}  ({accuracy*100:.1f}%)\n")
print("    Reporte por clase:")
print(classification_report(y_test, y_pred, target_names=clases, digits=4))

# ── PASO 8: Visualizaciones ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("MLP — Clasificación Estado de Salud de Plantas",
             fontsize=14, fontweight='bold')

# ── Gráfica 1: Matriz de confusión ────────────────────────
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Matriz de Confusión")

# ── Gráfica 2: Curva de pérdida durante entrenamiento ─────
axes[1].plot(mlp.loss_curve_, color='steelblue', linewidth=2, label='Entrenamiento')
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_ is not None:
    # Convertir scores de validación a pérdida aproximada (1 - accuracy)
    val_loss = [1 - s for s in mlp.validation_scores_]
    axes[1].plot(val_loss, color='tomato', linewidth=2,
                 linestyle='--', label='Validación')
axes[1].set_xlabel("Épocas")
axes[1].set_ylabel("Pérdida (log-loss)")
axes[1].set_title("Curva de Aprendizaje")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# ── Gráfica 3: Distribución de predicciones vs real ───────
ancho   = 0.35
x_pos   = np.arange(len(clases))
real_ct = [np.sum(y_test  == i) for i in range(len(clases))]
pred_ct = [np.sum(y_pred  == i) for i in range(len(clases))]

axes[2].bar(x_pos - ancho/2, real_ct, ancho, label='Real',    color='steelblue', alpha=0.8)
axes[2].bar(x_pos + ancho/2, pred_ct, ancho, label='Predicho',color='tomato',    alpha=0.8)
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(clases)
axes[2].set_ylabel("Cantidad de muestras")
axes[2].set_title("Distribución Real vs Predicha")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("mlp_clasificacion_resultados.png", dpi=150, bbox_inches='tight')
print("\n[8] Gráficas guardadas: mlp_clasificacion_resultados.png")

# ── PASO 9: Resumen ejecutivo ─────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN EJECUTIVO")
print("=" * 60)
print(f"  Modelo:         MLP Clasificador (128-64-32, ReLU, Adam)")
print(f"  Dataset:        {df.shape[0]} registros, {len(FEATURES)} features")
print(f"  Clases:         {' / '.join(clases)}")
print(f"  Accuracy:       {accuracy*100:.1f}%")
print(f"  Épocas:         {mlp.n_iter_}")

from sklearn.metrics import classification_report as cr
rep = cr(y_test, y_pred, target_names=clases, output_dict=True)
for clase in clases:
    f1 = rep[clase]['f1-score']
    print(f"  F1 {clase:<8}:   {f1:.4f}")

print("\n Proceso completado exitosamente")