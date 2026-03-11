# ============================================================
#  MLP HÍBRIDO — Índice de Salud de Plantas (ISP)
#  Regresión sobre ISP_Final + Clasificación derivada
#  Tesis de grado — Perceptrón Multicapa
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MLP HÍBRIDO — Índice de Salud de Plantas")
print("  Regresión: ISP_Final (0–1)")
print("  Clasificación derivada: Bueno / Regular / Malo")
print("=" * 60)

# ── PASO 1: Cargar dataset ─────────────────────────────────
df = pd.read_excel(
    r"C:\Users\eduar\Downloads\data_10k.xlsx",
    sheet_name="agriculture_10k"
)
print(f"\n[1] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
print(f"    Reales:     {(df['Origen']=='Real').sum()}")
print(f"    Sintéticos: {(df['Origen']=='Sintetico').sum()}")

# ── PASO 2: Selección de variables ─────────────────────────
# Features: solo variables observables originales
# ISP_Final: target de regresión (índice continuo 0-1)
# Estado:    se usa SOLO para evaluación, no como input
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
TARGET_REG   = "ISP_Final"   # target regresión
TARGET_CLASE = "Estado"      # usado solo en evaluación

# Umbrales agronómicos para convertir ISP → clase
UMBRAL_BUENO   = 0.55
UMBRAL_REGULAR = 0.30

def isp_a_clase(v):
    if v >= UMBRAL_BUENO:   return "Bueno"
    if v >= UMBRAL_REGULAR: return "Regular"
    return "Malo"

df_model = df[FEATURES + [TARGET_REG, TARGET_CLASE]].copy()
print(f"\n[2] Features: {len(FEATURES)} variables")
print(f"    Target regresión: {TARGET_REG}  (min={df_model[TARGET_REG].min():.3f}, max={df_model[TARGET_REG].max():.3f})")
print(f"    Distribución Estado real:")
print(df_model[TARGET_CLASE].value_counts().to_string())

# ── PASO 3: Preprocesamiento ───────────────────────────────
CATEGORICAS = [
    "Tipo_Suelo",
    "Tipo_Irrigacion",
    "Uso_Fertilizantes",
    "Presencia_Plagas_Enfermedades",
    "Tipo_Producto",
]
le = LabelEncoder()
for col in CATEGORICAS:
    df_model[col] = le.fit_transform(df_model[col].astype(str).str.strip())

print(f"\n[3] Variables categóricas codificadas con LabelEncoder")

# ── PASO 4: División entrenamiento / prueba ────────────────
X = df_model[FEATURES].values
y_reg   = df_model[TARGET_REG].values
y_clase = df_model[TARGET_CLASE].values     # solo para evaluación

X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clase,
    test_size=0.2,
    random_state=42
)
print(f"\n[4] División 80/20:")
print(f"    Entrenamiento: {len(X_train)} muestras")
print(f"    Prueba:        {len(X_test)} muestras")

# ── PASO 5: Escalado ──────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"[5] Escalado StandardScaler aplicado")

# ── PASO 6: Construcción y entrenamiento del MLP ──────────
# MLPRegressor: predice ISP_Final como valor continuo
# Arquitectura 128→64→32 neuronas, activación ReLU, Adam
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)

print("\n[6] Entrenando MLP Regresor (128 → 64 → 32)...")
mlp.fit(X_train_scaled, y_train)
print(f"    Épocas ejecutadas: {mlp.n_iter_}")
print(f"    Pérdida final:     {mlp.loss_:.6f}")

# ── PASO 7: Predicciones ──────────────────────────────────
y_pred_isp   = mlp.predict(X_test_scaled)

# Convertir ISP predicho → clase usando umbrales agronómicos
y_pred_clase = np.array([isp_a_clase(v) for v in y_pred_isp])

# ── PASO 8: Métricas de REGRESIÓN ─────────────────────────
r2   = r2_score(y_test, y_pred_isp)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_isp))
mae  = mean_absolute_error(y_test, y_pred_isp)
mape = np.mean(np.abs((y_test - y_pred_isp) / np.clip(y_test, 1e-6, None))) * 100

print(f"\n[7] ── MÉTRICAS DE REGRESIÓN (ISP_Final) ──")
print(f"    R²:   {r2:.4f}  ({r2*100:.1f}% varianza explicada)")
print(f"    RMSE: {rmse:.4f}  (en escala 0–1, equivale a ±{rmse*100:.1f}pp de ISP)")
print(f"    MAE:  {mae:.4f}")
print(f"    MAPE: {mape:.2f}%")

# ── PASO 9: Métricas de CLASIFICACIÓN (derivadas) ─────────
acc    = accuracy_score(yc_test, y_pred_clase)
clases = sorted(set(yc_test) | set(y_pred_clase))

print(f"\n[8] ── MÉTRICAS DE CLASIFICACIÓN (ISP → Estado) ──")
print(f"    Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
print()
print(classification_report(yc_test, y_pred_clase,
                             labels=clases, digits=4))

# ── PASO 10: Visualizaciones ──────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(
    "MLP Híbrido — Índice de Salud de Plantas\n"
    "Regresión (ISP_Final) + Clasificación derivada (Bueno / Regular / Malo)",
    fontsize=13, fontweight='bold', y=0.98
)

# ── Gráfica 1: Real vs Predicho (ISP) ─────────────────────
ax = axes[0, 0]
scatter_colors = {"Bueno":"#2e7d32", "Regular":"#f9a825", "Malo":"#c62828"}
for clase in clases:
    mask = yc_test == clase
    ax.scatter(y_test[mask], y_pred_isp[mask],
               alpha=0.4, s=18, label=clase,
               color=scatter_colors.get(clase, "gray"))
lims = [min(y_test.min(), y_pred_isp.min()) - 0.02,
        max(y_test.max(), y_pred_isp.max()) + 0.02]
ax.plot(lims, lims, 'k--', lw=1.5, label='Predicción perfecta', alpha=0.6)
# Líneas de umbrales
for umbral, label in [(UMBRAL_BUENO, f"Umbral Bueno ({UMBRAL_BUENO})"),
                       (UMBRAL_REGULAR, f"Umbral Regular ({UMBRAL_REGULAR})")]:
    ax.axhline(umbral, color='gray', linestyle=':', lw=1, alpha=0.7)
    ax.axvline(umbral, color='gray', linestyle=':', lw=1, alpha=0.7)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("ISP Real"); ax.set_ylabel("ISP Predicho")
ax.set_title(f"Regresión: Real vs Predicho  —  R² = {r2:.4f}")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# ── Gráfica 2: Curva de pérdida ───────────────────────────
ax = axes[0, 1]
ax.plot(mlp.loss_curve_, color='steelblue', lw=2, label='Entrenamiento')
if hasattr(mlp, 'validation_scores_') and mlp.validation_scores_:
    val_loss = [abs(1 - s) for s in mlp.validation_scores_]
    ax.plot(val_loss, color='tomato', lw=2, ls='--', label='Validación (aprox.)')
ax.set_xlabel("Épocas"); ax.set_ylabel("Pérdida (MSE)")
ax.set_title("Curva de Aprendizaje")
ax.legend(); ax.grid(True, alpha=0.3)

# ── Gráfica 3: Matriz de confusión ────────────────────────
ax = axes[1, 0]
cm = confusion_matrix(yc_test, y_pred_clase, labels=clases)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f"Matriz de Confusión  —  Accuracy = {acc*100:.1f}%")

# ── Gráfica 4: Distribución ISP predicho vs real ──────────
ax = axes[1, 1]
ax.hist(y_test,      bins=40, alpha=0.5, color='steelblue', label='ISP Real',    density=True)
ax.hist(y_pred_isp,  bins=40, alpha=0.5, color='tomato',    label='ISP Predicho',density=True)
for umbral in [UMBRAL_REGULAR, UMBRAL_BUENO]:
    ax.axvline(umbral, color='black', linestyle='--', lw=1.5, alpha=0.7)
ax.text(0.15,  ax.get_ylim()[1]*0.9, 'Malo',    fontsize=9, ha='center', color='#c62828')
ax.text(0.425, ax.get_ylim()[1]*0.9, 'Regular', fontsize=9, ha='center', color='#f57f17')
ax.text(0.775, ax.get_ylim()[1]*0.9, 'Bueno',   fontsize=9, ha='center', color='#2e7d32')
ax.set_xlabel("ISP_Final"); ax.set_ylabel("Densidad")
ax.set_title("Distribución del ISP: Real vs Predicho")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlp_hibrido_resultados.png", dpi=150, bbox_inches='tight')
print("\n[9] Gráficas guardadas: mlp_hibrido_resultados.png")

# ── PASO 11: Resumen ejecutivo ────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN EJECUTIVO")
print("=" * 60)
print(f"  Modelo:       MLPRegressor (128→64→32, ReLU, Adam)")
print(f"  Dataset:      {len(df)} registros ({(df['Origen']=='Real').sum()} reales + {(df['Origen']=='Sintetico').sum()} sintéticos)")
print(f"  Features:     {len(FEATURES)} variables de entrada")
print(f"  Épocas:       {mlp.n_iter_}")
print()
print(f"  ── Regresión (ISP_Final) ──")
print(f"  R²:           {r2:.4f}  → {r2*100:.1f}% varianza explicada")
print(f"  RMSE:         {rmse:.4f}  → error promedio ±{rmse*100:.1f}pp")
print(f"  MAE:          {mae:.4f}")
print()
print(f"  ── Clasificación derivada (ISP → Estado) ──")
print(f"  Accuracy:     {acc:.4f}  → {acc*100:.1f}%")
from sklearn.metrics import f1_score
f1_macro = f1_score(yc_test, y_pred_clase, average='macro', labels=clases)
f1_w     = f1_score(yc_test, y_pred_clase, average='weighted', labels=clases)
print(f"  F1 macro:     {f1_macro:.4f}")
print(f"  F1 weighted:  {f1_w:.4f}")
print()
print(f"  Umbrales ISP → Estado:")
print(f"    ISP ≥ {UMBRAL_BUENO}  → Bueno")
print(f"    ISP ≥ {UMBRAL_REGULAR}  → Regular")
print(f"    ISP <  {UMBRAL_REGULAR}  → Malo")
print()
print("Proceso completado exitosamente")