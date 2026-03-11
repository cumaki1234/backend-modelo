# ============================================================
#  MLP — Regresión sobre Índice de Salud de Plantas (ISP)
#  Tesis de grado — Perceptrón Multicapa (MLPRegressor)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  MLP — Regresión: Índice de Salud de Plantas (ISP)")
print("=" * 60)

# ── PASO 1: Cargar dataset ─────────────────────────────────
df = pd.read_excel(
    r"C:\Users\eduar\Downloads\data_10k.xlsx",
    sheet_name="agriculture_10k"
)
print(f"\n[1] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ── PASO 2: Selección de variables ─────────────────────────
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
TARGET = "ISP_Final"

df_model = df[FEATURES + [TARGET]].copy()
print(f"[2] Features: {len(FEATURES)} | Target: {TARGET}")
print(f"    ISP min={df_model[TARGET].min():.3f}  max={df_model[TARGET].max():.3f}  media={df_model[TARGET].mean():.3f}")

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

print(f"[3] Preprocesamiento completo")

# ── PASO 4: División entrenamiento / prueba ────────────────
X = df_model[FEATURES].values
y = df_model[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[4] División 80/20:")
print(f"    Entrenamiento: {len(X_train)} muestras")
print(f"    Prueba:        {len(X_test)} muestras")

# ── PASO 5: Escalado ──────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"[5] Escalado StandardScaler aplicado")

# ── PASO 6: Construcción y entrenamiento ──────────────────
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

# ── PASO 7: Evaluación ────────────────────────────────────
y_pred = mlp.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.clip(y_test, 1e-6, None))) * 100

print(f"\n[7] Métricas de evaluación:")
print(f"    R²:   {r2:.4f}  ({r2*100:.1f}% varianza explicada)")
print(f"    RMSE: {rmse:.4f}  (error promedio ±{rmse*100:.1f}pp en escala 0–1)")
print(f"    MAE:  {mae:.4f}")
print(f"    MAPE: {mape:.2f}%")

# ── PASO 8: Visualizaciones ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MLP Regresor — Índice de Salud de Plantas (ISP_Final)",
             fontsize=14, fontweight='bold')

# Gráfica 1: Real vs Predicho
axes[0].scatter(y_test, y_pred, alpha=0.4, color='steelblue', s=15)
lims = [min(y_test.min(), y_pred.min()) - 0.02,
        max(y_test.max(), y_pred.max()) + 0.02]
axes[0].plot(lims, lims, 'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("ISP Real")
axes[0].set_ylabel("ISP Predicho")
axes[0].set_title(f"Real vs Predicho  —  R² = {r2:.4f}")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Gráfica 2: Curva de pérdida
axes[1].plot(mlp.loss_curve_, color='steelblue', linewidth=2, label='Entrenamiento')
axes[1].set_xlabel("Épocas")
axes[1].set_ylabel("Pérdida (MSE)")
axes[1].set_title("Curva de Aprendizaje")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mlp_regresion_resultados.png", dpi=150, bbox_inches='tight')
print("\n[8] Gráficas guardadas: mlp_regresion_resultados.png")

# ── PASO 9: Resumen ejecutivo ─────────────────────────────
print("\n" + "=" * 60)
print("  RESUMEN EJECUTIVO")
print("=" * 60)
print(f"  Modelo:    MLPRegressor (128→64→32, ReLU, Adam)")
print(f"  Dataset:   {len(df)} registros, {len(FEATURES)} features")
print(f"  Target:    ISP_Final (índice de salud 0–1)")
print(f"  Épocas:    {mlp.n_iter_}")
print(f"  R²:        {r2:.4f}  →  {r2*100:.1f}% varianza explicada")
print(f"  RMSE:      {rmse:.4f}  →  error promedio ±{rmse*100:.1f}pp")
print(f"  MAE:       {mae:.4f}")
print(f"  MAPE:      {mape:.2f}%")
print("\n✅ Proceso completado exitosamente")