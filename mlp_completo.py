# ============================================================
#  MLP - Predicción de Rendimiento de Hortalizas de Ciclo Corto
#  Tesis - Perceptrón Multicapa (MLPRegressor)
# ============================================================

# ── Librerías ─────────────────────────────────────────────
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
print("  MLP — Rendimiento de Hortalizas de Ciclo Corto")
print("=" * 60)

# ── PASO 1: Cargar dataset ─────────────────────────────────
df = pd.read_csv(r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\dataset_hortalizas_ciclo_corto_v2.csv", sep=";")
print(f"\n[1] Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# ── PASO 2: Preprocesamiento ───────────────────────────────
df["Fecha_Siembra"] = pd.to_datetime(df["Fecha_Siembra"])
df["Mes_Siembra"]   = df["Fecha_Siembra"].dt.month
df["Dia_Anio"]      = df["Fecha_Siembra"].dt.dayofyear
df.drop(columns=["Fecha_Siembra"], inplace=True)

le = LabelEncoder()
for col in ["Tipo_Producto","Tipo_Suelo","Tipo_Irrigacion",
            "Uso_Fertilizantes","Presencia_Plagas_Enfermedades"]:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=["Rendimiento_kg_ha"])
y = df["Rendimiento_kg_ha"]
print(f"[2] Preprocesamiento completo — Features: {X.shape[1]}")

# ── PASO 3: División y escalado ────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print(f"[3] División — Entrenamiento: {len(X_train)} | Prueba: {len(X_test)}")

# ── PASO 4: Construcción y entrenamiento del MLP ───────────
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)

print("[4] Entrenando MLP (128-64-32)...")
mlp.fit(X_train_scaled, y_train)
print(f"    Épocas: {mlp.n_iter_} | Pérdida final: {mlp.loss_:.2f}")

# ── PASO 5: Evaluación ────────────────────────────────────
y_pred = mlp.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"\n[5] Métricas de evaluación:")
print(f"    R²:   {r2:.4f}  ({r2*100:.1f}% varianza explicada)")
print(f"    RMSE: {rmse:,.0f} kg/ha")
print(f"    MAE:  {mae:,.0f} kg/ha")
print(f"    MAPE: {mape:.1f}%")

# ── PASO 6: Visualizaciones ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MLP — Predicción de Rendimiento de Hortalizas", fontsize=14, fontweight='bold')

# Gráfica 1: Real vs Predicho
axes[0].scatter(y_test, y_pred, alpha=0.5, color='steelblue', s=20)
lim = [y_test.min(), y_test.max()]
axes[0].plot(lim, lim, 'r--', lw=2, label='Predicción perfecta')
axes[0].set_xlabel("Rendimiento Real (kg/ha)")
axes[0].set_ylabel("Rendimiento Predicho (kg/ha)")
axes[0].set_title(f"Real vs Predicho  —  R² = {r2:.3f}")
axes[0].legend()

# Gráfica 2: Curva de pérdida
axes[1].plot(mlp.loss_curve_, color='steelblue', linewidth=2)
axes[1].set_xlabel("Épocas")
axes[1].set_ylabel("Pérdida (MSE)")
axes[1].set_title("Curva de aprendizaje")

plt.tight_layout()
plt.savefig("mlp_resultados.png", dpi=150, bbox_inches='tight')
print("\n[6] Gráficas guardadas: mlp_resultados.png")
print("\n Proceso completado exitosamente")
