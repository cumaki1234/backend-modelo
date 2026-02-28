# ============================================================
#  PASO 1: Entrenar el modelo y guardarlo en disco
#  Esto se ejecuta UNA SOLA VEZ para generar los archivos
#  modelo_mlp.pkl, scaler.pkl y encoders.pkl
# ============================================================
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("=" * 55)
print("  Entrenamiento y guardado del modelo MLP")
print("=" * 55)

# ── Cargar dataset ─────────────────────────────────────────
df = pd.read_csv(r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\dataset_hortalizas_ciclo_corto_v2.csv", sep=";")

# ── Preprocesamiento ───────────────────────────────────────
df["Fecha_Siembra"] = pd.to_datetime(df["Fecha_Siembra"])
df["Mes_Siembra"]   = df["Fecha_Siembra"].dt.month
df["Dia_Anio"]      = df["Fecha_Siembra"].dt.dayofyear
df.drop(columns=["Fecha_Siembra"], inplace=True)

columnas_cat = [
    "Tipo_Producto", "Tipo_Suelo", "Tipo_Irrigacion",
    "Uso_Fertilizantes", "Presencia_Plagas_Enfermedades"
]

# Guardar encoders para usarlos en la API
encoders = {}
for col in columnas_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop(columns=["Rendimiento_kg_ha"])
y = df["Rendimiento_kg_ha"]

# ── División y escalado ────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Entrenar MLP ───────────────────────────────────────────
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

print("\n Entrenando modelo...")
mlp.fit(X_train_sc, y_train)

y_pred = mlp.predict(X_test_sc)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)

print(f"Entrenamiento completado — Épocas: {mlp.n_iter_}")
print(f"   R²:   {r2:.4f}")
print(f"   RMSE: {rmse:,.0f} kg/ha")
print(f"   MAE:  {mae:,.0f} kg/ha")

# ── Guardar modelo, scaler y encoders ─────────────────────
os.makedirs("/home/claude/api_hortalizas", exist_ok=True)

joblib.dump(mlp,      r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\10MO\modelo_mlp.pkl")
joblib.dump(scaler,   r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\10MO\scaler.pkl")
joblib.dump(encoders, r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\10MO\encoders.pkl")

print("\n Archivos guardados en /api_hortalizas/:")
print("   - modelo_mlp.pkl   (red neuronal entrenada)")
print("   - scaler.pkl       (normalizador de datos)")
print("   - encoders.pkl     (codificadores de texto)")
