"""
entrenar_modelo.py
==================
Entrena el MLP sobre el dataset de hortalizas (data_10k.xlsx)
y guarda los archivos necesarios para la API Flask:
  - modelo_mlp.pkl
  - scaler.pkl
  - encoders.pkl
  - feature_names.pkl
  - modelo_info.json   (metadatos / referencia)

Ejecutar UNA SOLA VEZ antes de levantar la API:
  python entrenar_modelo.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
#  CONFIGURACIÓN — ajusta solo estas rutas
# ──────────────────────────────────────────────────────────

RUTA_DATASET = r"C:\Users\eduar\Downloads\data_10k.xlsx"
HOJA_EXCEL   = "agriculture_10k"
CARPETA_PKL  = r"C:\Users\eduar\OneDrive\Desktop\mis cosas\universidad\10MO"

# ──────────────────────────────────────────────────────────

print("=" * 55)
print("  Entrenamiento MLP — Dataset hortalizas (ISP_Final)")
print("=" * 55)

os.makedirs(CARPETA_PKL, exist_ok=True)

# ── 1. Cargar dataset ──────────────────────────────────────
print(f"\n[1/5] Cargando dataset: {RUTA_DATASET}")
df = pd.read_excel(RUTA_DATASET, sheet_name=HOJA_EXCEL)
print(f"      {df.shape[0]:,} filas × {df.shape[1]} columnas")

# ── 2. Feature engineering ─────────────────────────────────
print("\n[2/5] Preprocesamiento...")
df["Fecha_Siembra"] = pd.to_datetime(df["Fecha_Siembra"])
df["Mes_Siembra"]   = df["Fecha_Siembra"].dt.month
df["Dia_Anio"]      = df["Fecha_Siembra"].dt.dayofyear
df.drop(columns=["Fecha_Siembra"], inplace=True)

# Quitar columnas que no son features
df.drop(columns=["Estado", "Origen"], inplace=True)

# ── 3. Encoding de variables categóricas ───────────────────
print("\n[3/5] Codificando variables categóricas...")
columnas_cat = [
    "Tipo_Producto",
    "Tipo_Suelo",
    "Tipo_Irrigacion",
    "Uso_Fertilizantes",
    "Presencia_Plagas_Enfermedades",
]

encoders = {}
for col in columnas_cat:
    df[col] = df[col].astype(str).str.strip()
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    print(f"      {col}: {list(le.classes_)}")

# ── 4. División y escalado ─────────────────────────────────
X = df.drop(columns=["ISP_Final"])
y = df["ISP_Final"]
feature_names = list(X.columns)

print(f"\n[4/5] Entrenando...")
print(f"      Features ({len(feature_names)}): {feature_names}")
print(f"      Target: ISP_Final  [{y.min():.4f} – {y.max():.4f}]")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 5. Entrenar MLP ───────────────────────────────────────
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42,
)
mlp.fit(X_train_sc, y_train)

# ── Métricas ───────────────────────────────────────────────
y_pred = mlp.predict(X_test_sc)
r2     = r2_score(y_test, y_pred)
rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae    = float(mean_absolute_error(y_test, y_pred))

def isp_a_estado(v):
    if v >= 0.55: return "Bueno"
    if v >= 0.30: return "Regular"
    return "Malo"

acc = sum(
    isp_a_estado(r) == isp_a_estado(p)
    for r, p in zip(y_test, y_pred)
) / len(y_test)

print(f"\n      Épocas: {mlp.n_iter_}")
print(f"      R²:     {r2:.4f}")
print(f"      RMSE:   {rmse:.4f}  (escala 0–1)")
print(f"      MAE:    {mae:.4f}")
print(f"      Accuracy de clasificación: {acc*100:.1f}%")

# ── 6. Guardar archivos ────────────────────────────────────
print(f"\n[5/5] Guardando archivos en: {CARPETA_PKL}")

joblib.dump(mlp,          os.path.join(CARPETA_PKL, "modelo_mlp.pkl"))
joblib.dump(scaler,       os.path.join(CARPETA_PKL, "scaler.pkl"))
joblib.dump(encoders,     os.path.join(CARPETA_PKL, "encoders.pkl"))
joblib.dump(feature_names, os.path.join(CARPETA_PKL, "feature_names.pkl"))

info = {
    "descripcion":  "MLP Regressor — ISP_Final (0–1) → Estado (Bueno/Regular/Malo)",
    "features":     feature_names,
    "target":       "ISP_Final",
    "umbrales_estado": {"Bueno": 0.55, "Regular": 0.30, "Malo": 0.0},
    "encoders_clases": {col: list(le.classes_) for col, le in encoders.items()},
    "arquitectura": {"capas": [128, 64, 32], "activacion": "relu", "solver": "adam"},
    "metricas_test": {
        "r2": round(r2, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "accuracy_clase": round(acc, 4),
        "epocas": mlp.n_iter_,
    },
}
with open(os.path.join(CARPETA_PKL, "modelo_info.json"), "w", encoding="utf-8") as f:
    json.dump(info, f, ensure_ascii=False, indent=2)

print("\n  Archivos generados:")
for archivo in ["modelo_mlp.pkl", "scaler.pkl", "encoders.pkl", "feature_names.pkl", "modelo_info.json"]:
    ruta = os.path.join(CARPETA_PKL, archivo)
    kb   = os.path.getsize(ruta) / 1024 if os.path.exists(ruta) else 0
    print(f"    ✓  {archivo:<30} ({kb:.1f} KB)")

print("\n" + "=" * 55)
print("  ¡Listo! Ahora puedes levantar la API Flask.")
print("  python app.py")
print("=" * 55)