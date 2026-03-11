"""
app.py  —  API Flask para el Simulador de Hortalizas
======================================================
Carga los archivos .pkl generados por entrenar_modelo.py
y expone un endpoint POST /simular que recibe las
variables del dataset y devuelve el ISP_Final predicho
junto con el Estado y recomendaciones.

Levantar:
  pip install flask flask-cors scikit-learn joblib pandas
  python app.py

Acceso local: http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
#  Rutas de los archivos PKL  (ajusta si cambiaste CARPETA_PKL)
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO   = os.path.join(BASE_DIR, "modelo_mlp.pkl")
RUTA_SCALER   = os.path.join(BASE_DIR, "scaler.pkl")
RUTA_ENCODERS = os.path.join(BASE_DIR, "encoders.pkl")
RUTA_FEATURES = os.path.join(BASE_DIR, "feature_names.pkl")
RUTA_INFO     = os.path.join(BASE_DIR, "modelo_info.json")

# ──────────────────────────────────────────────────────────
#  Cargar modelo al arrancar
# ──────────────────────────────────────────────────────────
print("Cargando modelo...")
modelo   = joblib.load(RUTA_MODELO)
scaler   = joblib.load(RUTA_SCALER)
encoders = joblib.load(RUTA_ENCODERS)
features = joblib.load(RUTA_FEATURES)

with open(RUTA_INFO, "r", encoding="utf-8") as f:
    info = json.load(f)

UMBRALES = info["umbrales_estado"]   # {"Bueno": 0.55, "Regular": 0.30, "Malo": 0.0}

print(f"  Modelo cargado — {len(features)} features")
print(f"  Features: {features}")
print("  Listo.\n")

# ──────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────
def isp_a_estado(isp: float) -> str:
    if isp >= UMBRALES["Bueno"]:   return "Bueno"
    if isp >= UMBRALES["Regular"]: return "Regular"
    return "Malo"

# Recomendaciones por estado + cultivo
RECOMENDACIONES = {
    "Bueno": {
        "default": "Las condiciones son óptimas para el cultivo. Mantén el riego y monitorea plagas regularmente.",
        "Tomate":    "Excelente — asegúrate de tutorar las plantas y controla la humedad relativa para evitar hongos.",
        "Lechuga":   "Condiciones ideales. Cosecha en las mañanas para mejor textura y sabor.",
        "Zanahoria": "El suelo está en buen estado. Asegúrate de que no haya compactación para raíces rectas.",
        "Espinaca":  "Óptimo. Considera una cosecha parcial para estimular nuevo crecimiento.",
        "Frijol":    "Excelente desarrollo. Verifica el soporte de enredaderas si es variedad trepadora.",
        "Maiz":      "Buenas condiciones. Monitorea el choclo para determinar el momento exacto de cosecha.",
        "Papa":      "Suelo ideal. Asegura un buen aporque para proteger los tubérculos de la luz.",
        "Cebolla":   "Óptimo. Reduce el riego cuando el follaje empiece a caer para inducir la maduración.",
        "Remolacha": "Condiciones perfectas. Puede cosechar cuando las raíces alcancen 5–7 cm de diámetro.",
        "Trigo":     "Excelente. Mantén vigilancia sobre roya y otras enfermedades fúngicas en esta etapa.",
    },
    "Regular": {
        "default": "Condiciones aceptables pero con margen de mejora. Revisa temperatura, pH y frecuencia de riego.",
        "Tomate":    "Revisa el pH del suelo (ideal 6.0–6.8) y aumenta el potasio si el follaje amarillea.",
        "Lechuga":   "Temperatura un poco alta o baja puede causar 'espigado'. Verifica la exposición solar.",
        "Zanahoria": "Ajusta el pH a 6.0–6.8 y asegura buena profundidad de suelo suelto (30 cm mínimo).",
        "Espinaca":  "Tolera frío mejor que calor. Si supera 25°C, considera sombra parcial.",
        "Frijol":    "Reduce el nitrógeno si hay exceso de follaje. Favorece fósforo para la floración.",
        "Maiz":      "Revisa la densidad de siembra y asegura polinización cruzada adecuada.",
        "Papa":      "pH bajo reduce disponibilidad de nutrientes. Encala si está por debajo de 5.5.",
        "Cebolla":   "Asegura buen drenaje. El exceso de humedad favorece enfermedades del bulbo.",
        "Remolacha": "Añade boro al suelo si hay decoloración en las hojas internas.",
        "Trigo":     "Monitorea la humedad relativa. Alta humedad favorece enfermedades fúngicas.",
    },
    "Malo": {
        "default": "Condiciones críticas. El cultivo está en riesgo. Intervención urgente requerida.",
        "Tomate":    "Riesgo alto. Revisa riego (evitar encharcamiento), trata plagas y ajusta el pH de inmediato.",
        "Lechuga":   "Condiciones muy desfavorables. Considera trasplante a zona con mejor temperatura y luz.",
        "Zanahoria": "Suelo probablemente compactado o pH inadecuado. Enmienda urgente del suelo.",
        "Espinaca":  "Temperatura o luz fuera de rango. Evalúa reubicación o cosecha anticipada.",
        "Frijol":    "Alta probabilidad de pérdida. Trata plagas y revisa el sistema de riego.",
        "Maiz":      "Déficit hídrico o exceso de calor. Riego urgente y cobertura vegetal del suelo.",
        "Papa":      "Riesgo de pudrición de tubérculos. Mejora drenaje y aplica fungicida preventivo.",
        "Cebolla":   "Probable daño por hongos o plagas del suelo. Trata con fungicida sistémico.",
        "Remolacha": "Estrés severo. Revisa salinidad del suelo y ajusta el riego.",
        "Trigo":     "Condiciones muy adversas. Evalúa si es viable continuar el ciclo o resembrar.",
    },
}

PASOS_POR_ESTADO = {
    "Bueno": [
        "✅ Mantén el plan de riego actual — no cambies la frecuencia.",
        "👁 Monitorea semanalmente para detectar plagas tempranas.",
        "📋 Registra la fecha estimada de cosecha según el ciclo del cultivo.",
    ],
    "Regular": [
        "⚠️ Revisa el pH del suelo — el rango ideal es 6.0–7.0 para la mayoría de cultivos.",
        "💧 Ajusta la frecuencia de riego: ni exceso ni déficit.",
        "🌿 Considera aplicar fertilizante orgánico para mejorar la estructura del suelo.",
        "🔍 Inspecciona hojas y raíces en busca de signos de estrés o plagas.",
    ],
    "Malo": [
        "🚨 Acción inmediata: revisa si hay encharcamiento o sequía severa.",
        "🧪 Analiza el suelo — puede haber deficiencias de nutrientes o pH incorrecto.",
        "🐛 Aplica control de plagas si detectas daño en hojas, tallos o raíces.",
        "☀️ Verifica que el cultivo reciba las horas de sol adecuadas para su especie.",
        "📞 Considera asesoría técnica si las condiciones no mejoran en 48–72 horas.",
    ],
}

# Título del resultado
TITULOS = {
    "Bueno":   "¡Condiciones óptimas! Tu cultivo está bien.",
    "Regular": "Condiciones aceptables — hay margen de mejora.",
    "Malo":    "Condiciones críticas — se requiere intervención urgente.",
}

# ──────────────────────────────────────────────────────────
#  App Flask
# ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Permite peticiones desde el frontend React


@app.route("/", methods=["GET"])
def index():
    """Health check básico."""
    return jsonify({
        "status": "ok",
        "modelo": "MLP Regressor — ISP_Final",
        "features": features,
        "umbrales": UMBRALES,
        "metricas": info.get("metricas_test", {}),
    })


@app.route("/info", methods=["GET"])
def get_info():
    """Devuelve metadatos del modelo: features, clases de encoders, umbrales."""
    return jsonify(info)


@app.route("/simular", methods=["POST"])
def simular():
    """
    Recibe un JSON con las variables del dataset y devuelve la predicción.

    Campos esperados:
      Temperatura                    (float/int)   ej: 22
      Humedad                        (float/int)   ej: 65
      pH_Suelo                       (float)       ej: 6.5
      Luz_Solar                      (float)       ej: 8.0
      Precipitacion                  (float/int)   ej: 30
      Altitud                        (float/int)   ej: 500
      Tipo_Suelo                     (str)         "Mixto" | "Arcilloso" | "Arenoso"
      Tipo_Irrigacion                (str)         "Goteo" | "Aspersion" | "Gravedad"
      Uso_Fertilizantes              (str)         "Organicos" | "Quimicos"
      Presencia_Plagas_Enfermedades  (str)         "No" | "Si"
      Tipo_Producto                  (str)         "Lechuga" | "Tomate" | etc.
      Mes_Siembra     (opcional, int 1–12)         default: mes actual
      Dia_Anio        (opcional, int 1–365)        default: día actual
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Body vacío o no es JSON válido"}), 400

    # ── Completar campos opcionales ────────────────────────
    from datetime import date
    hoy = date.today()
    data.setdefault("Mes_Siembra", hoy.month)
    data.setdefault("Dia_Anio",    hoy.timetuple().tm_yday)

    # ── Validar campos requeridos ──────────────────────────
    campos_cat = ["Tipo_Suelo", "Tipo_Irrigacion", "Uso_Fertilizantes",
                  "Presencia_Plagas_Enfermedades", "Tipo_Producto"]
    campos_num = ["Temperatura", "Humedad", "pH_Suelo", "Luz_Solar",
                  "Precipitacion", "Altitud"]

    errores = []
    for c in campos_cat + campos_num:
        if c not in data:
            errores.append(f"Falta campo: '{c}'")
    if errores:
        return jsonify({"error": "Campos faltantes", "detalle": errores}), 400

    # ── Limpiar y encodear categóricas ─────────────────────
    fila = {}
    for c in campos_cat:
        valor = str(data[c]).strip()
        le    = encoders[c]
        clases_validas = list(le.classes_)
        # Buscar match case-insensitive + sin espacios extra
        match = next(
            (cls for cls in clases_validas if cls.strip().lower() == valor.lower()),
            None
        )
        if match is None:
            return jsonify({
                "error": f"Valor inválido para '{c}': '{valor}'",
                "valores_validos": clases_validas,
            }), 400
        fila[c] = int(le.transform([match])[0])

    # ── Numéricas ──────────────────────────────────────────
    for c in campos_num:
        try:
            fila[c] = float(data[c])
        except (ValueError, TypeError):
            return jsonify({"error": f"'{c}' debe ser numérico, recibido: {data[c]}"}), 400

    fila["Mes_Siembra"] = int(data["Mes_Siembra"])
    fila["Dia_Anio"]    = int(data["Dia_Anio"])

    # ── Construir DataFrame en el orden correcto ───────────
    df_input = pd.DataFrame([fila], columns=features)

    # ── Escalar y predecir ─────────────────────────────────
    X_sc  = scaler.transform(df_input)
    isp   = float(np.clip(modelo.predict(X_sc)[0], 0.0, 1.0))
    estado = isp_a_estado(isp)

    # ── Armar respuesta ────────────────────────────────────
    cultivo = str(data["Tipo_Producto"]).strip()
    rec_por_cultivo = RECOMENDACIONES[estado]
    recomendacion   = rec_por_cultivo.get(cultivo, rec_por_cultivo["default"])

    respuesta = {
        # ── Predicción principal ──
        "isp_final": round(isp, 4),
        "estado":    estado,
        "titulo":    TITULOS[estado],
        "mensaje":   recomendacion,

        # ── Compatibilidad con el frontend anterior ──
        "humedadFinal": round(isp * 100, 1),   # para la barra de humedad visual
        "recomendacion": recomendacion,
        "pasos":    PASOS_POR_ESTADO[estado],

        # ── Detalle técnico ──
        "detalle": {
            "cultivo":    cultivo,
            "isp_raw":    round(isp, 6),
            "umbrales":   UMBRALES,
            "input_recibido": {k: data[k] for k in campos_cat + campos_num},
        },
    }
    return jsonify(respuesta)


@app.route("/encoders", methods=["GET"])
def get_encoders():
    """Devuelve los valores válidos para cada variable categórica."""
    return jsonify({
        col: list(le.classes_)
        for col, le in encoders.items()
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  API Flask — Simulador de Hortalizas")
    print(f"  Endpoint: POST /simular")
    print(f"  Corriendo en: http://0.0.0.0:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=True)