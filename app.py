# ============================================================
#  API Flask — Predicción de Rendimiento de Hortalizas
#  Modelo: Perceptrón Multicapa (MLP)
# ============================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  # Permite peticiones desde el frontend React

# ── Cargar modelo, scaler y encoders al iniciar la API ─────
BASE = os.path.dirname(os.path.abspath(__file__))
modelo   = joblib.load(os.path.join(BASE, "modelo_mlp.pkl"))
scaler   = joblib.load(os.path.join(BASE, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE, "encoders.pkl"))

# ── Umbrales de rendimiento por hortaliza (kg/ha) ──────────
# Basados en rangos agronómicos reales de la FAO
# Óptimo:   rendimiento >= 75% del máximo esperado
# Regular:  rendimiento entre 40% y 75% del máximo esperado
# Crítico:  rendimiento < 40% del máximo esperado
UMBRALES = {
    "Lechuga":    {"optimo": 26000, "regular": 14000},
    "Zanahoria":  {"optimo": 38000, "regular": 20000},
    "Remolacha":  {"optimo": 34000, "regular": 18000},
    "Cebolla":    {"optimo": 45000, "regular": 24000},
    "Espinaca":   {"optimo": 15000, "regular":  8000},
    "Tomate":     {"optimo": 75000, "regular": 40000},
    "Rábano":     {"optimo": 19000, "regular": 10000},
}

RECOMENDACIONES = {
    "Óptimo": {
        "mensaje": "Las condiciones del cultivo son favorables. El rendimiento se encuentra dentro del rango óptimo para esta hortaliza.",
        "recomendacion": "Mantener las condiciones actuales de riego, fertilización y control de plagas. Realizar monitoreo periódico del pH del suelo."
    },
    "Regular": {
        "mensaje": "El cultivo presenta condiciones aceptables pero con margen de mejora. El rendimiento está por debajo del nivel óptimo esperado.",
        "recomendacion": "Revisar el sistema de irrigación y el tipo de fertilizante aplicado. Verificar el pH del suelo y corregir si está fuera del rango ideal. Inspeccionar presencia de plagas."
    },
    "Crítico": {
        "mensaje": "El cultivo se encuentra en estado crítico. El rendimiento predicho es significativamente bajo para esta hortaliza.",
        "recomendacion": "Se recomienda intervención inmediata: revisar condiciones de temperatura y humedad, aplicar tratamiento fitosanitario, evaluar cambio de tipo de suelo o sistema de riego. Considerar consulta con agrónomo."
    }
}

def clasificar_estado(hortaliza, rendimiento_kg_ha):
    """Convierte el rendimiento numérico en un estado descriptivo."""
    umbrales = UMBRALES.get(hortaliza, {"optimo": 30000, "regular": 15000})
    if rendimiento_kg_ha >= umbrales["optimo"]:
        return "Óptimo"
    elif rendimiento_kg_ha >= umbrales["regular"]:
        return "Regular"
    else:
        return "Crítico"

def predecir(datos):
    """Preprocesa los datos de entrada y devuelve la predicción."""
    # Extraer fecha
    from datetime import datetime
    fecha = datetime.strptime(datos["Fecha_Siembra"], "%Y-%m-%d")
    mes   = fecha.month
    dia   = fecha.timetuple().tm_yday

    # Construir fila con el mismo orden que se usó al entrenar
    fila = {
        "Tipo_Producto":                 datos["Tipo_Producto"],
        "Temperatura":                   float(datos["Temperatura"]),
        "Humedad":                       float(datos["Humedad"]),
        "Tipo_Suelo":                    datos["Tipo_Suelo"],
        "Precipitacion":                 float(datos["Precipitacion"]),
        "Altitud":                       float(datos["Altitud"]),
        "Tipo_Irrigacion":               datos["Tipo_Irrigacion"],
        "pH_Suelo":                      float(datos["pH_Suelo"]),
        "Luz_Solar":                     float(datos["Luz_Solar"]),
        "Uso_Fertilizantes":             datos["Uso_Fertilizantes"],
        "Presencia_Plagas_Enfermedades": datos["Presencia_Plagas_Enfermedades"],
        "Mes_Siembra":                   mes,
        "Dia_Anio":                      dia,
    }

    df = pd.DataFrame([fila])

    # Aplicar encoders a columnas categóricas
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Escalar y predecir
    df_scaled = scaler.transform(df)
    rendimiento = float(modelo.predict(df_scaled)[0])
    return rendimiento

# ── RUTA PRINCIPAL: predicción ─────────────────────────────
@app.route("/predecir", methods=["POST"])
def ruta_predecir():
    """
    Recibe los datos del cultivo y devuelve el estado de rendimiento.

    Body JSON esperado:
    {
        "Tipo_Producto": "Lechuga",
        "Temperatura": 15.5,
        "Humedad": 70.0,
        "Tipo_Suelo": "Franco",
        "Precipitacion": 450.0,
        "Altitud": 2200.0,
        "Tipo_Irrigacion": "Goteo",
        "pH_Suelo": 6.5,
        "Luz_Solar": 8.0,
        "Uso_Fertilizantes": "NPK",
        "Presencia_Plagas_Enfermedades": "No",
        "Fecha_Siembra": "2024-03-15"
    }
    """
    try:
        datos = request.get_json()
        if not datos:
            return jsonify({"error": "No se recibieron datos JSON"}), 400

        # Validar campos requeridos
        campos_requeridos = [
            "Tipo_Producto", "Temperatura", "Humedad", "Tipo_Suelo",
            "Precipitacion", "Altitud", "Tipo_Irrigacion", "pH_Suelo",
            "Luz_Solar", "Uso_Fertilizantes",
            "Presencia_Plagas_Enfermedades", "Fecha_Siembra"
        ]
        faltantes = [c for c in campos_requeridos if c not in datos]
        if faltantes:
            return jsonify({
                "error": "Campos faltantes",
                "campos_faltantes": faltantes
            }), 400

        # Validar que la hortaliza sea válida
        hortalizas_validas = list(UMBRALES.keys())
        if datos["Tipo_Producto"] not in hortalizas_validas:
            return jsonify({
                "error": f"Hortaliza no válida. Opciones: {hortalizas_validas}"
            }), 400

        # Predecir y clasificar
        rendimiento = predecir(datos)
        estado      = clasificar_estado(datos["Tipo_Producto"], rendimiento)
        info        = RECOMENDACIONES[estado]

        # Obtener umbrales para incluirlos en la respuesta
        u = UMBRALES[datos["Tipo_Producto"]]

        return jsonify({
            "hortaliza":              datos["Tipo_Producto"],
            "rendimiento_predicho":   round(rendimiento, 2),
            "unidad":                 "kg/ha",
            "estado":                 estado,
            "mensaje":                info["mensaje"],
            "recomendacion":          info["recomendacion"],
            "umbrales_referencia": {
                "optimo":   f">= {u['optimo']:,} kg/ha",
                "regular":  f"{u['regular']:,} – {u['optimo']:,} kg/ha",
                "critico":  f"< {u['regular']:,} kg/ha"
            }
        }), 200

    except ValueError as e:
        return jsonify({"error": f"Valor inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# ── RUTA: simulación de balance hídrico ───────────────────
@app.route("/simular", methods=["POST"])
def ruta_simular():
    """
    Simula la evolución de humedad del suelo durante N días
    usando balance hídrico y evapotranspiración (Hargreaves).

    Body JSON esperado:
    {
        "radiacion": 500,
        "humedadSuelo": 55,
        "precipitacion": 3,
        "temperatura": 22,
        "cultivo": "lechuga",
        "etapa": "desarrollo",
        "riegoAplicado": 0,
        "dias": 7
    }
    """
    try:
        d = request.get_json()
        if not d:
            return jsonify({"error": "No se recibieron datos JSON"}), 400

        campos = ["radiacion", "humedadSuelo", "precipitacion",
                  "temperatura", "cultivo", "etapa"]
        faltantes = [c for c in campos if c not in d]
        if faltantes:
            return jsonify({"error": "Campos faltantes", "campos_faltantes": faltantes}), 400

        # ── Coeficientes de cultivo (Kc) ──────────────────
        KC = {
            "lechuga":   {"inicial": 0.7,  "desarrollo": 0.85, "maduracion": 1.0 },
            "tomate":    {"inicial": 0.6,  "desarrollo": 0.9,  "maduracion": 1.15},
            "zanahoria": {"inicial": 0.5,  "desarrollo": 0.85, "maduracion": 1.05},
            "espinaca":  {"inicial": 0.65, "desarrollo": 0.9,  "maduracion": 1.0 },
            "rabano":    {"inicial": 0.6,  "desarrollo": 0.8,  "maduracion": 0.95},
        }

        cultivo = d["cultivo"].lower()
        etapa   = d["etapa"].lower()
        if cultivo not in KC:
            return jsonify({"error": f"Cultivo no válido. Opciones: {list(KC.keys())}"}), 400
        if etapa not in KC[cultivo]:
            return jsonify({"error": f"Etapa no válida. Opciones: inicial, desarrollo, maduracion"}), 400

        kc          = KC[cultivo][etapa]
        num_dias    = max(1, min(30, int(d.get("dias", 7))))
        humedad     = float(d["humedadSuelo"])
        temperatura = float(d["temperatura"])
        radiacion   = float(d["radiacion"])
        precipit    = float(d["precipitacion"])
        riego       = float(d.get("riegoAplicado", 0))

        # Nuevos parametros opcionales
        horas_sol   = float(d.get("horasSol", 6))
        techo       = d.get("techo", "abierto")
        contenedor  = d.get("contenedor", "tierra")
        riego_habit = d.get("riegoHabitual", "normal")

        factor_sol        = min(horas_sol / 12.0, 1.0)
        factor_techo      = {"abierto": 1.0, "semitecho": 0.65, "techo": 0.35}.get(techo, 1.0)
        radiacion_ef      = radiacion * factor_sol * factor_techo
        factor_contenedor = {"tierra": 1.0, "maceta_grande": 1.3, "maceta_chica": 1.7}.get(contenedor, 1.0)
        extra_riego_diario = {"poco": 0.3, "normal": 0.8, "abundante": 1.8}.get(riego_habit, 0.8)

        et_acum    = 0.0
        datos_dias = []

        for dia in range(num_dias + 1):
            ra  = (radiacion_ef * 0.0864) / 2.45
            et0 = max(0.3, 0.0023 * ra * (temperatura + 17.8) * (10 ** 0.5))
            etc = et0 * kc * factor_contenedor

            agua_entrada = precipit + riego + extra_riego_diario
            drenaje      = (humedad - 80) * 0.3 if humedad > 80 else 0
            humedad      = max(5.0, min(100.0, humedad + agua_entrada - etc - drenaje))
            et_acum      += etc

            estado_dia = "critico" if humedad < 30 else "alerta" if humedad < 55 else "optimo"
            datos_dias.append({
                "dia":       f"D{dia}",
                "humedad":   round(humedad, 1),
                "critico":   30,
                "optimo":    60,
                "etc":       round(etc, 2),
                "estado_dia": estado_dia,
                "accion":    "Riega hoy" if estado_dia == "critico"
                             else "Considera regar" if estado_dia == "alerta"
                             else "Sin riego",
            })

        calendario = []
        for i, dato in enumerate(datos_dias):
            calendario.append({
                "dia":      i,
                "etiqueta": f"Dia {i}" if i > 0 else "Hoy",
                "humedad":  dato["humedad"],
                "estado":   dato["estado_dia"],
                "accion":   dato["accion"],
                "regar":    dato["estado_dia"] == "critico",
                "vigilar":  dato["estado_dia"] == "alerta",
            })

        # ── Nombres amigables para mostrar al usuario ─────
        nombres = {
            "lechuga": "lechuga", "tomate": "tomate",
            "zanahoria": "zanahoria", "espinaca": "espinaca", "rabano": "rábano"
        }
        etapas_texto = {
            "inicial": "recién sembrada", "desarrollo": "en crecimiento", "maduracion": "casi lista para cosechar"
        }
        nombre_cultivo = nombres.get(cultivo, cultivo)
        nombre_etapa   = etapas_texto.get(etapa, etapa)

        # Consejos prácticos específicos por cultivo
        consejos = {
            "lechuga":   "Riega despacio y cerca de la base para que la tierra absorba bien el agua. Evita mojar las hojas directamente.",
            "tomate":    "El tomate agradece que lo riegues en la mañana. Riega la tierra, no la planta, para evitar enfermedades.",
            "zanahoria": "La zanahoria necesita que el agua llegue profundo. Riega lentamente para que no se encharque la superficie.",
            "espinaca":  "La espinaca prefiere tierra siempre ligeramente húmeda. Riega poco pero con más frecuencia.",
            "rabano":    "El rábano crece muy rápido. Mantén la tierra húmeda de forma constante para que no se ponga fibroso.",
        }
        consejo_cultivo = consejos.get(cultivo, "Riega cerca de la raíz de la planta, no sobre las hojas.")

        # ── Clasificar estado final ────────────────────────
        hf = humedad
        if hf < 30:
            estado        = "critico"
            titulo        = "¡Tu planta necesita agua ahora!"
            desc          = "La tierra está muy seca. Si no la riegas hoy, tu cultivo puede dañarse."
            mensaje       = (f"Tu {nombre_cultivo} ({nombre_etapa}) está pasando sed. "
                             f"La tierra ha perdido demasiada humedad y si no actúas hoy, "
                             f"las hojas empezarán a marchitarse y la planta puede no recuperarse. "
                             f"No te preocupes, aún estás a tiempo de salvarla.")
            recomendacion = (f"Riega hoy mismo, lo antes posible. Usa suficiente agua para que la tierra quede "
                             f"húmeda unos 10 cm hacia abajo — puedes comprobarlo metiendo un dedo. "
                             f"{consejo_cultivo} "
                             f"Repite el riego mañana y observa cómo responde la planta.")
            pasos = [
                "💧 Riega hoy, no esperes al día siguiente",
                "👆 Mete un dedo en la tierra para ver si llegó el agua profundo",
                "🌿 Si las hojas están caídas, mejorarán en pocas horas tras regar",
                "📅 Vuelve a revisar mañana si necesita más agua",
            ]
        elif hf < 55:
            estado        = "alerta"
            titulo        = "Pronto necesitará agua"
            desc          = "La tierra se está secando. Es buen momento para regar antes de que la planta lo note."
            mensaje       = (f"Tu {nombre_cultivo} ({nombre_etapa}) está bien por ahora, "
                             f"pero la tierra se está quedando sin humedad. "
                             f"Si no llueve en los próximos días o no la riegas, "
                             f"la planta comenzará a sufrir. Es mejor actuar antes de que eso pase.")
            recomendacion = (f"Riega en los próximos 1 o 2 días. No necesitas hacerlo de emergencia, "
                             f"pero no lo dejes para después de mañana. "
                             f"{consejo_cultivo} "
                             f"Si ves que el clima va a estar muy soleado y caluroso, riega hoy mejor.")
            pasos = [
                "📅 Riega hoy o mañana, no lo dejes para después",
                "🌤 Si hace mucho sol, riega más temprano o al atardecer",
                "👆 Comprueba la tierra con el dedo antes de regar para confirmar",
                "🔄 Repite en 3 días si no ha llovido",
            ]
        else:
            estado        = "optimo"
            titulo        = "¡Tu planta está bien!"
            desc          = "La tierra tiene suficiente humedad. No necesitas regar por ahora."
            mensaje       = (f"Tu {nombre_cultivo} ({nombre_etapa}) está en buenas condiciones. "
                             f"La tierra tiene la humedad que necesita y la planta puede crecer "
                             f"sin problemas durante los próximos días. "
                             f"¡Buen trabajo cuidando tu cultivo!")
            recomendacion = (f"No necesitas regar hoy. Aprovecha para revisar que la planta esté sana: "
                             f"mira si tiene hojas amarillas, insectos o manchas extrañas. "
                             f"{consejo_cultivo} "
                             f"Vuelve a revisar la tierra en 2 o 3 días.")
            pasos = [
                "✅ No riegues hoy, la tierra ya tiene suficiente agua",
                "🔍 Revisa que las hojas se vean sanas y de buen color",
                "🐛 Fíjate si hay insectos o manchas en las hojas",
                "📅 Vuelve a revisar en 2 o 3 días",
            ]

        return jsonify({
            "datos":          datos_dias,
            "calendario":     calendario,
            "humedadFinal":   round(hf, 1),
            "etTotal":        round(et_acum, 2),
            "kc":             kc,
            "estado":         estado,
            "titulo":         titulo,
            "desc":           desc,
            "mensaje":        mensaje,
            "recomendacion":  recomendacion,
            "pasos":          pasos,
            "dias":           num_dias,
            "factores":       {
                "horasSol":    float(d.get("horasSol", 6)),
                "techo":       d.get("techo", "abierto"),
                "contenedor":  d.get("contenedor", "tierra"),
                "riegoHabitual": d.get("riegoHabitual", "normal"),
            },
        }), 200

    except ValueError as e:
        return jsonify({"error": f"Valor inválido: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


# ── RUTA: información del modelo ───────────────────────────
@app.route("/info", methods=["GET"])
def info_modelo():
    """Devuelve información general del modelo y opciones válidas."""
    return jsonify({
        "modelo":         "Perceptrón Multicapa (MLP)",
        "arquitectura":   "13 → 128 → 64 → 32 → 1",
        "objetivo":       "Predicción de rendimiento de hortalizas (kg/ha)",
        "hortalizas":     list(UMBRALES.keys()),
        "estados":        ["Óptimo", "Regular", "Crítico"],
        "valores_validos": {
            "Tipo_Suelo":     list(encoders["Tipo_Suelo"].classes_),
            "Tipo_Irrigacion": list(encoders["Tipo_Irrigacion"].classes_),
            "Uso_Fertilizantes": list(encoders["Uso_Fertilizantes"].classes_),
            "Presencia_Plagas_Enfermedades": list(encoders["Presencia_Plagas_Enfermedades"].classes_),
        }
    }), 200


# ── RUTA: health check ─────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mensaje": "API funcionando correctamente"}), 200


# ── Iniciar servidor ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  API Flask — Predicción de Hortalizas")
    print("  Servidor iniciado en http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)