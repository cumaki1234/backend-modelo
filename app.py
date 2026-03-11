"""
app.py  —  API Flask — Simulador de Hortalizas
================================================
Recomendaciones dinámicas: analiza qué variables están
fuera del rango ideal para ese cultivo específico y genera
un diagnóstico preciso en lugar de textos genéricos.

Levantar:
  pip install flask flask-cors scikit-learn joblib pandas
  python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, json, os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
#  Archivos PKL  (deben estar en la misma carpeta que app.py)
# ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modelo   = joblib.load(os.path.join(BASE_DIR, "modelo_mlp.pkl"))
scaler   = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
encoders = joblib.load(os.path.join(BASE_DIR, "encoders.pkl"))
features = joblib.load(os.path.join(BASE_DIR, "feature_names.pkl"))

with open(os.path.join(BASE_DIR, "modelo_info.json"), encoding="utf-8") as f:
    info = json.load(f)

UMBRALES = info["umbrales_estado"]   # {"Bueno": 0.55, "Regular": 0.30, "Malo": 0.0}

print(f"Modelo cargado — {len(features)} features: {features}")

# ──────────────────────────────────────────────────────────
#  RANGOS IDEALES por cultivo (percentil 5–95 del Estado=Bueno)
#  Extraídos del dataset data_10k.xlsx
# ──────────────────────────────────────────────────────────
RANGOS_IDEALES = {
    "Cebolla":   {"Temperatura": (11, 27),  "Humedad": (46, 80),  "pH_Suelo": (4.7, 8.0), "Luz_Solar": (7.2, 17.2),  "Precipitacion": (28, 80),  "Altitud": (220,  3075)},
    "Espinaca":  {"Temperatura": (8,  20),  "Humedad": (54, 87),  "pH_Suelo": (5.6, 8.1), "Luz_Solar": (6.9, 17.3),  "Precipitacion": (35, 85),  "Altitud": (113,  2788)},
    "Frijol":    {"Temperatura": (9,  27),  "Humedad": (54, 82),  "pH_Suelo": (5.2, 8.0), "Luz_Solar": (7.3, 13.4),  "Precipitacion": (30, 82),  "Altitud": (0,    2226)},
    "Lechuga":   {"Temperatura": (10, 26),  "Humedad": (52, 87),  "pH_Suelo": (4.9, 7.5), "Luz_Solar": (6.6, 13.6),  "Precipitacion": (34, 88),  "Altitud": (176,  3058)},
    "Maiz":      {"Temperatura": (15, 40),  "Humedad": (43, 83),  "pH_Suelo": (4.8, 8.1), "Luz_Solar": (10.5, 15.4), "Precipitacion": (37, 92),  "Altitud": (0,    2345)},
    "Papa":      {"Temperatura": (12, 26),  "Humedad": (55, 85),  "pH_Suelo": (4.9, 6.9), "Luz_Solar": (6.6, 12.2),  "Precipitacion": (36, 90),  "Altitud": (1066, 3919)},
    "Remolacha": {"Temperatura": (10, 29),  "Humedad": (53, 83),  "pH_Suelo": (5.7, 7.4), "Luz_Solar": (11.8, 17.1), "Precipitacion": (29, 89),  "Altitud": (212,  2818)},
    "Tomate":    {"Temperatura": (13, 31),  "Humedad": (54, 80),  "pH_Suelo": (5.2, 7.4), "Luz_Solar": (6.9, 17.0),  "Precipitacion": (29, 85),  "Altitud": (0,    2092)},
    "Trigo":     {"Temperatura": (11, 26),  "Humedad": (45, 76),  "pH_Suelo": (5.7, 7.9), "Luz_Solar": (11.1, 17.7), "Precipitacion": (27, 84),  "Altitud": (193,  3276)},
    "Zanahoria": {"Temperatura": (13, 28),  "Humedad": (53, 82),  "pH_Suelo": (5.7, 7.8), "Luz_Solar": (6.8, 15.4),  "Precipitacion": (30, 85),  "Altitud": (220,  2817)},
}

# ──────────────────────────────────────────────────────────
#  LÓGICA DE MACETAS
#  El dataset no incluye tamaño de maceta pero sí tiene impacto
#  real en el cultivo (drenaje, volumen de sustrato, frecuencia riego)
# ──────────────────────────────────────────────────────────

# Cultivos que NO son viables en maceta pequeña por sistema radicular
MACETA_RESTRICCIONES = {
    "chica": {
        "no_viables":  ["Maiz", "Trigo", "Papa", "Zanahoria"],
        "advertencia": ["Tomate", "Frijol", "Cebolla", "Remolacha"],
        "ideales":     ["Lechuga", "Espinaca"],
        "vol_litros":  3,
        "factor_riego": 1.7,   # se seca 70% más rápido que tierra abierta
        "desc":        "Maceta pequeña (≤ 3L)",
    },
    "mediana": {
        "no_viables":  ["Maiz", "Trigo"],
        "advertencia": ["Papa", "Zanahoria"],
        "ideales":     ["Lechuga", "Espinaca", "Tomate", "Frijol", "Cebolla", "Remolacha"],
        "vol_litros":  10,
        "factor_riego": 1.35,
        "desc":        "Maceta mediana (5–15L)",
    },
    "grande": {
        "no_viables":  ["Maiz", "Trigo"],
        "advertencia": [],
        "ideales":     ["Lechuga", "Espinaca", "Tomate", "Frijol", "Cebolla", "Remolacha", "Papa", "Zanahoria"],
        "vol_litros":  25,
        "factor_riego": 1.15,
        "desc":        "Maceta grande (20–30L)",
    },
    "jardin": {
        "no_viables":  [],
        "advertencia": [],
        "ideales":     ["Maiz", "Trigo", "Papa", "Zanahoria", "Tomate", "Frijol", "Cebolla", "Remolacha", "Lechuga", "Espinaca"],
        "vol_litros":  None,
        "factor_riego": 1.0,
        "desc":        "Jardín / tierra directa",
    },
}

# Raíz aproximada para cultivos de raíz (restricción de profundidad)
PROFUNDIDAD_RAIZ = {
    "Zanahoria": {"cm": 30, "nota": "La zanahoria necesita mínimo 30 cm de profundidad para desarrollar raíces rectas."},
    "Papa":      {"cm": 30, "nota": "La papa forma tubérculos subterráneos y necesita al menos 30–40 cm de sustrato."},
    "Remolacha": {"cm": 20, "nota": "La remolacha necesita unos 20 cm de profundidad para el bulbo."},
    "Maiz":      {"cm": 40, "nota": "El maíz tiene raíces profundas y además crece muy alto — no es apto para maceta."},
    "Trigo":     {"cm": 40, "nota": "El trigo forma matas grandes y necesita espacio horizontal considerable."},
}

# ──────────────────────────────────────────────────────────
#  TEXTOS DE RECOMENDACIÓN DINÁMICA POR VARIABLE + CULTIVO
#  Se activan cuando la variable está fuera del rango ideal
# ──────────────────────────────────────────────────────────

def recomendacion_temperatura(cultivo, valor, rango):
    bajo, alto = rango
    if valor < bajo:
        diff = round(bajo - valor, 1)
        consejos = {
            "Espinaca":  "La espinaca tolera frío mejor que otros cultivos, pero por debajo de 8°C el crecimiento se detiene casi por completo.",
            "Maiz":      f"El maíz es muy sensible al frío. Con {valor}°C la germinación falla — espera una temporada más cálida.",
            "Tomate":    f"El tomate no tolera temperaturas tan bajas. Por debajo de 10°C los frutos no cuajan. Considera invernadero o esperar.",
            "Lechuga":   "La lechuga aguanta algo de frío, pero puede 'espigarse' (subir a flor sin dar hojas) si el frío es prolongado.",
            "Papa":      "La papa puede tolerar algo de frío, pero heladas dañan el follaje. Protege con acolchado o malla antigranizo.",
        }
        base = consejos.get(cultivo, f"La temperatura actual ({valor}°C) está {diff}°C por debajo del mínimo ideal ({bajo}°C) para {cultivo}. El crecimiento será muy lento o nulo.")
        return f"🌡 Temperatura baja: {base}"
    elif valor > alto:
        diff = round(valor - alto, 1)
        consejos = {
            "Lechuga":   f"Con {valor}°C la lechuga tiende a 'espigar': se estira, amarga y pierde calidad. Busca sombra parcial en las horas más calurosas.",
            "Espinaca":  f"La espinaca entra en estrés térmico por encima de 25°C. Con {valor}°C producirá menos hojas y subirá a flor.",
            "Papa":      f"La papa no forma tubérculos bien con calor excesivo. Temperaturas sobre 30°C bloquean el proceso de tuberización.",
            "Tomate":    f"Con {valor}°C el tomate puede perder flores (caída por calor). El rango ideal es 20–28°C. Riega más frecuente y coloca sombra.",
            "Maiz":      f"El maíz aguanta más calor que otros, pero {valor}°C puede causar estrés hídrico. Aumenta la frecuencia de riego.",
            "Frijol":    f"El frijol pierde flores y vainas con calor sostenido por encima de 30°C.",
        }
        base = consejos.get(cultivo, f"La temperatura actual ({valor}°C) está {diff}°C por encima del máximo ideal ({alto}°C) para {cultivo}. Hay riesgo de estrés térmico.")
        return f"🌡 Temperatura alta: {base}"
    return None

def recomendacion_humedad(cultivo, valor, rango):
    bajo, alto = rango
    if valor < bajo:
        consejos = {
            "Lechuga":  f"La lechuga necesita ambiente húmedo. Con {valor}% de humedad ambiental las hojas se marchitan y pierden tersura.",
            "Espinaca": f"La espinaca prefiere humedad alta. Con {valor}% el crecimiento se resiente — considera riego por aspersión para aumentar la humedad local.",
            "Tomate":   f"Con {valor}% de humedad el tomate puede desarrollar punta negra (blossom end rot) por estrés hídrico intermitente.",
        }
        base = consejos.get(cultivo, f"La humedad ambiental ({valor}%) está por debajo del rango ideal ({bajo}–{alto}%) para {cultivo}. Aumenta la frecuencia de riego y evita exposición directa al viento.")
        return f"💧 Humedad baja: {base}"
    elif valor > alto:
        consejos = {
            "Tomate":    f"Con {valor}% de humedad el tomate es muy propenso a hongos (tizón tardío, botrytis). Mejora la ventilación entre plantas.",
            "Cebolla":   f"La cebolla con {valor}% de humedad desarrolla pudrición del cuello. Asegura buen drenaje y espaciado entre bulbos.",
            "Trigo":     f"Alta humedad en trigo favorece roya y otras enfermedades fúngicas. Evita riego por aspersión en estas condiciones.",
            "Papa":      f"Con {valor}% de humedad la papa es muy susceptible a tizón tardío (Phytophthora). Aplica fungicida preventivo.",
        }
        base = consejos.get(cultivo, f"La humedad ({valor}%) supera el máximo ideal ({alto}%) para {cultivo}. El exceso de humedad favorece hongos y enfermedades.")
        return f"💧 Humedad alta: {base}"
    return None

def recomendacion_ph(cultivo, valor, rango):
    bajo, alto = rango
    if valor < bajo:
        diff = round(bajo - valor, 1)
        consejos = {
            "Papa":    f"La papa en realidad prefiere pH ligeramente ácido (5.0–6.5), pero {valor} puede ser demasiado bajo y reducir disponibilidad de fósforo.",
            "Tomate":  f"Con pH {valor} el tomate no absorbe bien calcio ni magnesio. Aplica cal agrícola para subir el pH al menos a {bajo}.",
            "Maiz":    f"El maíz necesita pH mínimo de {bajo} para absorber correctamente el zinc. Encala el suelo.",
            "default": f"Con pH {valor} el suelo es demasiado ácido para {cultivo} (ideal {bajo}–{alto}). Aplica cal agrícola o dolomita: 100–200 g/m² y riega bien para que se integre.",
        }
        base = consejos.get(cultivo, consejos["default"])
        return f"🧪 pH bajo ({valor}): {base}"
    elif valor > alto:
        consejos = {
            "Papa":    f"La papa prefiere pH ácido (hasta 6.0). Con pH {valor} favoreces la sarna común (Streptomyces). Baja el pH con azufre elemental.",
            "Tomate":  f"Con pH {valor} el tomate no absorbe bien el hierro y manganeso — hojas amarillas entre nervios. Aplica sulfato de hierro.",
            "default": f"El pH actual ({valor}) está por encima del ideal ({alto}) para {cultivo}. Aplica azufre elemental (30–50 g/m²) para acidificar gradualmente.",
        }
        base = consejos.get(cultivo, consejos["default"])
        return f"🧪 pH alto ({valor}): {base}"
    return None

def recomendacion_luz(cultivo, valor, rango):
    bajo, alto = rango
    if valor < bajo:
        consejos = {
            "Maiz":      f"El maíz necesita mínimo {bajo}h de sol directo. Con {valor}h el rendimiento cae drásticamente — es un cultivo de pleno sol.",
            "Tomate":    f"El tomate con menos de {bajo}h de sol produce menos frutos y más follaje. Reubica si es posible a un lugar más soleado.",
            "Remolacha": f"La remolacha necesita buena luz para acumular azúcares en la raíz. Con {valor}h el bulbo será pequeño.",
            "Trigo":     f"El trigo es un cereal que necesita pleno sol. Con {valor}h el espigado y el llenado del grano serán deficientes.",
        }
        base = consejos.get(cultivo, f"{cultivo} recibe solo {valor}h de luz solar directa, por debajo de las {bajo}h ideales. La fotosíntesis insuficiente reduce el crecimiento y producción.")
        return f"☀️ Luz insuficiente: {base}"
    elif valor > alto:
        consejos = {
            "Lechuga":  f"Con {valor}h de sol la lechuga tiende a espigar y amargar. Usa sombrite del 30–40% en las horas de mayor radiación.",
            "Espinaca": f"La espinaca con {valor}h de sol directo entra en estrés y sube a flor rápidamente. Sombra parcial en verano.",
            "Frijol":   f"El frijol con {valor}h puede sufrir quemaduras en hojas. Riega más frecuente para compensar la evapotranspiración.",
        }
        base = consejos.get(cultivo, f"{cultivo} está recibiendo {valor}h de sol, algo más de las {alto}h ideales. Vigila la hidratación — el exceso de sol aumenta la evaporación del sustrato.")
        return f"☀️ Luz excesiva: {base}"
    return None

def recomendacion_precipitacion(cultivo, valor, rango, escenario=None):
    bajo, alto = rango
    if valor < bajo:
        if escenario == "maceta":
            base = f"El riego aplicado ({valor}mm) es insuficiente. Las macetas pierden humedad más rápido que la tierra abierta — riega con mayor frecuencia o en mayor cantidad."
        else:
            base = f"La precipitación/riego ({valor}mm) está por debajo del mínimo necesario ({bajo}mm) para {cultivo}. Complementa con riego manual o por goteo."
        return f"🌧 Agua insuficiente: {base}"
    elif valor > alto:
        consejos = {
            "Cebolla":   "El exceso de agua pudre el bulbo de la cebolla — asegura drenaje rápido y reduce la frecuencia de riego.",
            "Papa":      "Demasiada agua favorece el tizón tardío y la pudrición de tubérculos. Mejora el drenaje con arena gruesa o grava.",
            "Tomate":    "El exceso de agua provoca rajado del fruto y raíces con poco oxígeno. Riega solo cuando la tierra esté seca en la superficie.",
            "Zanahoria": "El exceso de humedad bifurca las raíces y favorece hongos del suelo. Reduce el riego.",
        }
        base = consejos.get(cultivo, f"Con {valor}mm de agua el suelo puede encharcarse para {cultivo} (ideal {bajo}–{alto}mm). Riesgo de asfixia radicular.")
        return f"🌧 Agua excesiva: {base}"
    return None

def recomendacion_altitud(cultivo, valor, rango):
    bajo, alto = rango
    if valor < bajo:
        base = f"La altitud de {valor}m está por debajo del rango óptimo para {cultivo} ({bajo}–{alto}m). A menor altitud las temperaturas tienden a ser más altas y la presión atmosférica mayor."
        return f"⛰ Altitud baja: {base}"
    elif valor > alto:
        consejos = {
            "Tomate":  f"El tomate a {valor}m sufre el frío nocturno propio de las alturas. Los frutos madurarán mucho más lentamente.",
            "Maiz":    f"El maíz a {valor}m tiene ciclos muy largos y menor rendimiento. Prefiere zonas bajas o templadas.",
            "Frijol":  f"El frijol sobre {alto}m puede ver reducida su fijación de nitrógeno por las bajas temperaturas nocturnas.",
        }
        base = consejos.get(cultivo, f"La altitud ({valor}m) supera el rango óptimo para {cultivo} ({bajo}–{alto}m). Las noches frías y la menor densidad del aire pueden desacelerar el cultivo.")
        return f"⛰ Altitud alta: {base}"
    return None

# ──────────────────────────────────────────────────────────
#  DIAGNÓSTICO DE PLAGAS (variable categórica, muy impactante)
# ──────────────────────────────────────────────────────────
PLAGAS_POR_CULTIVO = {
    "Tomate":    "En tomate las plagas más comunes son mosca blanca, ácaros (araña roja) y polilla del tomate. Usa trampas amarillas y aplica jabón potásico o neem si la infestación es leve. Para ataques severos: imidacloprid o spinosad.",
    "Lechuga":   "La lechuga es frecuentemente atacada por pulgones, babosas y minador de hojas. Aplica jabón insecticida en hojas (cara inferior). Usa cerveza en trampas para babosas.",
    "Espinaca":  "La espinaca puede sufrir minador de hojas y pulgones. Retira hojas afectadas manualmente y aplica extracto de ajo o jabón potásico.",
    "Zanahoria": "La zanahoria puede ser atacada por la mosca de la zanahoria (Psila rosae) y nematodos. Usa mallas antinsectos y rota cultivos. Para nematodos: aporta materia orgánica.",
    "Papa":      "La papa sufre comúnmente escarabajo de la papa (Leptinotarsa) y pulgones vectores de virus. Recoge escarabajos a mano o aplica Bacillus thuringiensis. Elimina plantas con síntomas virales.",
    "Tomate":    "El tomate es afectado por mosca blanca, ácaros, trips y Tuta absoluta. Usa trampas, neem y en casos graves insecticida sistémico.",
    "Maiz":      "El maíz puede ser atacado por gusano cogollero (Spodoptera). Aplica Bacillus thuringiensis o spinosad en el cogollo antes de que penetre.",
    "Frijol":    "El frijol sufre de trips, ácaros y barrenadores de vainas. Rota cultivos y aplica extracto de neem preventivamente.",
    "Cebolla":   "La cebolla puede ser afectada por trips de la cebolla y mosca de la cebolla. Usa mallas y aplica insecticida sistémico si el daño es visible en hojas.",
    "Remolacha": "La remolacha es atacada por pulgones y pulguillas. Aplica jabón potásico y asegura buen espaciado para ventilación.",
    "Trigo":     "El trigo puede sufrir pulgones del cereal y chicharritas. En cultivo extensivo aplica insecticida en estadio de espigado si supera el umbral económico.",
}

PLAGAS_GENERALES = "Hay presencia de plagas o enfermedades. Primero identifica el agente: ¿son insectos visibles, manchas en hojas, podredumbre? Si son insectos, inicia con control biológico (jabón potásico, neem). Si son hongos, mejora la ventilación y aplica fungicida cúprico. Aísla plantas afectadas para evitar propagación."

# ──────────────────────────────────────────────────────────
#  FUNCIÓN PRINCIPAL: generar diagnóstico completo
# ──────────────────────────────────────────────────────────
def generar_diagnostico(data: dict, isp: float, estado: str, escenario: str = "jardin") -> dict:
    """
    Analiza cada variable respecto al rango ideal del cultivo
    y devuelve un diagnóstico preciso con recomendaciones específicas.
    """
    cultivo = str(data.get("Tipo_Producto", "")).strip()
    rangos  = RANGOS_IDEALES.get(cultivo, {})
    problemas = []   # lista de strings con los problemas detectados
    positivos = []   # cosas que están bien

    # ── Temperatura ────────────────────────────────────────
    temp = float(data.get("Temperatura", 0))
    r = rangos.get("Temperatura")
    if r:
        msg = recomendacion_temperatura(cultivo, temp, r)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ Temperatura ({temp}°C) dentro del rango óptimo.")

    # ── Humedad ────────────────────────────────────────────
    hum = float(data.get("Humedad", 0))
    r = rangos.get("Humedad")
    if r:
        msg = recomendacion_humedad(cultivo, hum, r)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ Humedad ambiental ({hum}%) en rango ideal.")

    # ── pH ─────────────────────────────────────────────────
    ph = float(data.get("pH_Suelo", 0))
    r = rangos.get("pH_Suelo")
    if r:
        msg = recomendacion_ph(cultivo, ph, r)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ pH del suelo ({ph}) en rango ideal.")

    # ── Luz solar ──────────────────────────────────────────
    luz = float(data.get("Luz_Solar", 0))
    r = rangos.get("Luz_Solar")
    if r:
        msg = recomendacion_luz(cultivo, luz, r)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ Horas de luz solar ({luz}h) adecuadas.")

    # ── Precipitación/Riego ────────────────────────────────
    precip = float(data.get("Precipitacion", 0))
    r = rangos.get("Precipitacion")
    if r:
        msg = recomendacion_precipitacion(cultivo, precip, r, escenario)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ Nivel de agua/precipitación ({precip}mm) adecuado.")

    # ── Altitud ────────────────────────────────────────────
    alt = float(data.get("Altitud", 0))
    r = rangos.get("Altitud")
    if r:
        msg = recomendacion_altitud(cultivo, alt, r)
        if msg: problemas.append(msg)
        else:   positivos.append(f"✅ Altitud ({alt}m) dentro del rango ideal.")

    # ── Irrigación ─────────────────────────────────────────
    irrig = str(data.get("Tipo_Irrigacion", "")).strip().lower()
    if irrig == "gravedad":
        problemas.append("💧 Riego por gravedad: es el método menos eficiente. Distribuye el agua de forma desigual y puede generar zonas de encharcamiento. Considera mejorar a aspersión o goteo si es posible.")
    elif irrig == "aspersion" or irrig == "aspersión":
        if cultivo in ["Tomate", "Papa", "Cebolla"]:
            problemas.append(f"💧 Riego por aspersión en {cultivo}: mojar el follaje de este cultivo favorece hongos (mildiu, botrytis). Prefiere riego por goteo dirigido a la base.")
        else:
            positivos.append("✅ Riego por aspersión adecuado para este cultivo.")
    elif irrig == "goteo":
        positivos.append("✅ Riego por goteo: el más eficiente. Minimiza evaporación y mantiene humedad uniforme en la raíz.")

    # ── Fertilizantes ──────────────────────────────────────
    fert = str(data.get("Uso_Fertilizantes", "")).strip().lower()
    if "quimico" in fert or "químico" in fert:
        problemas.append("🌿 Fertilizantes químicos: si se usan en exceso pueden acidificar el suelo y reducir la actividad microbiana. Considera complementar con materia orgánica para mantener la estructura del suelo.")
    elif "organi" in fert:
        positivos.append("✅ Fertilizantes orgánicos: mejoran la estructura del suelo y la actividad microbiana a largo plazo.")

    # ── Plagas ─────────────────────────────────────────────
    plagas = str(data.get("Presencia_Plagas_Enfermedades", "No")).strip().lower()
    if plagas == "si" or plagas == "sí":
        detalle_plaga = PLAGAS_POR_CULTIVO.get(cultivo, PLAGAS_GENERALES)
        problemas.append(f"🐛 Presencia de plagas/enfermedades detectada: {detalle_plaga}")

    # ── Análisis de maceta (si aplica) ─────────────────────
    alertas_maceta = []
    tam_maceta = str(data.get("tamano_maceta", "jardin")).lower()
    if tam_maceta in MACETA_RESTRICCIONES:
        config = MACETA_RESTRICCIONES[tam_maceta]
        if cultivo in config["no_viables"]:
            alertas_maceta.append({
                "nivel": "critico",
                "texto": f"⚠️ {cultivo} NO es viable en {config['desc']}. {PROFUNDIDAD_RAIZ.get(cultivo, {}).get('nota', 'El sistema radicular de este cultivo supera las dimensiones del contenedor.')}",
            })
        elif cultivo in config["advertencia"]:
            alertas_maceta.append({
                "nivel": "advertencia",
                "texto": f"⚠️ {cultivo} en {config['desc']} es posible pero limitante. El crecimiento será menor que en tierra y necesitarás riego más frecuente (aprox. {config['factor_riego']}× más que en suelo abierto).",
            })
        else:
            alertas_maceta.append({
                "nivel": "ok",
                "texto": f"✅ {cultivo} es adecuado para {config['desc']}.",
            })
        if tam_maceta != "jardin" and config["factor_riego"] > 1.0:
            alertas_maceta.append({
                "nivel": "info",
                "texto": f"💧 Frecuencia de riego en maceta: las macetas pierden agua aprox. {round((config['factor_riego']-1)*100)}% más rápido que el suelo abierto por mayor relación superficie/volumen. Revisa la humedad del sustrato a diario metiendo un dedo — si los primeros 2 cm están secos, riega.",
            })

    # ── Construir resumen ──────────────────────────────────
    n_prob = len(problemas)
    if n_prob == 0:
        resumen = f"Todas las condiciones evaluadas están dentro del rango ideal para {cultivo}. El modelo predice ISP={isp:.2f} ({estado})."
    elif n_prob == 1:
        resumen = f"Se detectó 1 condición fuera del rango ideal para {cultivo} (ISP={isp:.2f}, {estado}). Corregirla debería mejorar el resultado."
    else:
        resumen = f"Se detectaron {n_prob} condiciones fuera del rango ideal para {cultivo} (ISP={isp:.2f}, {estado}). Atender los puntos críticos mejorará significativamente el resultado."

    return {
        "resumen":        resumen,
        "problemas":      problemas,
        "positivos":      positivos,
        "alertas_maceta": alertas_maceta,
        "n_problemas":    n_prob,
    }

# ──────────────────────────────────────────────────────────
#  PASOS DE ACCIÓN según número y tipo de problemas
# ──────────────────────────────────────────────────────────
PASOS_BASE = {
    "Bueno": [
        "✅ Mantén el plan de riego y fertilización actuales.",
        "👁 Monitorea semanalmente en busca de plagas o cambios en el follaje.",
        "📋 Registra la fecha estimada de cosecha según el ciclo del cultivo.",
    ],
    "Regular": [
        "⚠️ Revisa las variables marcadas con ⚠️ en el diagnóstico.",
        "💧 Ajusta el riego — ni exceso ni déficit afectan por igual.",
        "🧪 Si el pH está fuera de rango, corrígelo antes de cualquier otra intervención.",
        "🔍 Inspecciona hojas, tallos y raíces en busca de síntomas de estrés.",
    ],
    "Malo": [
        "🚨 Intervención urgente en las variables críticas del diagnóstico.",
        "🧪 Analiza el suelo (pH, humedad) — son los factores más fáciles de corregir rápido.",
        "🐛 Si hay plagas, aplica control biológico o químico según la severidad.",
        "📞 Si el problema persiste en 72h, consulta a un agrónomo.",
        "🔄 Considera si el cultivo elegido es adecuado para tus condiciones actuales.",
    ],
}

TITULOS = {
    "Bueno":   "✅ Condiciones óptimas para el cultivo",
    "Regular": "⚠️ Condiciones aceptables — hay margen de mejora",
    "Malo":    "🚨 Condiciones críticas — se requiere intervención",
}

# ──────────────────────────────────────────────────────────
#  App Flask
# ──────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

def isp_a_estado(isp: float) -> str:
    if isp >= UMBRALES["Bueno"]:   return "Bueno"
    if isp >= UMBRALES["Regular"]: return "Regular"
    return "Malo"

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status":   "ok",
        "modelo":   "MLP Regressor — ISP_Final",
        "features": features,
        "umbrales": UMBRALES,
        "cultivos_soportados": list(RANGOS_IDEALES.keys()),
    })

@app.route("/info", methods=["GET"])
def get_info():
    return jsonify(info)

@app.route("/encoders", methods=["GET"])
def get_encoders():
    return jsonify({col: list(le.classes_) for col, le in encoders.items()})

@app.route("/simular", methods=["POST"])
def simular():
    """
    POST /simular

    Campos obligatorios:
      Temperatura                    (float)   ej: 22
      Humedad                        (float)   ej: 65
      pH_Suelo                       (float)   ej: 6.5
      Luz_Solar                      (float)   ej: 8.0
      Precipitacion                  (float)   ej: 30
      Altitud                        (float)   ej: 500
      Tipo_Suelo                     (str)     "Mixto" | "Arcilloso" | "Arenoso"
      Tipo_Irrigacion                (str)     "Goteo" | "Aspersion" | "Gravedad"
      Uso_Fertilizantes              (str)     "Organicos" | "Quimicos"
      Presencia_Plagas_Enfermedades  (str)     "No" | "Si"
      Tipo_Producto                  (str)     "Lechuga" | "Tomate" | etc.

    Campos opcionales:
      Mes_Siembra    (int, 1–12)      default: mes actual
      Dia_Anio       (int, 1–365)     default: día actual
      tamano_maceta  (str)            "chica" | "mediana" | "grande" | "jardin"
      escenario      (str)            "maceta" | "jardin"
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Body vacío o no es JSON válido"}), 400

    from datetime import date
    hoy = date.today()
    data.setdefault("Mes_Siembra", hoy.month)
    data.setdefault("Dia_Anio", hoy.timetuple().tm_yday)
    data.setdefault("tamano_maceta", "jardin")
    escenario = str(data.get("escenario", "jardin")).lower()

    # ── Validar campos ─────────────────────────────────────
    campos_cat = ["Tipo_Suelo", "Tipo_Irrigacion", "Uso_Fertilizantes",
                  "Presencia_Plagas_Enfermedades", "Tipo_Producto"]
    campos_num = ["Temperatura", "Humedad", "pH_Suelo", "Luz_Solar",
                  "Precipitacion", "Altitud"]

    errores = [f"Falta campo: '{c}'" for c in campos_cat + campos_num if c not in data]
    if errores:
        return jsonify({"error": "Campos faltantes", "detalle": errores}), 400

    # ── Encodear categóricas ───────────────────────────────
    fila = {}
    for c in campos_cat:
        valor = str(data[c]).strip()
        le    = encoders[c]
        match = next((cls for cls in le.classes_ if cls.strip().lower() == valor.lower()), None)
        if match is None:
            return jsonify({"error": f"Valor inválido '{c}': '{valor}'", "valores_validos": list(le.classes_)}), 400
        fila[c] = int(le.transform([match])[0])

    # ── Numéricas ──────────────────────────────────────────
    for c in campos_num:
        try:
            fila[c] = float(data[c])
        except (ValueError, TypeError):
            return jsonify({"error": f"'{c}' debe ser numérico, recibido: {data[c]}"}), 400

    fila["Mes_Siembra"] = int(data["Mes_Siembra"])
    fila["Dia_Anio"]    = int(data["Dia_Anio"])

    # ── Predecir ───────────────────────────────────────────
    df_input = pd.DataFrame([fila], columns=features)
    isp      = float(np.clip(modelo.predict(scaler.transform(df_input))[0], 0.0, 1.0))
    estado   = isp_a_estado(isp)

    # ── Diagnóstico dinámico ───────────────────────────────
    diag = generar_diagnostico(data, isp, estado, escenario)

    # ── Respuesta ──────────────────────────────────────────
    return jsonify({
        # Predicción principal
        "isp_final":   round(isp, 4),
        "estado":      estado,
        "titulo":      TITULOS[estado],
        "mensaje":     diag["resumen"],

        # Diagnóstico variable por variable
        "diagnostico": {
            "problemas":      diag["problemas"],
            "positivos":      diag["positivos"],
            "n_problemas":    diag["n_problemas"],
            "alertas_maceta": diag["alertas_maceta"],
        },

        # Acciones
        "pasos": PASOS_BASE[estado],

        # Compatibilidad con frontend anterior
        "humedadFinal":  round(isp * 100, 1),
        "recomendacion": diag["resumen"],

        # Detalle técnico
        "detalle": {
            "cultivo": str(data.get("Tipo_Producto", "")),
            "isp_raw": round(isp, 6),
            "umbrales": UMBRALES,
        },
    })

if __name__ == "__main__":
    print("=" * 55)
    print("  API Flask — Simulador de Hortalizas v2")
    print("  POST /simular  →  predicción + diagnóstico dinámico")
    print("  http://0.0.0.0:5000")
    print("=" * 55)
    app.run(host="0.0.0.0", port=5000, debug=True)