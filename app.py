import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================

st.set_page_config(
    page_title="Expedición México",
    page_icon="🧭",
    layout="wide"
)

# =====================================================
# IDENTIDAD VISUAL
# UNIVERSIDAD NACIONAL ROSARIO CASTELLANOS
# =====================================================

st.markdown("""
<style>

/* ===== APP ===== */

.stApp {
    background: #F7F8FA;
    color: #1E293B;
}

/* ===== CONTENEDOR ===== */

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ===== TITULOS ===== */

h1 {
    color: #7A003C !important;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
}

h2 {
    color: #7A003C !important;
    font-weight: 700 !important;
    margin-top: 2rem !important;
}

h3 {
    color: #9A1B5A !important;
    font-weight: 700 !important;
}

/* ===== TEXTOS ===== */

p, div, span, li, label {
    color: #1E293B !important;
    font-size: 1rem;
}

/* ===== TARJETAS ===== */

.card {
    background: white;
    padding: 2rem;
    border-radius: 22px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 10px 30px rgba(0,0,0,.05);
    margin-bottom: 1.5rem;
}

/* ===== BOTONES ===== */

.stButton button {
    background: #7A003C !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    width: 100%;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 25px rgba(122,0,60,.18);
}

.stButton button:hover {
    background: #9A1B5A !important;
}

/* ===== INPUT ===== */

.stTextInput input {
    border-radius: 14px !important;
    border: 2px solid #D1D5DB !important;
    padding: .9rem !important;
}

/* ===== RADIO ===== */

.stRadio > div {
    background: white;
    padding: 1.2rem;
    border-radius: 16px;
    border: 1px solid #E5E7EB;
    margin-bottom: 1rem;
}

/* ===== MÉTRICAS ===== */

[data-testid="stMetric"] {
    background: white;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 6px 18px rgba(0,0,0,.05);
}

/* ===== INTERPRETACIONES ===== */

.interpretacion {
    background: white;
    border-radius: 18px;
    padding: 1.5rem;
    border: 1px solid #E5E7EB;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(0,0,0,.04);
}

/* ===== ALERTAS ===== */

.stAlert {
    border-radius: 16px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FUNCIONES
# =====================================================

def nivel(valor):

    if valor >= 75:
        return "alto"
    elif valor >= 45:
        return "moderado"
    else:
        return "bajo"


def interpretar_variable(nombre, valor):

    interpretaciones = {

        "riesgo": {
            "alto":"Tiendes a aceptar incertidumbre y asumir escenarios con mayor exposición a pérdida o recompensa.",
            "moderado":"Evalúas escenarios antes de actuar y equilibras riesgo con estabilidad.",
            "bajo":"Prefieres minimizar incertidumbre y mantener control sobre posibles pérdidas."
        },

        "exploracion": {
            "alto":"Mostraste apertura a descubrir nuevas rutas y explorar escenarios poco estructurados.",
            "moderado":"Tu exploración se mantuvo equilibrada entre seguridad y descubrimiento.",
            "bajo":"Preferiste caminos conocidos y contextos más estables."
        },

        "precision": {
            "alto":"Tus decisiones mostraron atención al detalle y análisis cuidadoso.",
            "moderado":"Equilibraste rapidez con precisión al tomar decisiones.",
            "bajo":"Tus decisiones priorizaron velocidad o exploración más que exactitud."
        },

        "adaptabilidad": {
            "alto":"Mostraste facilidad para ajustarte a contextos cambiantes.",
            "moderado":"Tu adaptación fue funcional en escenarios moderadamente inciertos.",
            "bajo":"Mostraste preferencia por estabilidad y estructuras claras."
        },

        "impulsividad": {
            "alto":"Tendiste a actuar rápidamente bajo presión o incertidumbre.",
            "moderado":"Combinaste rapidez con momentos de reflexión.",
            "bajo":"Preferiste analizar antes de actuar."
        },

        "planeacion": {
            "alto":"Tus decisiones mostraron orientación fuerte hacia organización y previsión.",
            "moderado":"Tu planeación fue funcional sin perder flexibilidad.",
            "bajo":"Tus decisiones priorizaron acción inmediata sobre planificación extensa."
        },

        "cooperacion": {
            "alto":"Mostraste disposición alta para integrar perspectivas grupales.",
            "moderado":"Tu cooperación fue equilibrada con autonomía.",
            "bajo":"Tendiste a priorizar decisiones individuales."
        }

    }

    return interpretaciones[nombre][nivel(valor)]


def crear_base():

    np.random.seed(42)

    perfiles = []

    configuraciones = [

        [25,40,88,55,25,88,65],
        [75,88,68,82,55,70,72],
        [55,65,78,88,42,75,82],
        [85,58,52,48,88,42,50]

    ]

    for perfil in configuraciones:

        for _ in range(45):

            perfiles.append({

                "riesgo": np.random.normal(perfil[0],8),
                "exploracion": np.random.normal(perfil[1],8),
                "precision": np.random.normal(perfil[2],7),
                "adaptabilidad": np.random.normal(perfil[3],7),
                "impulsividad": np.random.normal(perfil[4],8),
                "planeacion": np.random.normal(perfil[5],7),
                "cooperacion": np.random.normal(perfil[6],8)

            })

    return pd.DataFrame(perfiles)


def perfil_cluster(c):

    perfiles = {

        0:"Cauteloso Analítico",
        1:"Explorador Estratégico",
        2:"Adaptativo Resiliente",
        3:"Impulsivo Bajo Presión"

    }

    return perfiles[c]

# =====================================================
# BASE
# =====================================================

base = crear_base()

variables = [
    "riesgo",
    "exploracion",
    "precision",
    "adaptabilidad",
    "impulsividad",
    "planeacion",
    "cooperacion"
]

# =====================================================
# MODELOS
# =====================================================

scaler = StandardScaler()

X = scaler.fit_transform(base[variables])

pca = PCA(n_components=2)

componentes = pca.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(componentes)

base["PCA1"] = componentes[:,0]
base["PCA2"] = componentes[:,1]
base["Cluster"] = clusters

# =====================================================
# LANDING
# =====================================================

st.title("🧭 Expedición México")

st.subheader("""
Descubre cómo tomas decisiones bajo incertidumbre mientras recorres distintos escenarios de México.
""")

st.markdown("""
<div class="card">

<h3>¿Qué es esta experiencia?</h3>

<p>
Expedición México es una simulación interactiva construida a partir de modelos de ciencia de datos y análisis conductual.
</p>

<p>
La experiencia toma inspiración metodológica de análisis previos realizados en Python sobre:
</p>

<ul>
<li>estrés académico,</li>
<li>presión bajo incertidumbre,</li>
<li>adaptabilidad,</li>
<li>y patrones de comportamiento.</li>
</ul>

<p>
A partir de esos análisis se desarrollaron:
</p>

<ul>
<li>clustering conductual (KMeans),</li>
<li>reducción dimensional mediante PCA,</li>
<li>análisis multivariado,</li>
<li>y simulaciones de perfiles de decisión.</li>
</ul>

<p>
Tus decisiones dentro de la expedición generan telemetría conductual que posteriormente es comparada con distintos perfiles simulados.
</p>

</div>
""", unsafe_allow_html=True)

st.info("""
La experiencia NO representa una evaluación clínica ni diagnóstica.
Es una simulación educativa de ciencia de datos aplicada a toma de decisiones bajo incertidumbre.
""")

st.divider()

# =====================================================
# INICIO
# =====================================================

st.header("Inicio de la expedición")

nombre = st.text_input(
    "Nombre o alias del participante"
)

if st.button("Comenzar expedición"):

    st.session_state["iniciado"] = True

# =====================================================
# ESCENARIOS
# =====================================================

if st.session_state.get("iniciado"):

    st.success(f"Bienvenido/a, {nombre if nombre else 'participante'}.")

    with st.form("formulario"):

        st.header("Iniciamos la aventura")

        r1 = st.radio(
            "Tu viaje inicia en la zona centro del país. ¿Cómo te gustaría avanzar?",
            [
                "Elegir una ruta segura que ya conoces",
                "Elegir una ruta equilibrada que permita explorar sin demasiada exposición",
                "Elegir una ruta poco conocida con alta recompensa cultural"
            ]
        )

        st.header("Siguiente parada — Oaxaca")

        r2 = st.radio(
            "¿Qué harías con tus viáticos?",
            [
                "Conservar recursos para reducir pérdidas",
                "Invertir parcialmente y mantener reservas",
                "Usar gran parte de recursos para acelerar el viaje"
            ]
        )

        st.header("Vámonos a Yucatán")

        r3 = st.radio(
            "Descubres información incompleta sobre una ruta histórica.",
            [
                "Esperar más información antes de actuar",
                "Avanzar y ajustar sobre la marcha",
                "Actuar rápidamente para aprovechar la oportunidad"
            ]
        )

        st.header("Prepárate para las turbulencias")

        r4 = st.radio(
            "Tu equipo no está de acuerdo sobre el siguiente destino.",
            [
                "Mantener el plan original",
                "Negociar una solución intermedia",
                "Cambiar completamente la estrategia"
            ]
        )

        st.header("Regreso a Ciudad de México")

        r5 = st.radio(
            "Llegas justo antes de que cierre el metro.",
            [
                "Corres para alcanzar el último tren",
                "Buscas rutas alternas antes de decidir",
                "Pides transporte privado aunque no estuviera planeado"
            ]
        )

        enviar = st.form_submit_button(
            "Generar reporte conductual"
        )

    if enviar:

        usuario = {
            "riesgo":50,
            "exploracion":50,
            "precision":50,
            "adaptabilidad":50,
            "impulsividad":50,
            "planeacion":50,
            "cooperacion":50
        }

        if "segura" in r1:
            usuario["riesgo"] -= 20
            usuario["precision"] += 20
            usuario["planeacion"] += 20

        elif "equilibrada" in r1:
            usuario["adaptabilidad"] += 15
            usuario["exploracion"] += 10

        else:
            usuario["riesgo"] += 30
            usuario["exploracion"] += 30
            usuario["impulsividad"] += 10

        if "Conservar" in r2:
            usuario["planeacion"] += 20
            usuario["precision"] += 15

        elif "parcialmente" in r2:
            usuario["adaptabilidad"] += 15

        else:
            usuario["riesgo"] += 15
            usuario["impulsividad"] += 20

        if "Esperar" in r3:
            usuario["precision"] += 15
            usuario["impulsividad"] -= 10

        elif "ajustar" in r3:
            usuario["adaptabilidad"] += 20

        else:
            usuario["riesgo"] += 20
            usuario["impulsividad"] += 25

        if "Mantener" in r4:
            usuario["cooperacion"] -= 15

        elif "Negociar" in r4:
            usuario["cooperacion"] += 25
            usuario["adaptabilidad"] += 10

        else:
            usuario["riesgo"] += 10

        if "Corres" in r5:
            usuario["impulsividad"] += 15
            usuario["riesgo"] += 10

        elif "Buscas" in r5:
            usuario["planeacion"] += 15
            usuario["precision"] += 15

        else:
            usuario["adaptabilidad"] += 15

        usuario_df = pd.DataFrame([usuario])

        usuario_scaled = scaler.transform(
            usuario_df[variables]
        )

        usuario_pca = pca.transform(usuario_scaled)

        cluster = int(
            kmeans.predict(usuario_pca)[0]
        )

        perfil = perfil_cluster(cluster)

        st.session_state["usuario"] = usuario
        st.session_state["perfil"] = perfil
        st.session_state["cluster"] = cluster
        st.session_state["usuario_pca"] = usuario_pca

# =====================================================
# DASHBOARD
# =====================================================

if "perfil" in st.session_state:

    usuario = st.session_state["usuario"]

    st.divider()

    st.header("📊 Reporte conductual de tu expedición")

    cluster_nombres = {
        0:"Cluster Analítico",
        1:"Cluster Explorador",
        2:"Cluster Adaptativo",
        3:"Cluster Reactivo"
    }

    col1, col2 = st.columns(2)

    col1.metric(
        "Perfil detectado",
        st.session_state["perfil"]
    )

    col2.metric(
        "Grupo conductual",
        cluster_nombres[st.session_state["cluster"]]
    )

    # =====================================================
    # MÉTRICAS
    # =====================================================

    st.header("1. Interpretación de tus métricas conductuales")

    for var in variables:

        st.markdown(f"""
        <div class="interpretacion">

        <h3>{var.capitalize()} — Nivel {nivel(usuario[var])}</h3>

        <p><strong>Puntaje:</strong> {round(usuario[var],1)}</p>

        <p>{interpretar_variable(var, usuario[var])}</p>

        </div>
        """, unsafe_allow_html=True)

# =====================================================
# ÍNDICES
# =====================================================

    st.header("2. Índices conductuales integrados")

    indice_exploracion = min(100, round(
        (
            usuario["riesgo"] * 0.35 +
            usuario["exploracion"] * 0.40 +
            usuario["adaptabilidad"] * 0.25
        ),1
    ))

    indice_estabilidad = min(100, round(
        (
            usuario["planeacion"] * 0.40 +
            usuario["precision"] * 0.35 +
            usuario["cooperacion"] * 0.25
        ),1
    ))

    indice_presion = min(100, round(
        (
            usuario["impulsividad"] * 0.45 +
            usuario["riesgo"] * 0.30 +
            usuario["adaptabilidad"] * 0.25
        ),1
    ))

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Exploración Estratégica",
            f"{indice_exploracion}/100"
        )

        st.progress(indice_exploracion / 100)

    with col2:

        st.metric(
            "Estabilidad Operativa",
            f"{indice_estabilidad}/100"
        )

        st.progress(indice_estabilidad / 100)

    with col3:

        st.metric(
            "Respuesta Bajo Presión",
            f"{indice_presion}/100"
        )

        st.progress(indice_presion / 100)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Exploración Estratégica",
            f"{indice_exploracion}/100"
        )

        st.progress(indice_exploracion/100)

    with col2:

        st.metric(
            "Estabilidad Operativa",
            f"{indice_estabilidad}/100"
        )

        st.progress(indice_estabilidad/100)

    with col3:

        st.metric(
            "Respuesta Bajo Presión",
            f"{indice_presion}/100"
        )

        st.progress(indice_presion/100)

    # =====================================================
    # PCA PREMIUM
    # =====================================================

    st.header("3. Tu posición en el mapa conductual")

    colores = {
        0:"#7A003C",
        1:"#2563EB",
        2:"#16A34A",
        3:"#EA580C"
    }

    nombres = {
        0:"Cauteloso Analítico",
        1:"Explorador Estratégico",
        2:"Adaptativo Resiliente",
        3:"Impulsivo Bajo Presión"
    }

    fig, ax = plt.subplots(figsize=(11,7))

    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("white")

    for c in sorted(base["Cluster"].unique()):

        cluster_data = base[
            base["Cluster"] == c
        ]

        ax.scatter(
            cluster_data["PCA1"],
            cluster_data["PCA2"],
            color=colores[c],
            alpha=0.35,
            s=45,
            label=nombres[c]
        )

    for c in sorted(base["Cluster"].unique()):

        centroide_x = base[
            base["Cluster"] == c
        ]["PCA1"].mean()

        centroide_y = base[
            base["Cluster"] == c
        ]["PCA2"].mean()

        ax.scatter(
            centroide_x,
            centroide_y,
            color=colores[c],
            s=320,
            edgecolor="black",
            linewidth=2
        )

        ax.text(
            centroide_x,
            centroide_y + 0.25,
            nombres[c],
            fontsize=9,
            ha="center",
            weight="bold"
        )

    ax.scatter(
        st.session_state["usuario_pca"][0][0],
        st.session_state["usuario_pca"][0][1],
        color="red",
        s=450,
        marker="X",
        edgecolor="black",
        linewidth=2,
        label="Tu perfil"
    )

    ax.set_title(
        "Mapa de similitud conductual",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(alpha=0.15)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    st.pyplot(fig)

   # =====================================================
# LECTURA APLICADA
# =====================================================

st.header("4. Lectura aplicada de tu expedición")

if st.session_state["perfil"] == "Cauteloso Analítico":

    significado = """
    Durante la expedición mostraste un patrón de decisión orientado hacia la estabilidad, el análisis y la reducción de incertidumbre. Tus respuestas sugieren preferencia por escenarios estructurados, previsibles y con mayor control sobre las consecuencias. Este perfil suele evaluar cuidadosamente los riesgos antes de actuar y prioriza la seguridad operativa sobre la improvisación.
    """

    patrones = """
    A lo largo del recorrido aparecieron conductas asociadas con planeación, precisión y control de recursos. Tus decisiones mostraron tendencia a evitar pérdidas innecesarias, buscar información antes de actuar y mantener estabilidad aun bajo presión temporal. También se observó menor impulsividad frente a escenarios inciertos.
    """

    implicaciones = """
    En contextos universitarios, este patrón puede favorecer organización académica, cumplimiento de objetivos y manejo estructurado de responsabilidades. En la vida cotidiana, suele relacionarse con personas que buscan estabilidad y control antes de tomar decisiones importantes. Sin embargo, en ciertos contextos podría existir dificultad para actuar rápidamente ante escenarios altamente cambiantes o ambiguos.
    """

elif st.session_state["perfil"] == "Explorador Estratégico":

    significado = """
    Tu expedición mostró un perfil orientado hacia exploración, apertura y búsqueda de oportunidades. Tus decisiones reflejaron comodidad frente a incertidumbre moderada y disposición a asumir riesgos cuando existe posibilidad de recompensa o descubrimiento. Este perfil combina curiosidad con capacidad de adaptación.
    """

    patrones = """
    Durante el recorrido aparecieron patrones asociados con exploración conductual, flexibilidad y tolerancia al riesgo. Mostraste tendencia a avanzar aun con información incompleta y a priorizar experiencias nuevas sobre estabilidad absoluta. También se observó rapidez para ajustarte a cambios de contexto.
    """

    implicaciones = """
    En la vida universitaria este patrón puede favorecer creatividad, aprendizaje autónomo y facilidad para adaptarse a entornos dinámicos. En escenarios personales y profesionales suele relacionarse con iniciativa, innovación y apertura a nuevas experiencias. No obstante, una exploración excesiva podría llevar ocasionalmente a asumir riesgos sin suficiente planeación.
    """

elif st.session_state["perfil"] == "Adaptativo Resiliente":

    significado = """
    Tus decisiones sugirieron un perfil equilibrado entre análisis, adaptación y flexibilidad. A lo largo de la expedición mostraste capacidad para ajustarte a situaciones cambiantes sin perder completamente estabilidad o estructura. Este perfil suele responder funcionalmente frente a escenarios ambiguos o inciertos.
    """

    patrones = """
    Se observaron conductas relacionadas con adaptación progresiva, negociación y ajuste contextual. Tus decisiones no se mantuvieron rígidas frente a los cambios, pero tampoco mostraron impulsividad extrema. El patrón general indica capacidad para reorganizar estrategias conforme aparecen nuevos elementos dentro del entorno.
    """

    implicaciones = """
    En contextos universitarios este perfil puede favorecer trabajo colaborativo, resolución de problemas y adaptación a carga académica variable. En la vida cotidiana suele asociarse con personas capaces de enfrentar cambios sin perder funcionalidad general. También puede facilitar manejo emocional más flexible frente a presión o incertidumbre.
    """

else:

    significado = """
    Durante la expedición apareció un patrón de decisión más reactivo e inmediato frente a presión o incertidumbre. Tus respuestas mostraron tendencia a actuar rápidamente cuando el entorno exige resolver situaciones en poco tiempo. Este perfil suele priorizar acción inmediata sobre análisis prolongado.
    """

    patrones = """
    A lo largo del recorrido se observaron decisiones asociadas con impulsividad, rapidez de respuesta y tolerancia alta a escenarios inciertos. En distintos momentos se priorizó avanzar rápidamente o resolver situaciones de forma inmediata aun con información incompleta.
    """

    implicaciones = """
    En escenarios universitarios este patrón puede favorecer capacidad de reacción rápida y respuesta eficiente ante presión temporal. En la vida cotidiana puede relacionarse con iniciativa y acción inmediata frente a problemas. Sin embargo, cuando las decisiones requieren análisis profundo, podría existir tendencia a actuar antes de evaluar completamente las consecuencias.
    """

st.subheader("¿Qué significa tu perfil?")

st.write(significado)

st.subheader("Patrones de decisión observados")

st.write(patrones)

st.subheader("¿Qué implicaciones puede tener esto?")

st.write(implicaciones)

st.info("""
Este reporte NO representa un diagnóstico psicológico ni clínico.

La experiencia transforma patrones estadísticos observados en simulaciones narrativas de toma de decisiones bajo incertidumbre. Su propósito es educativo y exploratorio dentro de un contexto de ciencia de datos aplicada al comportamiento humano.
""")
