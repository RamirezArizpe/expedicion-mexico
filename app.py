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
    page_title="Expedición México · Simulación Conductual",
    page_icon="🧭",
    layout="wide"
)

# =====================================================
# IDENTIDAD VISUAL: UNIVERSIDAD NACIONAL ROSARIO CASTELLANOS
# =====================================================
st.markdown("""
<style>
/* App background y tipografía */
.stApp {
    background: #F7F8FA;
    color: #1E293B;
}

.block-container {
    max-width: 1080px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Encabezados institucionales */
h1 {
    color: #7A003C !important;
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem !important;
}

h2 {
    color: #7A003C !important;
    font-weight: 700 !important;
    margin-top: 1.8rem !important;
    font-size: 1.5rem !important;
}

h3 {
    color: #9A1B5A !important;
    font-weight: 700 !important;
}

/* Tarjetas y Contenedores */
.card {
    background: #FFFFFF;
    padding: 1.8rem 2rem;
    border-radius: 20px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 10px 25px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}

.interpretacion {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    border: 1px solid #E5E7EB;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
}

/* Botones institucionales */
.stButton button {
    background: #7A003C !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.5rem !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(122,0,60,0.18);
    transition: all 0.2s ease;
}

.stButton button:hover {
    background: #9A1B5A !important;
    transform: translateY(-1px);
}

/* Opciones de Radio */
.stRadio > div {
    background: #FFFFFF;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
    margin-bottom: 0.8rem;
}

/* Métricas */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA SCIENCE & CACHING (Optimizador de rendimiento)
# =====================================================
VARIABLES = [
    "riesgo", "exploracion", "precision", 
    "adaptabilidad", "impulsividad", "planeacion", "cooperacion"
]

@st.cache_data
def inicializar_modelos():
    np.random.seed(42)
    perfiles = []
    
    # 4 Configuraciones base para la simulación
    configuraciones = [
        [25, 40, 88, 55, 25, 88, 65],  # Cauteloso
        [75, 88, 68, 82, 55, 70, 72],  # Explorador
        [55, 65, 78, 88, 42, 75, 82],  # Adaptativo
        [85, 58, 52, 48, 88, 42, 50]   # Reactivo/Impulsivo
    ]
    
    for perfil in configuraciones:
        for _ in range(45):
            perfiles.append({
                "riesgo": np.random.normal(perfil[0], 8),
                "exploracion": np.random.normal(perfil[1], 8),
                "precision": np.random.normal(perfil[2], 7),
                "adaptabilidad": np.random.normal(perfil[3], 7),
                "impulsividad": np.random.normal(perfil[4], 8),
                "planeacion": np.random.normal(perfil[5], 7),
                "cooperacion": np.random.normal(perfil[6], 8)
            })
            
    df_base = pd.DataFrame(perfiles)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(df_base[VARIABLES])
    
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(componentes)
    
    df_base["PCA1"] = componentes[:, 0]
    df_base["PCA2"] = componentes[:, 1]
    df_base["Cluster"] = clusters
    
    return df_base, scaler, pca, kmeans

base, scaler, pca, kmeans = inicializar_modelos()

# =====================================================
# REGLAS E INTERPRETACIONES CONDUCTUALES
# =====================================================
def nivel(valor):
    if valor >= 75:
        return "alto"
    elif valor >= 45:
        return "moderado"
    return "bajo"

def interpretar_variable(nombre, valor):
    interpretaciones = {
        "riesgo": {
            "alto": "Tiendes a aceptar incertidumbre y asumir escenarios con mayor exposición a pérdida o recompensa.",
            "moderado": "Evalúas escenarios antes de actuar y equilibras riesgo con estabilidad.",
            "bajo": "Prefieres minimizar incertidumbre y mantener control sobre posibles pérdidas."
        },
        "exploracion": {
            "alto": "Mostraste apertura a descubrir nuevas rutas y explorar escenarios poco estructurados.",
            "moderado": "Tu exploración se mantuvo equilibrada entre seguridad y descubrimiento.",
            "bajo": "Preferiste caminos conocidos y contextos más estables."
        },
        "precision": {
            "alto": "Tus decisiones mostraron atención al detalle y análisis cuidadoso.",
            "moderado": "Equilibraste rapidez con precisión al tomar decisiones.",
            "bajo": "Tus decisiones priorizaron velocidad o exploración más que exactitud."
        },
        "adaptabilidad": {
            "alto": "Mostraste facilidad para ajustarte a contextos cambiantes.",
            "moderado": "Tu adaptación fue funcional en escenarios moderadamente inciertos.",
            "bajo": "Mostraste preferencia por estabilidad y estructuras claras."
        },
        "impulsividad": {
            "alto": "Tendiste a actuar rápidamente bajo presión o incertidumbre.",
            "moderado": "Combinaste rapidez con momentos de reflexión.",
            "bajo": "Preferiste analizar antes de actuar."
        },
        "planeacion": {
            "alto": "Tus decisiones mostraron orientación fuerte hacia organización y previsión.",
            "moderado": "Tu planeación fue funcional sin perder flexibilidad.",
            "bajo": "Tus decisiones priorizaron acción inmediata sobre planificación extensa."
        },
        "cooperacion": {
            "alto": "Mostraste disposición alta para integrar perspectivas grupales.",
            "moderado": "Tu cooperación fue equilibrada con autonomía.",
            "bajo": "Tendiste a priorizar decisiones individuales."
        }
    }
    return interpretaciones[nombre][nivel(valor)]

def perfil_cluster(c):
    perfiles = {
        0: "Cauteloso Analítico",
        1: "Explorador Estratégico",
        2: "Adaptativo Resiliente",
        3: "Impulsivo Bajo Presión"
    }
    return perfiles[c]

# =====================================================
# LANDING Y BIENVENIDA
# =====================================================
st.title("🧭 Expedición México")
st.markdown("##### Descubre cómo tomas decisiones bajo incertidumbre mientras recorres distintos escenarios de México.")

with st.expander("ℹ️ ¿Qué es esta experiencia y cuál es su sustento?", expanded=False):
    st.markdown("""
    **Expedición México** es una simulación interactiva construida a partir de modelos de ciencia de datos y análisis conductual:
    * **Clustering Conductual:** Agrupamiento no supervisado mediante *K-Means* ($k=4$).
    * **Reducción Dimensional:** Proyección en espacio vectorial bidimensional vía *PCA*.
    * **Análisis Multivariado:** Normalización z-score sobre 7 dimensiones de toma de decisión.
    
    *Nota metodológica: La experiencia NO representa una evaluación clínica ni diagnóstica. Es un simulador educativo de toma de decisiones.*
    """)

st.divider()

# =====================================================
# CONTROL DE ESTADO DE SESIÓN
# =====================================================
if "iniciado" not in st.session_state:
    st.session_state["iniciado"] = False

if not st.session_state["iniciado"]:
    st.header("Inicio de la expedición")
    nombre_input = st.text_input("Nombre o alias del participante", placeholder="Ej. Alex R.")
    
    if st.button("Comenzar expedición"):
        st.session_state["nombre"] = nombre_input.strip() if nombre_input.strip() else "Participante"
        st.session_state["iniciado"] = True
        st.rerun()

# =====================================================
# FORMULARIO DE ESCENARIOS (NARRATIVA INMERSIVA)
# =====================================================
if st.session_state["iniciado"]:
    st.success(f"Bienvenido/a a la expedición, **{st.session_state.get('nombre', 'Participante')}**.")

    with st.form("formulario_expedicion"):
        
        # Parada 1
        st.header("1. El Punto de Partida — Altiplano Central")
        r1 = st.radio(
            "La expedición parte al amanecer. Tienes en tus manos una carta de ruta validada y el rumor de un sendero poco transitado con vestigios arqueológicos intactos. ¿Cuál es tu primera decisión?",
            [
                "Avanzar estrictamente por la carretera federal planificada; la prioridad es cumplir el itinerario sin contratiempos.",
                "Tomar la ruta escénica secundaria: permite explorar el paisaje sin alejarse demasiado del camino principal.",
                "Internarse de inmediato por la brecha comunitaria; los mayores descubrimientos están fuera del mapa oficial."
            ],
            index=0
        )

        # Parada 2
        st.header("2. Gestión en la Sierra — Valles Centrales de Oaxaca")
        r2 = st.radio(
            "En la sierra oaxaqueña surge la opción de alquilar transporte de montaña y guías locales para llegar a una reserva remota antes de que llueva. El costo es alto frente a tus fondos. ¿Cómo administras tus recursos?",
            [
                "Conservar el fondo intacto y explorar únicamente las rutas a pie cubiertas por el presupuesto asignado.",
                "Invertir parcialmente en una visita corta para mantener un fondo de reserva ante cualquier eventualidad.",
                "Desembolsar lo necesario sin titubear; la oportunidad de cruzar la sierra antes de la tormenta justifica el gasto."
            ],
            index=0
        )

        # Parada 3
        st.header("3. Ambigüedad en la Selva — Península de Yucatán")
        r3 = st.radio(
            "En la selva yucateca, el GPS pierde enlace satelital justo ante una bifurcación no documentada. Quedan 90 minutos de luz natural para ubicar el campamento. ¿Cómo respondes ante la falta de datos?",
            [
                "Esperar en el lugar, montar campamento base de resguardo y buscar señal con calma a primera hora del día.",
                "Avanzar con brújula manual y marcas de terreno, ajustando el rumbo paso a paso sobre la marcha.",
                "Actuar rápidamente y seguir el cauce del arroyo más cercano para avanzar antes del anochecer."
            ],
            index=0
        )

        # Parada 4
        st.header("4. Conflicto en el Campamento — Noche de Tormenta")
        r4 = st.radio(
            "El clima empeora y el equipo se divide: una mitad exige refugiarse de inmediato y la otra insiste en mantener el ascenso hacia el observatorio. Debes destrabar la discusión. ¿Qué postura adoptas?",
            [
                "Mantener el plan original aprobado al inicio y pedir a toda la tripulación apegarse a la disciplina.",
                "Negociar una solución intermedia: una pausa técnica breve para dialogar y fijar un punto medio de resguardo.",
                "Cambiar completamente la estrategia y ordenar una retirada táctica hacia una ruta alterna no explorada."
            ],
            index=0
        )

        # Parada 5
        st.header("5. Carrera Contra el Tiempo — Retorno a la Terminal CDMX")
        r5 = st.radio(
            "De regreso en la capital, llegas a la estación con las muestras y el equipo justo cuando anuncian el silbatazo del último convoy del metro. Hay aglomeración en los torniquetes. ¿Qué haces?",
            [
                "Corres a máxima velocidad para superar el tumulto y alcanzar a subirte antes del cierre de puertas.",
                "Buscas rutas alternas en el mapa metropolitano para combinar trolebús o metrobús de forma más ordenada.",
                "Pides transporte privado por aplicación móvil aunque no estuviera planeado, garantizando resguardar el material."
            ],
            index=0
        )

        enviar = st.form_submit_button("Generar reporte conductual")

    # =====================================================
    # MOTOR CONDUCTUAL (Mapeo de Variables)
    # =====================================================
    if enviar:
        usuario = {var: 50.0 for var in VARIABLES}

        # R1: Altiplano Central
        if "carretera federal" in r1:
            usuario["riesgo"] -= 20
            usuario["precision"] += 20
            usuario["planeacion"] += 20
        elif "escénica secundaria" in r1:
            usuario["adaptabilidad"] += 15
            usuario["exploracion"] += 10
        else:
            usuario["riesgo"] += 30
            usuario["exploracion"] += 30
            usuario["impulsividad"] += 10

        # R2: Oaxaca
        if "Conservar el fondo" in r2:
            usuario["planeacion"] += 20
            usuario["precision"] += 15
        elif "Invertir parcialmente" in r2:
            usuario["adaptabilidad"] += 15
        else:
            usuario["riesgo"] += 15
            usuario["impulsividad"] += 20

        # R3: Yucatán
        if "Esperar en el lugar" in r3:
            usuario["precision"] += 15
            usuario["impulsividad"] -= 10
        elif "Avanzar con brújula" in r3:
            usuario["adaptabilidad"] += 20
        else:
            usuario["riesgo"] += 20
            usuario["impulsividad"] += 25

        # R4: Tormenta
        if "Mantener el plan" in r4:
            usuario["cooperacion"] -= 15
        elif "Negociar una solución" in r4:
            usuario["cooperacion"] += 25
            usuario["adaptabilidad"] += 10
        else:
            usuario["riesgo"] += 10

        # R5: Terminal CDMX
        if "Corres a máxima" in r5:
            usuario["impulsividad"] += 15
            usuario["riesgo"] += 10
        elif "Buscas rutas" in r5:
            usuario["planeacion"] += 15
            usuario["precision"] += 15
        else:
            usuario["adaptabilidad"] += 15

        # Normalización de rango [0, 100]
        for var in usuario:
            usuario[var] = float(max(0, min(100, usuario[var])))

        usuario_df = pd.DataFrame([usuario])
        usuario_scaled = scaler.transform(usuario_df[VARIABLES])
        usuario_pca = pca.transform(usuario_scaled)
        cluster_id = int(kmeans.predict(usuario_pca)[0])

        st.session_state["usuario"] = usuario
        st.session_state["cluster"] = cluster_id
        st.session_state["perfil"] = perfil_cluster(cluster_id)
        st.session_state["usuario_pca"] = usuario_pca
        st.session_state["procesado"] = True

# =====================================================
# DASHBOARD DE RESULTADOS
# =====================================================
if st.session_state.get("procesado"):
    usuario = st.session_state["usuario"]
    cluster_idx = st.session_state["cluster"]
    
    st.divider()
    st.header("📊 Reporte conductual de tu expedición")

    cluster_nombres = {
        0: "Cluster Analítico",
        1: "Cluster Explorador",
        2: "Cluster Adaptativo",
        3: "Cluster Reactivo"
    }

    c_col1, c_col2 = st.columns(2)
    c_col1.metric("Perfil detectado", st.session_state["perfil"])
    c_col2.metric("Grupo conductual (Clustering)", cluster_nombres[cluster_idx])

    # 1. Métricas Individuales
    st.subheader("1. Interpretación de tus métricas conductuales")
    
    for var in VARIABLES:
        st.markdown(f"""
        <div class="interpretacion">
            <h4 style="color: #7A003C; margin-bottom: 4px;">{var.capitalize()} — Nivel {nivel(usuario[var])}</h4>
            <p style="margin-bottom: 4px;"><strong>Puntaje:</strong> {round(usuario[var], 1)} / 100</p>
            <p style="color: #475569; margin: 0;">{interpretar_variable(var, usuario[var])}</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Índices Sintéticos Integrados
    st.subheader("2. Índices conductuales integrados")

    indice_exploracion = min(100.0, round(
        usuario["riesgo"] * 0.35 + usuario["exploracion"] * 0.40 + usuario["adaptabilidad"] * 0.25, 1
    ))
    indice_estabilidad = min(100.0, round(
        usuario["planeacion"] * 0.40 + usuario["precision"] * 0.35 + usuario["cooperacion"] * 0.25, 1
    ))
    indice_presion = min(100.0, round(
        usuario["impulsividad"] * 0.45 + usuario["riesgo"] * 0.30 + usuario["adaptabilidad"] * 0.25, 1
    ))

    idx_c1, idx_c2, idx_c3 = st.columns(3)
    with idx_c1:
        st.metric("Exploración Estratégica", f"{indice_exploracion} / 100")
        st.progress(indice_exploracion / 100.0)
    with idx_c2:
        st.metric("Estabilidad Operativa", f"{indice_estabilidad} / 100")
        st.progress(indice_estabilidad / 100.0)
    with idx_c3:
        st.metric("Respuesta Bajo Presión", f"{indice_presion} / 100")
        st.progress(indice_presion / 100.0)

    # 3. Mapa Dimensional PCA
    st.subheader("3. Tu posición en el mapa conductual (PCA 2D)")

    colores = {0: "#7A003C", 1: "#2563EB", 2: "#16A34A", 3: "#EA580C"}
    nombres_clusters = {
        0: "Cauteloso Analítico", 1: "Explorador Estratégico",
        2: "Adaptativo Resiliente", 3: "Impulsivo Bajo Presión"
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#FFFFFF")

    # Scatter de la base sintética
    for c in sorted(base["Cluster"].unique()):
        c_data = base[base["Cluster"] == c]
        ax.scatter(
            c_data["PCA1"], c_data["PCA2"],
            color=colores[c], alpha=0.25, s=35,
            label=nombres_clusters[c]
        )

    # Centroides de cluster
    for c in sorted(base["Cluster"].unique()):
        cx = base[base["Cluster"] == c]["PCA1"].mean()
        cy = base[base["Cluster"] == c]["PCA2"].mean()
        ax.scatter(cx, cy, color=colores[c], s=260, edgecolor="#1E293B", linewidth=1.8, zorder=4)
        ax.text(cx, cy + 0.3, nombres_clusters[c], fontsize=8.5, ha="center", weight="bold", color="#1E293B")

    # Coordenada del participante
    u_pca = st.session_state["usuario_pca"][0]
    ax.scatter(u_pca[0], u_pca[1], color="#DC2626", s=380, marker="X", edgecolor="#000000", linewidth=2, label="Tu perfil", zorder=5)

    ax.set_title("Mapa de Similitud Conductual (Espacio Reducido PCA)", fontsize=13, fontweight="bold", pad=12, color="#1E293B")
    ax.grid(alpha=0.2, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    st.pyplot(fig)

    # 4. Lectura Aplicada
    st.subheader("4. Lectura aplicada de tu expedición")

    narrativas = {
        "Cauteloso Analítico": {
            "significado": "Durante la expedición mostraste un patrón de decisión orientado hacia la estabilidad, el análisis y la reducción de incertidumbre. Tus respuestas sugieren preferencia por escenarios estructurados, previsibles y con mayor control sobre las consecuencias. Este perfil suele evaluar cuidadosamente los riesgos antes de actuar y prioriza la seguridad operativa sobre la improvisación.",
            "patrones": "A lo largo del recorrido aparecieron conductas asociadas con planeación, precisión y control de recursos. Tus decisiones mostraron tendencia a evitar pérdidas innecesarias, buscar información antes de actuar y mantener estabilidad aun bajo presión temporal. También se observó menor impulsividad frente a escenarios inciertos.",
            "implicaciones": "En contextos universitarios, este patrón puede favorecer organización académica, cumplimiento de objetivos y manejo estructurado de responsabilidades. En la vida cotidiana, suele relacionarse con personas que buscan estabilidad y control antes de tomar decisiones importantes. Sin embargo, en ciertos contextos podría existir dificultad para actuar rápidamente ante escenarios altamente cambiantes o ambiguos."
        },
        "Explorador Estratégico": {
            "significado": "Tu expedición mostró un perfil orientado hacia exploración, apertura y búsqueda de oportunidades. Tus decisiones reflejaron comodidad frente a incertidumbre moderada y disposición a asumir riesgos cuando existe posibilidad de recompensa o descubrimiento. Este perfil combina curiosidad con capacidad de adaptación.",
            "patrones": "Durante el recorrido aparecieron patrones asociados con exploración conductual, flexibilidad y tolerancia al riesgo. Mostraste tendencia a avanzar aun con información incompleta y a priorizar experiencias nuevas sobre estabilidad absoluta. También se observó rapidez para ajustarte a cambios de contexto.",
            "implicaciones": "En la vida universitaria este patrón puede favorecer creatividad, aprendizaje autónomo y facilidad para adaptarse a entornos dinámicos. En escenarios personales y profesionales suele relacionarse con iniciativa, innovación y apertura a nuevas experiencias. No obstante, una exploración excesiva podría llevar ocasionalmente a asumir riesgos sin suficiente planeación."
        },
        "Adaptativo Resiliente": {
            "significado": "Tus decisiones sugirieron un perfil equilibrado entre análisis, adaptación y flexibilidad. A lo largo de la expedición mostraste capacidad para ajustarte a situaciones cambiantes sin perder completamente estabilidad o estructura. Este perfil suele responder funcionalmente frente a escenarios ambiguos o inciertos.",
            "patrones": "Se observaron conductas relacionadas con adaptación progresiva, negociación y ajuste contextual. Tus decisiones no se mantuvieron rígidas frente a los cambios, pero tampoco mostraron impulsividad extrema. El patrón general indica capacidad para reorganizar estrategias conforme aparecen nuevos elementos dentro del entorno.",
            "implicaciones": "En contextos universitarios este perfil puede favorecer trabajo colaborativo, resolución de problemas y adaptación a carga académica variable. En la vida cotidiana suele asociarse con personas capaces de enfrentar cambios sin perder funcionalidad general. También puede facilitar manejo emocional más flexible frente a presión o incertidumbre."
        },
        "Impulsivo Bajo Presión": {
            "significado": "Durante la expedición apareció un patrón de decisión más reactivo e inmediato frente a presión o incertidumbre. Tus respuestas mostraron tendencia a actuar rápidamente cuando el entorno exige resolver situaciones en poco tiempo. Este perfil suele priorizar acción inmediata sobre análisis prolongado.",
            "patrones": "A lo largo del recorrido se observaron decisiones asociadas con impulsividad, rapidez de respuesta y tolerancia alta a escenarios inciertos. En distintos momentos se priorizó avanzar rápidamente o resolver situaciones de forma inmediata aun con información incompleta.",
            "implicaciones": "En escenarios universitarios este patrón puede favorecer capacidad de reacción rápida y respuesta eficiente ante presión temporal. En la vida cotidiana puede relacionarse con iniciativa y acción inmediata frente a problemas. Sin embargo, cuando las decisiones requieren análisis profundo, podría existir tendencia a actuar antes de evaluar completamente las consecuencias."
        }
    }

    perfil_actual = st.session_state["perfil"]
    st.markdown(f"**¿Qué significa tu perfil?**\n\n{narrativas[perfil_actual]['significado']}")
    st.markdown(f"**Patrones de decisión observados:**\n\n{narrativas[perfil_actual]['patrones']}")
    st.markdown(f"**¿Qué implicaciones puede tener esto?**\n\n{narrativas[perfil_actual]['implicaciones']}")

    st.info("Este reporte transforma patrones estadísticos observados en simulaciones narrativas de toma de decisiones bajo incertidumbre. Su propósito es formativo y exploratorio dentro de un contexto de ciencia de datos aplicada.")

    if st.button("Reiniciar expedición"):
        for key in ["iniciado", "usuario", "cluster", "perfil", "usuario_pca", "procesado"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
