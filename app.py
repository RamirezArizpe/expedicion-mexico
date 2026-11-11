import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go

# =====================================================
# CONFIGURACIÓN GENERAL
# =====================================================
st.set_page_config(
    page_title="Expedición México · Simulador de Decisiones",
    page_icon="🧭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CSS EDITORIAL: DISEÑO ORGÁNICO E INSTITUCIONAL
# =====================================================
st.markdown("""
<style>
/* Base de la aplicación */
.stApp {
    background-color: #F9FAFB;
    color: #111827;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

.block-container {
    max-width: 800px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* Tipografía de títulos */
h1 {
    color: #5C0626 !important;
    font-size: 2.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em;
    line-height: 1.15 !important;
    margin-bottom: 0.25rem !important;
}

h2 {
    color: #5C0626 !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em;
    margin-top: 1rem !important;
    margin-bottom: 0.5rem !important;
}

h3 {
    color: #1F2937 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
}

/* Cajas contenedoras sobrias (estilo editorial / pliego) */
.editorial-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 8px;
    padding: 1.75rem 2rem;
    margin-bottom: 1.5rem;
}

.step-meta {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6B7280;
    margin-bottom: 0.5rem;
}

.step-prompt {
    font-size: 1rem;
    color: #374151;
    line-height: 1.6;
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

/* Opciones de Radio tipo tarjeta sobria */
div[role="radiogroup"] > label {
    background: #FFFFFF !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 6px !important;
    padding: 0.95rem 1.15rem !important;
    margin-bottom: 0.6rem !important;
    transition: border-color 0.15s ease, background-color 0.15s ease !important;
    cursor: pointer !important;
}

div[role="radiogroup"] > label:hover {
    border-color: #9CA3AF !important;
    background: #F9FAFB !important;
}

div[role="radiogroup"] > label[data-checked="true"] {
    border-color: #5C0626 !important;
    background: #FDF2F4 !important;
}

div[role="radiogroup"] > label span {
    font-size: 0.95rem !important;
    color: #1F2937 !important;
    line-height: 1.45 !important;
}

/* Botones estructurados */
.stButton button {
    border-radius: 6px !important;
    padding: 0.65rem 1.4rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
}

button[kind="primary"] {
    background: #5C0626 !important;
    color: #FFFFFF !important;
    border: 1px solid #5C0626 !important;
}

button[kind="primary"]:hover {
    background: #47041D !important;
    border-color: #47041D !important;
}

/* Métricas del Dashboard */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 1rem 1.25rem;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #6B7280 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    color: #5C0626 !important;
}

/* Bloques de escenarios de vida */
.context-block {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-top: 2px solid #5C0626;
    border-radius: 6px;
    padding: 1.2rem 1.35rem;
    margin-bottom: 0.85rem;
}

.context-title {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #5C0626;
    margin-bottom: 0.4rem;
}

.context-desc {
    font-size: 0.92rem;
    color: #374151;
    line-height: 1.55;
    margin: 0;
}

/* Pestañas limpias */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #E5E7EB;
}

.stTabs [data-baseweb="tab"] {
    font-size: 0.85rem;
    font-weight: 600;
    color: #6B7280;
    padding-bottom: 8px;
}

.stTabs [aria-selected="true"] {
    color: #5C0626 !important;
    border-bottom-color: #5C0626 !important;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA SCIENCE & CACHING
# =====================================================
VARIABLES = [
    "riesgo", "exploracion", "precision", 
    "adaptabilidad", "impulsividad", "planeacion", "cooperacion"
]

@st.cache_data
def inicializar_modelos():
    np.random.seed(42)
    perfiles = []
    
    configuraciones = [
        [25, 40, 88, 55, 25, 88, 65],  # Cauteloso
        [75, 88, 68, 82, 55, 70, 72],  # Explorador
        [55, 65, 78, 88, 42, 75, 82],  # Adaptativo
        [85, 58, 52, 48, 88, 42, 50]   # Impulsivo
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
# DICCIONARIOS DE INTERPRETACIÓN
# =====================================================
def nivel(valor):
    if valor >= 75: return "alto"
    elif valor >= 45: return "moderado"
    return "bajo"

def interpretar_variable(nombre, valor):
    interpretaciones = {
        "riesgo": {
            "alto": "Aceptas la incertidumbre con soltura y te orientas a escenarios con amplio margen de ganancia o pérdida.",
            "moderado": "Mides el terreno antes de dar el paso; buscas un balance práctico entre audacia y seguridad.",
            "bajo": "Prefieres pisar sobre seguro y tener el control de los imprevistos antes de comprometer recursos."
        },
        "exploracion": {
            "alto": "Muestras disposición constante a abrir camino, probar rutas alternas y salir de lo conocido.",
            "moderado": "Disfrutas conocer opciones nuevas, pero te apoyas en paradas seguras para no perder el rumbo.",
            "bajo": "Prefieres consolidar y dominar los caminos probados antes de ponerte a experimentar."
        },
        "precision": {
            "alto": "Prestas atención minuciosa al detalle. Rara vez tomas decisiones sin revisar la información previa.",
            "moderado": "Cuidas los detalles importantes sin dejar que el análisis frene el ritmo de avance.",
            "bajo": "Priorizas el ritmo y la acción inmediata por encima de la exactitud milimétrica."
        },
        "adaptabilidad": {
            "alto": "Muestras agilidad frente a imprevistos: si el plan inicial se altera, resuelves con los recursos a la mano.",
            "moderado": "Te ajustas de forma práctica a los cambios habituales de ruta sin perder el orden de avance.",
            "bajo": "Rindes mucho mejor cuando las pautas, los tiempos y el camino están claramente fijados desde el inicio."
        },
        "impulsividad": {
            "alto": "Reaccionas de inmediato ante la presión o la prisa; prefieres actuar de una vez a quedarte en la deliberación.",
            "moderado": "Sabes alternar entre reacciones rápidas y pausas breves de valoración.",
            "bajo": "Procesas las opciones con calma; prefieres no precipitarte aunque el entorno exija rapidez."
        },
        "planeacion": {
            "alto": "Tu enfoque principal es la anticipación: valoras tener un mapa claro, orden en los tiempos y previsión de gastos.",
            "moderado": "Trazas un plan base sólido, pero dejas suficiente margen de maniobra sobre el terreno.",
            "bajo": "Te orientas a la acción directa; te resulta más natural resolver frente al hecho que diseñar planes extensos."
        },
        "cooperacion": {
            "alto": "Buscas construir acuerdos, conciliar posturas y procurar que las decisiones representen al grupo.",
            "moderado": "Sabes colaborar en equipo sin perder tu autonomía ni tu criterio propio.",
            "bajo": "Confías primordialmente en tu propio criterio para actuar con agilidad y evitar discusiones prolongadas."
        }
    }
    return interpretaciones[nombre][nivel(valor)]

def perfil_cluster(c):
    return {
        0: "Cauteloso Analítico",
        1: "Explorador Estratégico",
        2: "Adaptativo Resiliente",
        3: "Impulsivo Bajo Presión"
    }[c]

# =====================================================
# MATRIZ DE ESCENARIOS
# =====================================================
ESCENARIOS = [
    {
        "estacion": "01 / Altiplano Central",
        "titulo": "Salida de madrugada",
        "descripcion": "El trayecto inicia temprano. Tienen la ruta trazada por autopista de cuota, pero un contacto local les avisa de una desviación por carretera libre donde hay una zona arqueológica poco conocida y sin filtros de acceso. ¿Qué decides?",
        "opciones": [
            "Seguir por la autopista directa; la prioridad es cumplir los tiempos fijados sin imprevistos mecánicos o de tráfico.",
            "Tomar la carretera libre un tramo: permite conocer el paisaje sin alterar de forma sensible el tiempo estimado.",
            "Irse directo por la brecha comunitaria; los descubrimientos más valiosos del viaje no están sobre el asfalto principal."
        ]
    },
    {
        "estacion": "02 / Sierra de Oaxaca",
        "titulo": "Gestión de recursos en la montaña",
        "descripcion": "En la sierra les ofrecen contratar transporte local y guías de la comunidad para subir a un mirador en la niebla antes de que comience a llover. El costo absorbe una parte considerable del presupuesto del viaje. ¿Cómo lo manejas?",
        "opciones": [
            "Cuidar la reserva y hacer el recorrido a pie por los senderos autorizados que no generan costo extra.",
            "Pagar solo una parte del recorrido para conocer el primer paraje y guardar un colchón para imprevistos.",
            "Cubrir completo el recorrido de una vez; una oportunidad así rara vez se repite y el gasto se ajusta después."
        ]
    },
    {
        "estacion": "03 / Península de Yucatán",
        "titulo": "Bifurcación sin cobertura",
        "descripcion": "Buscando el acceso a un cenote en la selva baja, los teléfonos se quedan sin señal y el mapa descargado no coincide con una desviación recién abierta. Queda apenas hora y media de luz natural. ¿Qué instrucción das?",
        "opciones": [
            "Hacer alto total, asegurar el vehículo y esperar al día siguiente con luz clara para pedir informes en el poblado.",
            "Bajar a revisar huellas y referencias físicas del terreno para avanzar con lentitud y verificación constante.",
            "Seguir la brecha más transitada a paso firme para ganar terreno antes de que caiga la noche por completo."
        ]
    },
    {
        "estacion": "04 / Campamento Base",
        "titulo": "Desacuerdo sobre el itinerario",
        "descripcion": "El pronóstico anuncia un descenso brusco de temperatura y el vehículo muestra desgaste en frenos. En la cena surge la discusión: una parte del grupo pide regresar a un albergue seguro y la otra insiste en mantener el ascenso. ¿Cómo intervienes?",
        "opciones": [
            "Recordar el plan original de la salida y pedir apegarse a lo acordado desde el principio por disciplina del equipo.",
            "Abrir un espacio breve de diálogo para escuchar ambas posturas y buscar una opción intermedia que dé tranquilidad a todos.",
            "Cortar la deliberación y proponer un destino alterno que no estaba en el mapa para replantear la ruta de raíz."
        ]
    },
    {
        "estacion": "05 / Terminal Pantitlán, CDMX",
        "titulo": "Cierre de jornada y retorno",
        "descripcion": "De regreso en la ciudad, llegan al punto de transbordo con equipo y mochilas justo cuando anuncian la salida del último tren de la noche. El acceso está congestionado. ¿Cómo reaccionas?",
        "opciones": [
            "Apretar el paso esquivando el flujo de gente para alcanzar a ingresar antes del cierre definitivo de andenes.",
            "Pausar en el exterior, abrir el mapa de movilidad urbana y revisar rutas alternas de superficie que sigan activas.",
            "Organizar un traslado privado entre todos aunque implique gasto adicional; la prioridad es llegar con el equipo íntegro."
        ]
    }
]

# =====================================================
# INICIALIZACIÓN DE ESTADOS
# =====================================================
if "etapa" not in st.session_state:
    st.session_state["etapa"] = "intro"
if "paso_actual" not in st.session_state:
    st.session_state["paso_actual"] = 0
if "respuestas" not in st.session_state:
    st.session_state["respuestas"] = {}
if "nombre" not in st.session_state:
    st.session_state["nombre"] = "Participante"

# =====================================================
# VISTA 1: INTRODUCCIÓN EDITORIAL
# =====================================================
if st.session_state["etapa"] == "intro":
    st.title("🧭 Expedición México")
    st.markdown("<p style='font-size: 1.05rem; color: #4B5563; margin-top: -6px;'>Simulación interactiva de toma de decisiones bajo incertidumbre</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="editorial-card">
        <h3 style="margin-top:0;">Propósito del ejercicio</h3>
        <p style="color: #4B5563; line-height: 1.6; margin-bottom: 0.8rem;">
            A través de cinco situaciones basadas en recorridos por distintas regiones del país, esta experiencia examina las decisiones que tomas cuando los recursos son limitados, los tiempos apremian o la información es parcial.
        </p>
        <p style="color: #4B5563; line-height: 1.6; margin-bottom: 1rem;">
            Las respuestas alimentan un modelo de <strong>agrupamiento no supervisado (K-Means)</strong> y una <strong>proyección dimensional (PCA)</strong> calculada a partir de perfiles de referencia. Al finalizar, obtendrás un análisis sobre tu tendencia predominante al resolver dilemas.
        </p>
        <p style="color: #6B7280; font-size: 0.84rem; line-height: 1.5; margin-bottom: 0; border-top: 1px solid #E5E7EB; padding-top: 10px;">
            Este ejercicio corresponde a una simulación de ciencia de datos aplicada a la conducta con fines educativos y de divulgación. Los resultados no constituyen una evaluación psicológica ni un diagnóstico clínico. La participación es voluntaria.
        </p>
    </div>
    """, unsafe_allow_html=True)

    nombre_in = st.text_input("Registro de participante (nombre o identificador):", placeholder="Ej. Mariana Soto")
    
    if st.button("Iniciar ejercicio", type="primary"):
        st.session_state["nombre"] = nombre_in.strip() if nombre_in.strip() else "Participante"
        st.session_state["etapa"] = "wizard"
        st.session_state["paso_actual"] = 0
        st.rerun()

# =====================================================
# VISTA 2: ASISTENTE PASO A PASO
# =====================================================
elif st.session_state["etapa"] == "wizard":
    idx = st.session_state["paso_actual"]
    total = len(ESCENARIOS)
    datos_escenario = ESCENARIOS[idx]

    progreso = (idx + 1) / total
    st.progress(progreso)

    st.markdown(f"""
    <div class="editorial-card">
        <div class="step-meta">Situación {idx + 1} de {total} · {datos_escenario['estacion']}</div>
        <h2 style="margin-top:0;">{datos_escenario['titulo']}</h2>
        <div class="step-prompt">{datos_escenario['descripcion']}</div>
    </div>
    """, unsafe_allow_html=True)

    opcion_guardada = st.session_state["respuestas"].get(idx, datos_escenario["opciones"][0])
    seleccion = st.radio(
        "Curso de acción seleccionado:",
        datos_escenario["opciones"],
        index=datos_escenario["opciones"].index(opcion_guardada),
        key=f"radio_{idx}",
        label_visibility="collapsed"
    )

    col_nav1, col_nav2 = st.columns([1, 1])

    with col_nav1:
        if idx > 0:
            if st.button("Anterior", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                st.session_state["paso_actual"] -= 1
                st.rerun()

    with col_nav2:
        if idx < total - 1:
            if st.button("Siguiente", type="primary", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                st.session_state["paso_actual"] += 1
                st.rerun()
        else:
            if st.button("Generar reporte", type="primary", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                
                with st.spinner("Procesando vectores de decisión y calculando posición PCA..."):
                    time.sleep(0.6)
                
                st.session_state["etapa"] = "resultados"
                st.rerun()

# =====================================================
# VISTA 3: REPORTE DE RESULTADOS EDITORIAL
# =====================================================
elif st.session_state["etapa"] == "resultados":
    usuario = {var: 50.0 for var in VARIABLES}
    r = st.session_state["respuestas"]

    # R1: Altiplano
    if "autopista" in r[0]:
        usuario["riesgo"] -= 20; usuario["precision"] += 20; usuario["planeacion"] += 20
    elif "carretera libre" in r[0]:
        usuario["adaptabilidad"] += 15; usuario["exploracion"] += 10
    else:
        usuario["riesgo"] += 30; usuario["exploracion"] += 30; usuario["impulsividad"] += 10

    # R2: Oaxaca
    if "Cuidar la reserva" in r[1]:
        usuario["planeacion"] += 20; usuario["precision"] += 15
    elif "Pagar solo una parte" in r[1]:
        usuario["adaptabilidad"] += 15
    else:
        usuario["riesgo"] += 15; usuario["impulsividad"] += 20

    # R3: Yucatán
    if "Hacer alto total" in r[2]:
        usuario["precision"] += 15; usuario["impulsividad"] -= 10
    elif "Bajar a revisar huellas" in r[2]:
        usuario["adaptabilidad"] += 20
    else:
        usuario["riesgo"] += 20; usuario["impulsividad"] += 25

    # R4: Campamento
    if "Recordar el plan" in r[3]:
        usuario["cooperacion"] -= 15
    elif "Abrir un espacio" in r[3]:
        usuario["cooperacion"] += 25; usuario["adaptabilidad"] += 10
    else:
        usuario["riesgo"] += 10

    # R5: Regreso
    if "Apretar el paso" in r[4]:
        usuario["impulsividad"] += 15; usuario["riesgo"] += 10
    elif "Pausar en el exterior" in r[4]:
        usuario["planeacion"] += 15; usuario["precision"] += 15
    else:
        usuario["adaptabilidad"] += 15

    for var in usuario:
        usuario[var] = float(max(0, min(100, usuario[var])))

    usuario_df = pd.DataFrame([usuario])
    usuario_scaled = scaler.transform(usuario_df[VARIABLES])
    usuario_pca = pca.transform(usuario_scaled)[0]
    cluster_id = int(kmeans.predict([usuario_pca])[0])
    perfil_nombre = perfil_cluster(cluster_id)

    st.title("Reporte de Toma de Decisiones")
    st.markdown(f"<p style='font-size: 0.95rem; color: #4B5563; margin-top: -6px;'>Participante: <strong>{st.session_state['nombre']}</strong> · Modelo de segmentación K-Means</p>", unsafe_allow_html=True)

    col_hero1, col_hero2 = st.columns(2)
    col_hero1.metric("Estilo de decisión", perfil_nombre)
    col_hero2.metric("Grupo en el modelo", f"Cluster {cluster_id + 1}")

    tab_detalle, tab_mapa, tab_radar = st.tabs([
        "Análisis y Contextos de Aplicación", 
        "Plano de Distribución (PCA)", 
        "Desglose por Dimensión"
    ])

    # TAB 1: DIAGNÓSTICO PROFUNDO
    with tab_detalle:
        idx_exp = min(100.0, round(usuario["riesgo"]*0.35 + usuario["exploracion"]*0.40 + usuario["adaptabilidad"]*0.25, 1))
        idx_est = min(100.0, round(usuario["planeacion"]*0.40 + usuario["precision"]*0.35 + usuario["cooperacion"]*0.25, 1))
        idx_pre = min(100.0, round(usuario["impulsividad"]*0.45 + usuario["riesgo"]*0.30 + usuario["adaptabilidad"]*0.25, 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("Iniciativa y Exploración", f"{idx_exp} / 100")
        c2.metric("Control y Estabilidad", f"{idx_est} / 100")
        c3.metric("Reacción Bajo Presión", f"{idx_pre} / 100")

        st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 1px solid #E5E7EB;' />", unsafe_allow_html=True)

        NARRATIVAS_EXPANDIDAS = {
            "Cauteloso Analítico": {
                "que_significa": """
                Durante el recorrido mostraste un estilo orientado a la prudencia, la estimación de consecuencias y la preservación de márgenes de seguridad. 
                Rara vez comprometes recursos o tiempos sin contar con elementos objetivos para evaluar el riesgo. 
                Tu tendencia natural es verificar condiciones previas, cuidar presupuestos y asegurar que las bases del plan se cumplan antes de abrir nuevas variantes.
                """,
                "superpoder": "Capacidad para establecer orden, certidumbre operativa y control sistemático en situaciones donde otros actúan por apresuramiento.",
                "punto_ciego": "La búsqueda excesiva de certeza o control puede traducirse en indecisión temporal, postergando acciones en contextos que demandan respuesta ágil.",
                "estudiantil": "En proyectos semestrales, seminarios o redacción de tesis destacas por el rigor metodológico, la organización de cronogramas y el control bibliográfico. Recomendación: procura fijar fechas límite de cierre para no demorar entregas buscando revisiones innecesarias.",
                "laboral": "Aportas un valor alto en análisis de procesos, planeación financiera, control de calidad, auditoría o áreas normativas. Recomendación: ejercita la presentación de propuestas en fases tempranas de desarrollo sin esperar a tenerlas completamente blindadas.",
                "cotidiana": "En la administración de finanzas personales, viajes o compras relevantes actúas con sobriedad y cuentas con reservas. En actividades grupales o de descanso, procura dar margen a la espontaneidad sin requerir que cada detalle esté previsto."
            },
            "Explorador Estratégico": {
                "que_significa": """
                Tu toma de decisiones se orienta al descubrimiento de oportunidades, la curiosidad de campo y la apertura a la experiencia. 
                Comprendes que cualquier proyecto implica un grado intrínseco de incertidumbre y aceptas asumir riesgos calculados si el aprendizaje o la ganancia potencial justifican el esfuerzo. 
                Te resulta natural considerar rutas alternas y adaptarte sobre la marcha.
                """,
                "superpoder": "Iniciativa y visión para identificar oportunidades fuera del marco convencional y movilizar a equipos hacia nuevos objetivos.",
                "punto_ciego": "Tendencia a subestimar el consumo de tiempo o recursos, asumiendo compromisos paralelos que pueden dificultar el cierre efectivo de los proyectos abiertos.",
                "estudiantil": "Sobresales en materias de formulación de proyectos, talleres de innovación, trabajo de campo y actividades de discusión. Recomendación: apóyate en metodologías de seguimiento estricto para no descuidar tareas rutinarias pero necesarias.",
                "laboral": "Idóneo para desarrollo de producto, apertura de proyectos, áreas comerciales, vinculación institucional y células de innovación. Recomendación: establece alianzas de trabajo con perfiles orientados a la operación para aterrizar iniciativas en metas tangibles.",
                "cotidiana": "Sueles proponer destinos, rutas o actividades que el grupo no contemplaba. Mantén el hábito de revisar aspectos logísticos mínimos (horarios, disponibilidad básica) para evitar contratiempos a quienes te acompañan."
            },
            "Adaptativo Resiliente": {
                "que_significa": """
                Reflejas un equilibrio consistente entre pragmatismo, flexibilidad y receptividad hacia las opiniones del grupo. 
                Tu cualidad principal es la estabilidad frente a imprevistos: cuando el escenario inicial cambia, evitas la frustración improductiva y reorganizas prioridades con los elementos disponibles. 
                Asimismo, actúas con destreza como punto de convergencia ante desacuerdos en el equipo.
                """,
                "superpoder": "Capacidad para mediar diferencias, sostener la continuidad de la operación y resolver imprevistos sin generar fricción en el entorno.",
                "punto_ciego": "En el afán de conciliar posturas o evitar tensiones interpersonales, puedes relegar tu propio criterio o asumir cargas de trabajo que no te corresponden.",
                "estudiantil": "Facilitas la integración en equipos de trabajo multidisciplinarios, equilibrando roles y mediando cuando se presentan atrasos. Recomendación: establece límites claros en la asignación de responsabilidades para evitar sobrecargas individuales.",
                "laboral": "Perfil altamente funcional en coordinación interdepartamental, gestión de cuentas, consultoría, atención a contingencias y liderazgo de equipos diversos. Recomendación: formula con claridad tu postura técnica antes de iniciar procesos de negociación.",
                "cotidiana": "Te integras con fluidez en distintos círculos y respondes con serenidad frente a cambios de planes en reuniones familiares o de amistades. Asegúrate de que tu flexibilidad no te lleve a postergar tus propias necesidades y tiempos de descanso."
            },
            "Impulsivo Bajo Presión": {
                "que_significa": """
                Frente a la urgencia y la presión temporal, tu mecanismo de decisión se activa en dirección a la acción inmediata. 
                Te incomoda la deliberación prolongada cuando el problema está presente y requiere atención práctica. 
                Prefieres avanzar, asumir la posibilidad de ajustar el rumbo sobre la marcha y resolver con determinación antes que paralizar el movimiento del grupo por exceso de duda.
                """,
                "superpoder": "Velocidad de ejecución y determinación en momentos críticos donde el retraso representa un costo más alto que un error corregible.",
                "punto_ciego": "Actuar con prontitud bajo tensión puede llevarte a omitir información técnica disponible o a generar soluciones temporales que demandan corrección posterior.",
                "estudiantil": "Capacidad notable para resolver entregas de alta demanda bajo presión y responder con agilidad en evaluaciones orales o prácticas. Recomendación: reserva periodos de planeación anticipada para evitar concentrar el esfuerzo exclusivamente en el cierre.",
                "laboral": "Aporte determinante en áreas de respuesta operativa, logística inmediata, resolución de quejas complejas o entornos de alta exigencia temporal. Recomendación: incorpora una pausa breve de verificación con integrantes del equipo antes de formalizar decisiones críticas.",
                "cotidiana": "Tomas la iniciativa ante imprevistos prácticos que requieren resolución rápida. Cuida que la energía con la que resuelves no sea percibida como impaciencia o imposición frente a personas que procesan los problemas a un ritmo diferente."
            }
        }

        info_perfil = NARRATIVAS_EXPANDIDAS[perfil_nombre]

        st.markdown(f"### Caracterización del perfil: {perfil_nombre}")
        st.write(info_perfil["que_significa"])

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"""
            <div style='background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:1rem; margin-bottom:1rem;'>
                <div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#047857; margin-bottom:0.25rem;'>Fortaleza principal</div>
                <div style='font-size:0.92rem; color:#374151; line-height:1.5;'>{info_perfil['superpoder']}</div>
            </div>
            """, unsafe_allow_html=True)
        with col_f2:
            st.markdown(f"""
            <div style='background:#F9FAFB; border:1px solid #E5E7EB; border-radius:6px; padding:1rem; margin-bottom:1rem;'>
                <div style='font-size:0.75rem; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; color:#B45309; margin-bottom:0.25rem;'>Punto de atención</div>
                <div style='font-size:0.92rem; color:#374151; line-height:1.5;'>{info_perfil['punto_ciego']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Contextos de aplicación cotidiana")
        
        st.markdown(f"""
        <div class="context-block">
            <div class="context-title">Vida académica y formativa</div>
            <p class="context-desc">{info_perfil['estudiantil']}</p>
        </div>
        <div class="context-block">
            <div class="context-title">Vida laboral y profesional</div>
            <p class="context-desc">{info_perfil['laboral']}</p>
        </div>
        <div class="context-block">
            <div class="context-title">Vida cotidiana y relaciones personales</div>
            <p class="context-desc">{info_perfil['cotidiana']}</p>
        </div>
        """, unsafe_allow_html=True)

    # TAB 2: PLOTLY DISPERSIÓN PCA
    with tab_mapa:
        st.markdown("<p style='font-size:0.9rem; color:#4B5563; margin-bottom:12px;'>Distribución de las decisiones en los dos primeros componentes principales (N=180 registros de referencia).</p>", unsafe_allow_html=True)
        
        colores_clusters = {0: "#5C0626", 1: "#1E40AF", 2: "#047857", 3: "#C2410C"}
        nombres_c = {0: "Cauteloso Analítico", 1: "Explorador Estratégico", 2: "Adaptativo Resiliente", 3: "Impulsivo Bajo Presión"}

        fig_pca = go.Figure()

        for c in sorted(base["Cluster"].unique()):
            sub = base[base["Cluster"] == c]
            fig_pca.add_trace(go.Scatter(
                x=sub["PCA1"], y=sub["PCA2"],
                mode="markers",
                name=nombres_c[c],
                marker=dict(color=colores_clusters[c], size=5.5, opacity=0.3),
                hoverinfo="name"
            ))

        for c in sorted(base["Cluster"].unique()):
            cx = base[base["Cluster"] == c]["PCA1"].mean()
            cy = base[base["Cluster"] == c]["PCA2"].mean()
            fig_pca.add_trace(go.Scatter(
                x=[cx], y=[cy],
                mode="markers+text",
                text=[nombres_c[c]],
                textposition="top center",
                textfont=dict(size=9.5, color="#1F2937", family="-apple-system, sans-serif"),
                marker=dict(color=colores_clusters[c], size=12, line=dict(color="#111827", width=1.5)),
                showlegend=False
            ))

        fig_pca.add_trace(go.Scatter(
            x=[usuario_pca[0]], y=[usuario_pca[1]],
            mode="markers+text",
            text=["Tu resultado"],
            textposition="bottom center",
            textfont=dict(size=11, color="#DC2626", family="-apple-system, sans-serif"),
            marker=dict(color="#DC2626", size=15, symbol="x", line=dict(width=2.5)),
            name="Tu resultado"
        ))

        fig_pca.update_layout(
            height=400,
            margin=dict(l=15, r=15, t=15, b=15),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5, font=dict(size=10.5)),
            xaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=9)),
            yaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False, tickfont=dict(size=9))
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # TAB 3: RADAR CHART Y DETALLE DE VARIABLES
    with tab_radar:
        st.markdown("<p style='font-size:0.9rem; color:#4B5563; margin-bottom:12px;'>Mapeo radial sobre escala estandarizada de 0 a 100 puntos.</p>", unsafe_allow_html=True)
        categories = [v.capitalize() for v in VARIABLES]
        values = [usuario[v] for v in VARIABLES]
        categories.append(categories[0])
        values.append(values[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(92, 6, 38, 0.12)',
            line=dict(color='#5C0626', width=1.8),
            name="Resultado"
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#9CA3AF", tickfont=dict(size=8.5)),
                angularaxis=dict(tickfont=dict(size=9.5, color="#374151")),
                bgcolor="#FFFFFF"
            ),
            height=370,
            margin=dict(l=35, r=35, t=15, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("### Valores obtenidos por dimensión")
        for var in VARIABLES:
            puntaje = usuario[var]
            st.markdown(f"**{var.capitalize()}** — `{round(puntaje, 1)} / 100` ({nivel(puntaje).capitalize()})")
            st.caption(interpretar_variable(var, puntaje))

    st.write("")
    if st.button("Reiniciar ejercicio", use_container_width=True):
        st.session_state["etapa"] = "intro"
        st.session_state["paso_actual"] = 0
        st.session_state["respuestas"] = {}
        st.rerun()
