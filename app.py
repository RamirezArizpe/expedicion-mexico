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
# CSS AVANZADO: UX/UI MODERNA
# =====================================================
st.markdown("""
<style>
/* Base de la app */
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.block-container {
    max-width: 840px;
    padding-top: 2rem;
    padding-bottom: 3.5rem;
}

/* Títulos con contraste nítido */
h1 {
    color: #7A003C !important;
    font-size: 2.3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    color: #7A003C !important;
    font-weight: 700 !important;
}

/* Tarjeta contenedora de estación */
.station-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 2rem 2.2rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
}

.station-badge {
    display: inline-block;
    background: rgba(122, 0, 60, 0.08);
    color: #7A003C;
    font-weight: 700;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 4px 12px;
    border-radius: 999px;
    margin-bottom: 10px;
}

.station-prompt {
    font-size: 1.05rem;
    color: #334155;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}

/* Estilización tipo tarjeta para opciones st.radio */
div[role="radiogroup"] > label {
    background: #F8FAFC !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 1rem 1.2rem !important;
    margin-bottom: 0.65rem !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

div[role="radiogroup"] > label:hover {
    border-color: #CBD5E1 !important;
    background: #F1F5F9 !important;
    transform: translateY(-1px);
}

div[role="radiogroup"] > label[data-checked="true"] {
    border-color: #7A003C !important;
    background: rgba(122, 0, 60, 0.04) !important;
}

/* Botones principales y secundarios */
.stButton button {
    border-radius: 12px !important;
    padding: 0.8rem 1.8rem !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

button[kind="primary"] {
    background: #7A003C !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(122, 0, 60, 0.2) !important;
}

button[kind="primary"]:hover {
    background: #9A1B5A !important;
    transform: translateY(-1px);
}

/* Tarjetas de recomendaciones por escenario */
.scenario-box {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #7A003C;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.02);
}

.scenario-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 700;
    font-size: 1.02rem;
    color: #7A003C;
    margin-bottom: 0.4rem;
}

.scenario-body {
    font-size: 0.94rem;
    color: #334155;
    line-height: 1.6;
    margin: 0;
}

/* Métricas del Dashboard */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.02);
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
# DICCIONARIOS DE INTERPRETACIÓN ACCESIBLE
# =====================================================
def nivel(valor):
    if valor >= 75: return "alto"
    elif valor >= 45: return "moderado"
    return "bajo"

def interpretar_variable(nombre, valor):
    interpretaciones = {
        "riesgo": {
            "alto": "Te sientes a gusto con la incertidumbre. Te atraen las apuestas audaces donde hay mucho por ganar, asumiendo el costo si las cosas no salen a la primera.",
            "moderado": "No le huyes a los retos, pero cuidas no aventarte a ciegas; buscas un balance práctico entre audacia y seguridad.",
            "bajo": "Prefieres pisar sobre seguro y tener el control de los imprevistos antes de comprometer tiempo o recursos."
        },
        "exploracion": {
            "alto": "Tienes curiosidad constante por abrir brecha, probar caminos nuevos y salir de la zona conocida.",
            "moderado": "Disfrutas conocer cosas nuevas, pero te apoyas en paradas seguras para no perder el rumbo.",
            "bajo": "Prefieres dominar y perfeccionar las rutas que ya conoces antes de ponerte a experimentar."
        },
        "precision": {
            "alto": "Te fijas mucho en los detalles. No te gusta tomar decisiones a la ligera sin revisar bien la información previa.",
            "moderado": "Cuidas los detalles importantes sin dejar que el exceso de análisis frene el ritmo de avance.",
            "bajo": "Priorizas actuar rápido y resolver sobre la marcha antes que detenerte en minucias."
        },
        "adaptabilidad": {
            "alto": "Tienes cintura para sortear imprevistos: si el plan inicial se cae, resuelves con lo que haya a la mano sin perder el temple.",
            "moderado": "Te ajustas de forma práctica a los cambios habituales sin que te descoloquen.",
            "bajo": "Rindes mucho mejor cuando las reglas, los tiempos y el camino están claramente marcados desde el inicio."
        },
        "impulsividad": {
            "alto": "Reaccionas de inmediato ante la presión o el apuro; prefieres moverte ya antes de quedarte pensando.",
            "moderado": "Sabes cuándo responder rápido y cuándo hacer una pausa corta para valorar la situación.",
            "bajo": "Procesas las opciones con calma; prefieres no apresurarte aunque el entorno tenga prisa."
        },
        "planeacion": {
            "alto": "Tu mayor fortaleza es anticipar: te gusta tener un mapa claro, itinerario previsto y recursos ordenados.",
            "moderado": "Trazas un plan base sólido, pero dejas suficiente margen para maniobrar según se presenten las cosas.",
            "bajo": "Eres de acción directa; te resulta más cómodo resolver frente al problema que pasar horas proyectando."
        },
        "cooperacion": {
            "alto": "Buscas sumar voluntades, conciliar puntos de vista y asegurar que todos en el grupo se sientan tomados en cuenta.",
            "moderado": "Sabes colaborar en equipo sin perder tu propia voz ni tu autonomía de decisión.",
            "bajo": "Confías primordialmente en tu propio criterio para avanzar con agilidad y evitar discusiones interminables."
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
# MATRIZ DE ESCENARIOS (Narrativa Mexicana)
# =====================================================
ESCENARIOS = [
    {
        "estacion": "Estación 1 · Altiplano Central",
        "titulo": "La Salida de Madrugada",
        "descripcion": "La camioneta sale al amanecer. Tienen la ruta marcada por autopista de cuota, pero un contacto local les avisa de una desviación por carretera libre donde hay una zona arqueológica poco conocida y sin filtros de acceso. ¿Qué decides?",
        "opciones": [
            "Seguir por la autopista directa; la prioridad es llegar a tiempo y sin sorpresas mecánicas o de tráfico.",
            "Tomar la carretera libre un tramo: permite conocer el paisaje sin desviarse demasiado del tiempo estimado.",
            "Irse directo por la brecha comunitaria; las mejores historias del viaje no están sobre el asfalto principal."
        ]
    },
    {
        "estacion": "Estación 2 · Sierra de Oaxaca",
        "titulo": "El Dilema del Presupuesto",
        "descripcion": "En la sierra les ofrecen contratar camionetas 4x4 y guías de la comunidad para subir a un mirador en la niebla antes de que caiga el aguacero de la tarde. La vuelta consume una parte importante de la lana del viaje. ¿Cómo lo manejas?",
        "opciones": [
            "Cuidar la reserva y hacer el recorrido a pie por los senderos autorizados que no generan costo extra.",
            "Pagar solo una parte del recorrido para conocer el primer paraje y guardar un colchón para imprevistos.",
            "Pagar completo el recorrido de una vez; una vista así difícilmente se repite y el dinero se repone después."
        ]
    },
    {
        "estacion": "Estación 3 · Península de Yucatán",
        "titulo": "Sin Señal en la Selva",
        "descripcion": "Buscando el acceso a un cenote en la selva baja, los celulares se quedan sin señal y el mapa descargado no coincide con una bifurcación recién abierta. Queda apenas hora y media de luz. ¿Qué instrucción das?",
        "opciones": [
            "Hacer alto total, no arriesgar la camioneta y esperar al día siguiente con luz clara para pedir informes en el pueblo.",
            "Bajar a revisar huellas de llantas y referencias del camino para avanzar despacio, metro a metro.",
            "Agarrar el camino más ancho a buen paso para ganar terreno antes de que se oculte el sol por completo."
        ]
    },
    {
        "estacion": "Estación 4 · Campamento Base",
        "titulo": "La Sobremesa Tensa",
        "descripcion": "El pronóstico anuncia frente frío y la camioneta viene algo justa de frenos. En la cena se desata la discusión: la mitad quiere regresar a descansar a una cabaña segura y la otra insiste en subir al cerro cueste lo que cueste. ¿Cómo intervienes?",
        "opciones": [
            "Recordar el plan original de la salida y pedirle a todos que nos apeguemos a lo acordado desde el principio.",
            "Poner una pausa de cinco minutos para escuchar ambas posturas y buscar una opción intermedia que calme los ánimos.",
            "Cortar la discusión de tajo y proponer cambiar el rumbo completo hacia un destino distinto que nadie traía en mente."
        ]
    },
    {
        "estacion": "Estación 5 · Terminal Pantitlán, CDMX",
        "titulo": "El Cierre de la Jornada",
        "descripcion": "De regreso en la ciudad, llegan a la estación de transbordo cargando mochilas y equipo justo cuando suena el timbre de cierre del último tren de la noche. La entrada está llena de gente. ¿Cómo reaccionas?",
        "opciones": [
            "Apretar el paso esquivando a la gente para alcanzar a entrar al vagón antes de que cierren las puertas.",
            "Hacer una pausa en el andén exterior, abrir la aplicación de movilidad y ver qué opciones de trolebús o metrobús siguen activas.",
            "Pedir un taxi de aplicación entre todos aunque cueste más; lo importante es llegar a casa seguros y con el equipo completo."
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
    st.session_state["nombre"] = "Viajero"

# =====================================================
# VISTA 1: INTRODUCCIÓN LIMPIA
# =====================================================
if st.session_state["etapa"] == "intro":
    st.title("🧭 Expedición México")
    st.markdown("##### Una experiencia interactiva de toma de decisiones bajo incertidumbre.")
    
    st.markdown("""
    <div class="station-card">
        <h3 style="margin-top:0;">Bienvenido a la expedición</h3>
        <p style="color: #475569; line-height: 1.6;">
            A lo largo de cinco paradas clave en la geografía mexicana, enfrentarás dilemas típicos de ruta donde los recursos, el tiempo o el clima te obligarán a tomar decisiones con información incompleta.
        </p>
        <p style="color: #475569; line-height: 1.6;">
            Detrás de cada respuesta, un modelo de <strong>Clustering Conductual (K-Means)</strong> y <strong>Reducción Dimensional (PCA)</strong> proyectará tu estilo de resolución frente a cientos de perfiles simulados.
        </p>
        <p style="color: #64748B; font-size: 0.9rem; line-height: 1.5; margin-bottom: 0; border-top: 1px dashed #E2E8F0; padding-top: 12px;">
            ⚠️ <em>Nota importante: Este prototipo es una simulación formativa de ciencia de datos aplicada a la conducta. Los resultados no constituyen una evaluación diagnóstica, clínica ni psicológica. Tu participación es totalmente libre y voluntaria.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)

    nombre_in = st.text_input("¿Cómo te registramos en la bitácora?", placeholder="Nombre o alias")
    
    if st.button("Iniciar travesía", type="primary"):
        st.session_state["nombre"] = nombre_in.strip() if nombre_in.strip() else "Viajero"
        st.session_state["etapa"] = "wizard"
        st.session_state["paso_actual"] = 0
        st.rerun()

# =====================================================
# VISTA 2: ASISTENTE PASO A PASO (WIZARD)
# =====================================================
elif st.session_state["etapa"] == "wizard":
    idx = st.session_state["paso_actual"]
    total = len(ESCENARIOS)
    datos_escenario = ESCENARIOS[idx]

    progreso = (idx + 1) / total
    st.progress(progreso)
    st.caption(f"Hito {idx + 1} de {total} · {datos_escenario['estacion']}")

    st.markdown(f"""
    <div class="station-card">
        <div class="station-badge">{datos_escenario['estacion']}</div>
        <h2 style="margin-top:0; font-size: 1.5rem;">{datos_escenario['titulo']}</h2>
        <div class="station-prompt">{datos_escenario['descripcion']}</div>
    </div>
    """, unsafe_allow_html=True)

    opcion_guardada = st.session_state["respuestas"].get(idx, datos_escenario["opciones"][0])
    seleccion = st.radio(
        "Selecciona tu curso de acción:",
        datos_escenario["opciones"],
        index=datos_escenario["opciones"].index(opcion_guardada),
        key=f"radio_{idx}",
        label_visibility="collapsed"
    )

    col_nav1, col_nav2 = st.columns([1, 1])

    with col_nav1:
        if idx > 0:
            if st.button("← Regresar", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                st.session_state["paso_actual"] -= 1
                st.rerun()

    with col_nav2:
        if idx < total - 1:
            if st.button("Avanzar →", type="primary", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                st.session_state["paso_actual"] += 1
                st.rerun()
        else:
            if st.button("Finalizar travesía ✨", type="primary", use_container_width=True):
                st.session_state["respuestas"][idx] = seleccion
                
                with st.spinner("Procesando telemetría y proyectando en el plano PCA..."):
                    time.sleep(0.7)
                
                st.session_state["etapa"] = "resultados"
                st.rerun()

# =====================================================
# VISTA 3: DASHBOARD INTERACTIVO Y NARRATIVA EXPANDIDA
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
    elif "Poner una pausa" in r[3]:
        usuario["cooperacion"] += 25; usuario["adaptabilidad"] += 10
    else:
        usuario["riesgo"] += 10

    # R5: Regreso
    if "Apretar el paso" in r[4]:
        usuario["impulsividad"] += 15; usuario["riesgo"] += 10
    elif "Hacer una pausa en el andén" in r[4]:
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

    st.title("📊 Bitácora de Llegada")
    st.markdown(f"Expedicionario: **{st.session_state['nombre']}**")

    col_hero1, col_hero2 = st.columns(2)
    col_hero1.metric("Estilo de toma de decisiones", perfil_nombre)
    col_hero2.metric("Cluster de afinidad", f"Grupo #{cluster_id + 1}")

    tab_detalle, tab_mapa, tab_radar = st.tabs([
        "📖 Diagnóstico y Escenarios de Vida", 
        "📍 Posición en el Mapa (PCA)", 
        "🕸️ Huella de Competencias"
    ])

    # TAB 1: DIAGNÓSTICO PROFUNDO Y ESCENARIOS COTIDIANOS
    with tab_detalle:
        # Índices sintéticos
        idx_exp = min(100.0, round(usuario["riesgo"]*0.35 + usuario["exploracion"]*0.40 + usuario["adaptabilidad"]*0.25, 1))
        idx_est = min(100.0, round(usuario["planeacion"]*0.40 + usuario["precision"]*0.35 + usuario["cooperacion"]*0.25, 1))
        idx_pre = min(100.0, round(usuario["impulsividad"]*0.45 + usuario["riesgo"]*0.30 + usuario["adaptabilidad"]*0.25, 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("Iniciativa y Exploración", f"{idx_exp} / 100")
        c2.metric("Cuidado y Estabilidad", f"{idx_est} / 100")
        c3.metric("Respuesta Bajo Presión", f"{idx_pre} / 100")

        st.divider()

        # DICCIONARIO DE INTERPRETACIONES EXTENDIDAS
        NARRATIVAS_EXPANDIDAS = {
            "Cauteloso Analítico": {
                "que_significa": """
                Durante la ruta mostraste un estilo guiado por la prudencia, el cálculo cuidadoso y la aversión a pérdidas innecesarias. 
                No te dejas deslumbrar con facilidad por promesas arriesgadas si comprometen la seguridad del grupo o el presupuesto. 
                Eres la persona que revisa la presión de las llantas, guarda el comprobante y se asegura de que haya suficiente combustible antes de entrar a la sierra.
                """,
                "superpoder": "Generas orden, estabilidad y certidumbre en momentos donde otros se precipitan o actúan por pura ocurrencia.",
                "punto_ciego": "A veces el exceso de precaución o la búsqueda de certeza absoluta puede hacerte dudar de más y perder oportunidades valiosas que exigen subirse al tren de inmediato.",
                "estudiantil": "En entregas de proyectos, tesis o semanas de exámenes eres impecable organizando calendarios, fuentes bibliográficas y rúbricas. Tu reto: no quedarte estancado corrigiendo detalles menores cuando el trabajo ya cumple con creces el objetivo.",
                "laboral": "Excelente en gestión de riesgos, auditoría, finanzas, control de calidad y planeación de operaciones. Los equipos aprecian tu rigor técnico. Tu recomendación: practica presentar propuestas en etapas tempranas sin esperar a que estén al 100% blindadas.",
                "cotidiana": "En salidas con amigos o compras personales cuidas la cartera y prefieres itinerarios probados que eviten malos ratos. Intenta de vez en cuando aceptar planes espontáneos sin tener que planear cada minuto del fin de semana."
            },
            "Explorador Estratégico": {
                "que_significa": """
                Tu estilo se alimenta del descubrimiento, la curiosidad activa y la búsqueda de nuevos horizontes. 
                Entiendes que viajar implica asumir cierto margen de incertidumbre y que los mejores aprendizajes casi siempre están fuera del asfalto principal. 
                Sabes sopesar cuándo vale la pena asumir un costo extra o tomar una brecha si la recompensa cultural o personal lo amerita.
                """,
                "superpoder": "Inyectas visión, empuje y entusiasmo para romper la monotonía y encontrar soluciones fuera de la caja.",
                "punto_ciego": "Puedes llegar a sobreestimar tus fuerzas o los recursos disponibles, comprometiendo los tiempos de entrega por querer abarcar más terreno del que da el día.",
                "estudiantil": "Brillas en materias de innovación, investigación de campo, talleres creativos y debates. Tu reto: la constancia en tareas rutinarias o metodologías rígidas donde no hay espacio para improvisar.",
                "laboral": "Ideal para desarrollo de nuevos proyectos, áreas comerciales, innovación abierta y liderazgo de producto. Tu recomendación: haz equipo con perfiles organizados que te ayuden a documentar y aterrizar tus ideas en cronogramas firmes.",
                "cotidiana": "Eres quien propone el restaurante nuevo, la ruta alterna o el viaje a un rincón del país que nadie conocía. Recuerda verificar detalles básicos (horarios, efectivo necesario) para no convertir la aventura en un dolor de cabeza para quienes te acompañan."
            },
            "Adaptativo Resiliente": {
                "que_significa": """
                Mostraste un balance admirable entre sensatez, flexibilidad y escucha activa. 
                Tu gran virtud es que no te quiebras cuando el plan inicial se descarrila: aceptas la realidad tal como viene, respiras hondo y buscas la salida más inteligente con los recursos que tienes a la mano. 
                Además, sabes mediar cuando los ánimos se calientan en el equipo.
                """,
                "superpoder": "Capacidad camaleónica para mantener la calma, destrabar desacuerdos y sacar adelante la misión sin generar fricciones innecesarias.",
                "punto_ciego": "Por intentar conciliar con todos y no generar conflicto, puedes postergar tus propias prioridades o ceder terreno frente a posturas más dominantes.",
                "estudiantil": "Eres el compañero ideal en trabajos en equipo: ayudas a repartir cargas de manera justa y resuelves cuando alguien no entrega a tiempo. Tu reto: aprender a poner límites claros y no asumir trabajo ajeno solo para evitar tensiones.",
                "laboral": "Altamente cotizado en gestión de personas, coordinación interdepartamental, consultoría y resolución de crisis operativas. Tu recomendación: expresa con firmeza tu propia postura antes de pasar a la fase de mediación.",
                "cotidiana": "Fluyes con naturalidad en cualquier reunión o viaje familiar; sabes disfrutar tanto una comida sencilla en carretera como una cena planeada. Cuida que tu flexibilidad no te lleve a postergar tus propias decisiones personales."
            },
            "Impulsivo Bajo Presión": {
                "que_significa": """
                Frente a la urgencia, el reloj en contra y los contratiempos, tu chip interior se enciende en modo acción inmediata. 
                No toleras quedarte de brazos cruzados debatiendo teorías mientras el problema empeora frente a tus ojos. 
                Prefieres dar un paso al frente, correr el riesgo de equivocarte y corregir sobre la marcha antes que congelarte por el miedo al fallo.
                """,
                "superpoder": "Valentía operativa y velocidad de respuesta fulminante en situaciones críticas donde titubear cuesta caro.",
                "punto_ciego": "Actuar en caliente con información incompleta puede orillarte a saltarte pasos indispensables o gastar energía resolviendo consecuencias que se habrían evitado con 5 minutos de reflexión previa.",
                "estudiantil": "Rindes bajo adrenalina pura; eres capaz de sacar entregas complejas en una sola noche y reaccionar con agilidad en exámenes orales. Tu reto: no dejar todo para el último momento para evitar el desgaste físico y mental innecesario.",
                "laboral": "Excelente en áreas de respuesta a emergencias, ventas de alta velocidad, logística de última milla o ambientes de alta presión. Tu recomendación: cuenta hasta diez y consulta con al menos una persona reflexiva antes de tomar decisiones que comprometan presupuestos importantes.",
                "cotidiana": "Eres el primero en brincar cuando se poncha una llanta o hay que destrabar un trámite urgente. Cuida que esa misma intensidad no se traduzca en contestaciones tajantes con familiares o amigos cuando las cosas no van al ritmo que tú quisieras."
            }
        }

        info_perfil = NARRATIVAS_EXPANDIDAS[perfil_nombre]

        st.markdown(f"### 🧭 Tu estilo al descubierto: {perfil_nombre}")
        st.write(info_perfil["que_significa"])

        col_sp1, col_sp2 = st.columns(2)
        with col_sp1:
            st.success(f"**🌟 Tu superpoder:**\n\n{info_perfil['superpoder']}")
        with col_sp2:
            st.warning(f"**⚡ Tu punto de atención:**\n\n{info_perfil['punto_ciego']}")

        st.markdown("### 🎯 Recomendaciones prácticas por escenario")
        
        st.markdown(f"""
        <div class="scenario-box">
            <div class="scenario-header">🎓 En la vida estudiantil y académica</div>
            <p class="scenario-body">{info_perfil['estudiantil']}</p>
        </div>
        <div class="scenario-box">
            <div class="scenario-header">💼 En la vida profesional y el trabajo</div>
            <p class="scenario-body">{info_perfil['laboral']}</p>
        </div>
        <div class="scenario-box">
            <div class="scenario-header">☕ En la vida cotidiana y con amigos</div>
            <p class="scenario-body">{info_perfil['cotidiana']}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 ¿Cómo entender tus 7 puntuaciones individuales?"):
            for var in VARIABLES:
                puntaje = usuario[var]
                st.markdown(f"**{var.capitalize()}** — `{round(puntaje, 1)} pts` ({nivel(puntaje).upper()})")
                st.caption(interpretar_variable(var, puntaje))

    # TAB 2: PLOTLY INTERACTIVO (PCA)
    with tab_mapa:
        st.markdown("##### Tu coordenada frente a la población simulada (N=180)")
        colores_clusters = {0: "#7A003C", 1: "#2563EB", 2: "#16A34A", 3: "#EA580C"}
        nombres_c = {0: "Cauteloso Analítico", 1: "Explorador Estratégico", 2: "Adaptativo Resiliente", 3: "Impulsivo Bajo Presión"}

        fig_pca = go.Figure()

        for c in sorted(base["Cluster"].unique()):
            sub = base[base["Cluster"] == c]
            fig_pca.add_trace(go.Scatter(
                x=sub["PCA1"], y=sub["PCA2"],
                mode="markers",
                name=nombres_c[c],
                marker=dict(color=colores_clusters[c], size=6, opacity=0.35),
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
                textfont=dict(size=10, color="#1E293B", family="sans-serif"),
                marker=dict(color=colores_clusters[c], size=14, line=dict(color="#000", width=1.5)),
                showlegend=False
            ))

        fig_pca.add_trace(go.Scatter(
            x=[usuario_pca[0]], y=[usuario_pca[1]],
            mode="markers+text",
            text=["¡TÚ AQUÍ!"],
            textposition="bottom center",
            textfont=dict(size=12, color="#DC2626", family="sans-serif"),
            marker=dict(color="#DC2626", size=18, symbol="x", line=dict(width=3)),
            name="Tu Posición"
        ))

        fig_pca.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            xaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    # TAB 3: RADAR CHART (TELARAÑA)
    with tab_radar:
        st.markdown("##### Desglose radial de las 7 dimensiones conductuales")
        categories = [v.capitalize() for v in VARIABLES]
        values = [usuario[v] for v in VARIABLES]
        categories.append(categories[0])
        values.append(values[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(122, 0, 60, 0.18)',
            line=dict(color='#7A003C', width=2),
            name="Tu Perfil"
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], color="#94A3B8"),
                bgcolor="#FFFFFF"
            ),
            height=380,
            margin=dict(l=30, r=30, t=20, b=30),
            showlegend=False
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.write("")
    if st.button("Volver a iniciar la expedición", use_container_width=True):
        st.session_state["etapa"] = "intro"
        st.session_state["paso_actual"] = 0
        st.session_state["respuestas"] = {}
        st.rerun()
