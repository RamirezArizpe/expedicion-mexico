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
    max-width: 820px;
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

/* Métricas del Dashboard */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 14px rgba(0,0,0,0.02);
}

.metric-card-custom {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1px solid #E2E8F0;
    margin-bottom: 0.65rem;
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
            "alto": "Aceptas la incertidumbre con soltura y te atraen los escenarios con margen amplio de ganancia o pérdida.",
            "moderado": "Mides el terreno antes de dar el paso; buscas un balance sano entre audacia y estabilidad.",
            "bajo": "Prefieres el terreno firme, cuidando siempre tener el control de los posibles costos."
        },
        "exploracion": {
            "alto": "Muestras curiosidad constante por abrir brecha y probar caminos poco transitados.",
            "moderado": "Equilibras bien la curiosidad por lo nuevo con paradas seguras y conocidas.",
            "bajo": "Te apoyas en rutas probadas, optimizando sobre lo seguro antes de improvisar."
        },
        "precision": {
            "alto": "Prestas atención minuciosa al detalle y buscas claridad antes de comprometer una decisión.",
            "moderado": "Cuidas los detalles esenciales sin dejar que el análisis frene el avance del grupo.",
            "bajo": "Priorizas el ritmo y la agilidad de movimiento por encima de la exactitud milimétrica."
        },
        "adaptabilidad": {
            "alto": "Te acomodas con rapidez a los giros inesperados y resuelves con lo que haya a la mano.",
            "moderado": "Respondes de forma práctica frente a imprevistos cotidianos sin perder el temple.",
            "bajo": "Trabajas mucho mejor cuando las reglas, los tiempos y las condiciones son claras desde el inicio."
        },
        "impulsividad": {
            "alto": "Tiendes a responder de golpe cuando arrecia la presión del tiempo o el entorno.",
            "moderado": "Sabes alternar entre reacciones inmediatas y pausas breves de valoración.",
            "bajo": "Prefieres procesar la situación con calma antes de mover una sola pieza."
        },
        "planeacion": {
            "alto": "Tu brújula está en la anticipación, el orden y tener previstos los pasos siguientes.",
            "moderado": "Trazas un plan base claro, pero mantienes margen de maniobra sobre la marcha.",
            "bajo": "Te orientas a la acción sobre el terreno; prefieres resolver conforme surgen las cosas."
        },
        "cooperacion": {
            "alto": "Buscas sumar puntos de vista y construir acuerdos donde todo el equipo se sienta parte.",
            "moderado": "Sabes conciliar con el grupo sin descuidar tu propio criterio y autonomía.",
            "bajo": "Confías primordialmente en tu criterio individual para no entorpecer el avance."
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
    st.session_state["etapa"] = "intro" # intro -> wizard -> resultados
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
    st.markdown("##### Una experiencia de toma de decisiones bajo incertidumbre.")
    
    st.markdown("""
    <div class="station-card">
        <h3 style="margin-top:0;">Bienvenido a la expedición</h3>
        <p style="color: #475569; line-height: 1.6;">
            A lo largo de cinco paradas clave en la geografía mexicana, enfrentarás dilemas típicos de ruta donde los recursos, el tiempo o el clima te orillarán a tomar decisiones.
        </p>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            Detrás de cada respuesta, un modelo de <strong>Clustering Conductual (K-Means)</strong> y <strong>Reducción Dimensional (PCA)</strong> proyectará tu estilo de resolución frente a perfiles simulados.
        </p>
        </p>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            <strong>Está accediendo a un prototipo de evaluación, los resultados no representan un diagnóstico psicológico y/o clínico, forman parte de una simulación educativa.</strong>
        </p>
        </p>
        <p style="color: #475569; line-height: 1.6; margin-bottom: 0;">
            <strong>Su participación y permanencia es totalmente voluntaria.</strong>
        </p
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

    # Barra de progreso superior
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

    # Radio button estilizado
    opcion_guardada = st.session_state["respuestas"].get(idx, datos_escenario["opciones"][0])
    seleccion = st.radio(
        "Selecciona tu curso de acción:",
        datos_escenario["opciones"],
        index=datos_escenario["opciones"].index(opcion_guardada),
        key=f"radio_{idx}",
        label_visibility="collapsed"
    )

    # Navegación entre pasos
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
                
                # Microanimación de cálculo
                with st.spinner("Procesando telemetría y proyectando en el plano PCA..."):
                    time.sleep(0.7)
                
                st.session_state["etapa"] = "resultados"
                st.rerun()

# =====================================================
# VISTA 3: DASHBOARD INTERACTIVO DE RESULTADOS
# =====================================================
elif st.session_state["etapa"] == "resultados":
    # 1. Mapeo de puntajes
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

    # Transformación matemática
    usuario_df = pd.DataFrame([usuario])
    usuario_scaled = scaler.transform(usuario_df[VARIABLES])
    usuario_pca = pca.transform(usuario_scaled)[0]
    cluster_id = int(kmeans.predict([usuario_pca])[0])
    perfil_nombre = perfil_cluster(cluster_id)

    # Encabezado de Resultados
    st.title("📊 Bitácora de Llegada")
    st.markdown(f"Expedicionario: **{st.session_state['nombre']}**")

    col_hero1, col_hero2 = st.columns(2)
    col_hero1.metric("Estilo de toma de decisiones", perfil_nombre)
    col_hero2.metric("Cluster de afinidad", f"Grupo #{cluster_id + 1}")

    # Tabs de visualización para no amontonar
    tab_mapa, tab_radar, tab_detalle = st.tabs([
        "📍 Posición en el Mapa (PCA)", 
        "🕸️ Huella de Competencias", 
        "📝 Diagnóstico e Índices"
    ])

    # TAB 1: PLOTLY INTERACTIVO (PCA)
    with tab_mapa:
        st.markdown("##### Tu coordenada frente a la población simulada (N=180)")
        colores_clusters = {0: "#7A003C", 1: "#2563EB", 2: "#16A34A", 3: "#EA580C"}
        nombres_c = {0: "Cauteloso Analítico", 1: "Explorador Estratégico", 2: "Adaptativo Resiliente", 3: "Impulsivo Bajo Presión"}

        fig_pca = go.Figure()

        # Scatter de la población
        for c in sorted(base["Cluster"].unique()):
            sub = base[base["Cluster"] == c]
            fig_pca.add_trace(go.Scatter(
                x=sub["PCA1"], y=sub["PCA2"],
                mode="markers",
                name=nombres_c[c],
                marker=dict(color=colores_clusters[c], size=6, opacity=0.35),
                hoverinfo="name"
            ))

        # Centroides
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

        # Posición del Usuario
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

    # TAB 2: RADAR CHART (TELARAÑA)
    with tab_radar:
        st.markdown("##### Desglose radial de las 7 dimensiones conductuales")
        categories = [v.capitalize() for v in VARIABLES]
        values = [usuario[v] for v in VARIABLES]
        # Cerrar el lazo
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

    # TAB 3: DIAGNÓSTICO Y TEXTO APLICADO
    with tab_detalle:
        # Índices sintéticos
        idx_exp = min(100.0, round(usuario["riesgo"]*0.35 + usuario["exploracion"]*0.40 + usuario["adaptabilidad"]*0.25, 1))
        idx_est = min(100.0, round(usuario["planeacion"]*0.40 + usuario["precision"]*0.35 + usuario["cooperacion"]*0.25, 1))
        idx_pre = min(100.0, round(usuario["impulsividad"]*0.45 + usuario["riesgo"]*0.30 + usuario["adaptabilidad"]*0.25, 1))

        c1, c2, c3 = st.columns(3)
        c1.metric("Exploración", f"{idx_exp}/100")
        c2.metric("Estabilidad", f"{idx_est}/100")
        c3.metric("Bajo Presión", f"{idx_pre}/100")

        st.divider()

        narrativas = {
            "Cauteloso Analítico": {
                "significado": "En la ruta mostraste preferencia por la estabilidad y por tener claras las variables antes de comprometer recursos. Cuidas los márgenes de seguridad para no arriesgar de más la misión.",
                "patrones": "Tus decisiones reflejan planeación, sobriedad en el gasto y prudencia ante imprevistos. Te tomas un momento para revisar opciones antes de actuar.",
                "implicaciones": "En proyectos de equipo, este estilo es el ancla que evita descuidos costosos. En escenarios ultra dinámicos, conviene soltar un poco el libreto para no perder ritmo."
            },
            "Explorador Estratégico": {
                "significado": "Tu estilo se orienta a descubrir y aprovechar oportunidades en el camino. Asumes la incertidumbre como parte natural del viaje si la experiencia lo amerita.",
                "patrones": "Disposición constante a tomar rutas alternas, avanzar con información parcial y probar lo desconocido sin descuidar la viabilidad del trayecto.",
                "implicaciones": "Aportas dinamismo y visión de avance. El punto de atención radica en no subestimar los tiempos y reservas para asegurar el regreso."
            },
            "Adaptativo Resiliente": {
                "significado": "Mostraste un balance sano entre sensatez, flexibilidad y escucha activa. No te aferras a planes rígidos si las circunstancias sobre el terreno cambian.",
                "patrones": "Habilidad para destrabar desacuerdos en el grupo, ajustar el itinerario con lo que hay a la mano y encontrar salidas prácticas.",
                "implicaciones": "En contextos colaborativos, eres quien mantiene la armonía operativa y saca adelante las tareas frente a los tropiezos cotidianos."
            },
            "Impulsivo Bajo Presión": {
                "significado": "Frente a la urgencia, tu reacción es resolver sobre la marcha y con determinación. Prefieres moverte antes que quedarte varado analizando de más.",
                "patrones": "Tus respuestas priorizan velocidad, resolución en caliente y tolerancia alta a la falta de certezas cuando el tiempo apremia.",
                "implicaciones": "Cualidad clave en momentos críticos. En proyectos de mediano plazo, una pausa breve para consultar con el equipo te evitará retrabajos."
            }
        }

        diag = narrativas[perfil_nombre]
        st.markdown(f"**En el terreno:** {diag['significado']}")
        st.markdown(f"**Tendencia detectada:** {diag['patrones']}")
        st.markdown(f"**Recomendación práctica:** {diag['implicaciones']}")

    st.write("")
    if st.button("Volver a iniciar la expedición", use_container_width=True):
        st.session_state["etapa"] = "intro"
        st.session_state["paso_actual"] = 0
        st.session_state["respuestas"] = {}
        st.rerun()
