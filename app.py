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
    page_title="Expedición México · Simulación de Decisiones",
    page_icon="🧭",
    layout="wide"
)

# =====================================================
# IDENTIDAD VISUAL: UNIVERSIDAD NACIONAL ROSARIO CASTELLANOS
# =====================================================
st.markdown("""
<style>
.stApp {
    background: #F7F8FA;
    color: #1E293B;
}

.block-container {
    max-width: 1040px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

h1 {
    color: #7A003C !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.4rem !important;
}

h2 {
    color: #7A003C !important;
    font-weight: 700 !important;
    margin-top: 1.8rem !important;
    font-size: 1.45rem !important;
}

h3 {
    color: #9A1B5A !important;
    font-weight: 700 !important;
}

.card {
    background: #FFFFFF;
    padding: 1.8rem 2rem;
    border-radius: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    margin-bottom: 1.5rem;
}

.interpretacion {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    border: 1px solid #E5E7EB;
    margin-bottom: 0.8rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.02);
}

.stButton button {
    background: #7A003C !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 1.6rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(122,0,60,0.18);
    transition: all 0.2s ease;
}

.stButton button:hover {
    background: #9A1B5A !important;
    transform: translateY(-1px);
}

.stRadio > div {
    background: #FFFFFF;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid #E5E7EB;
    margin-bottom: 0.8rem;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    border: 1px solid #E5E7EB;
    box-shadow: 0 4px 14px rgba(0,0,0,0.03);
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
        [85, 58, 52, 48, 88, 42, 50]   # Reactivo / Impulsivo
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
# INTERPRETACIONES Y REGLAS
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
    perfiles = {
        0: "Cauteloso Analítico",
        1: "Explorador Estratégico",
        2: "Adaptativo Resiliente",
        3: "Impulsivo Bajo Presión"
    }
    return perfiles[c]

# =====================================================
# LANDING
# =====================================================
st.title("🧭 Expedición México")
st.markdown("##### Una travesía interactiva para conocer tu estilo al resolver dilemas en ruta.")

with st.expander("ℹ️ Acerca de esta simulación", expanded=False):
    st.markdown("""
    **Expedición México** pone a prueba tu criterio frente a imprevistos en ruta a través de modelos de ciencia de datos:
    * **Clustering Conductual:** Agrupamiento de patrones de decisión con *K-Means* ($k=4$).
    * **Reducción Dimensional:** Visualización espacial de tus elecciones mediante *PCA*.
    * **Variables analizadas:** Riesgo, exploración, precisión, adaptabilidad, impulsividad, planeación y cooperación.
    
    *Nota: Este ejercicio es una simulación formativa de toma de decisiones; no constituye una valoración clínica ni psicológica.*
    """)

st.divider()

# =====================================================
# CONTROL DE SESIÓN
# =====================================================
if "iniciado" not in st.session_state:
    st.session_state["iniciado"] = False

if not st.session_state["iniciado"]:
    st.header("Punto de reunión")
    nombre_input = st.text_input("¿Cómo te gustaría registrarte en la bitácora?", placeholder="Nombre o alias")
    
    if st.button("Subir a la expedición"):
        st.session_state["nombre"] = nombre_input.strip() if nombre_input.strip() else "Viajero"
        st.session_state["iniciado"] = True
        st.rerun()

# =====================================================
# DILEMAS EN RUTA (NARRATIVA EN ESPAÑOL DE MÉXICO)
# =====================================================
if st.session_state["iniciado"]:
    st.success(f"Todo listo en la bitácora. Buen viaje, **{st.session_state.get('nombre', 'Viajero')}**.")

    with st.form("formulario_expedicion"):
        
        # Parada 1: Altiplano Central
        st.header("1. La salida — Altiplano Central")
        r1 = st.radio(
            "La camioneta sale al amanecer. Tienen la ruta marcada por autopista de cuota, pero un contacto local les avisa de una desviación por carretera libre donde hay una zona arqueológica poco conocida y sin filtros de acceso. ¿Qué decides?",
            [
                "Seguir por la autopista directa; la prioridad es llegar a tiempo y sin sorpresas mecánicas o de tráfico.",
                "Tomar la carretera libre un tramo: permite conocer el paisaje sin desviarse demasiado del tiempo estimado.",
                "Irse directo por la brecha comunitaria; las mejores historias del viaje no están sobre el asfalto principal."
            ],
            index=0
        )

        # Parada 2: Valles de Oaxaca
        st.header("2. El dilema del presupuesto — Sierra de Oaxaca")
        r2 = st.radio(
            "En la sierra les ofrecen contratar camionetas 4x4 y guías de la comunidad para subir a un mirador en la niebla antes de que caiga el aguacero de la tarde. La vuelta consume una parte importante de la lana del viaje. ¿Cómo lo manejas?",
            [
                "Cuidar la reserva y hacer el recorrido a pie por los senderos autorizados que no generan costo extra.",
                "Pagar solo una parte del recorrido para conocer el primer paraje y guardar un colchón para imprevistos.",
                "Pagar completo el recorrido de una vez; una vista así difícilmente se repite y el dinero se repone después."
            ],
            index=0
        )

        # Parada 3: Selva de Yucatán
        st.header("3. Sin señal — Península de Yucatán")
        r3 = st.radio(
            "Buscando el acceso a un cenote en la selva baja, los celulares se quedan sin señal y el mapa descargado no coincide con una bifurcación recién abierta. Queda apenas hora y media de luz. ¿Qué instrucción das?",
            [
                "Hacer alto total, no arriesgar la camioneta y esperar al día siguiente con luz clara para pedir informes en el pueblo.",
                "Bajar a revisar huellas de llantas y referencias del camino para avanzar despacio, metro a metro.",
                "Agarrar el camino más ancho a buen paso para ganar terreno antes de que se oculte el sol por completo."
            ],
            index=0
        )

        # Parada 4: Desacuerdo en el equipo
        st.header("4. La sobremesa tensa — El campamento")
        r4 = st.radio(
            "El pronóstico anuncia frente frío y la camioneta viene algo justa de frenos. En la cena se desata la discusión: la mitad quiere regresar a descansar a una cabaña segura y la otra insiste en subir al cerro cueste lo que cueste. ¿Cómo intervienes?",
            [
                "Recordar el plan original de la salida y pedirle a todos que nos apeguemos a lo acordado desde el principio.",
                "Poner una pausa de cinco minutos para escuchar ambas posturas y buscar una opción intermedia que calme los ánimos.",
                "Cortar la discusión de tajo y proponer cambiar el rumbo completo hacia un destino distinto que nadie traía en mente."
            ],
            index=0
        )

        # Parada 5: El regreso
        st.header("5. El cierre de la jornada — Terminal Pantitlán, CDMX")
        r5 = st.radio(
            "De regreso en la ciudad, llegan a la estación de transbordo cargando mochilas y equipo justo cuando suena el timbre de cierre del último tren de la noche. La entrada está llena de gente. ¿Cómo reaccionas?",
            [
                "Apretar el paso esquivando a la gente para alcanzar a entrar al vagón antes de que cierren las puertas.",
                "Hacer una pausa en el andén exterior, abrir la aplicación de movilidad y ver qué opciones de trolebús o metrobús siguen activas.",
                "Pedir un taxi de aplicación entre todos aunque cueste más; lo importante es llegar a casa seguros y con el equipo completo."
            ],
            index=0
        )

        enviar = st.form_submit_button("Consultar mi bitácora final")

    # =====================================================
    # MOTOR CONDUCTUAL
    # =====================================================
    if enviar:
        usuario = {var: 50.0 for var in VARIABLES}

        # R1: Altiplano
        if "autopista" in r1:
            usuario["riesgo"] -= 20
            usuario["precision"] += 20
            usuario["planeacion"] += 20
        elif "carretera libre" in r1:
            usuario["adaptabilidad"] += 15
            usuario["exploracion"] += 10
        else:
            usuario["riesgo"] += 30
            usuario["exploracion"] += 30
            usuario["impulsividad"] += 10

        # R2: Oaxaca
        if "Cuidar la reserva" in r2:
            usuario["planeacion"] += 20
            usuario["precision"] += 15
        elif "Pagar solo una parte" in r2:
            usuario["adaptabilidad"] += 15
        else:
            usuario["riesgo"] += 15
            usuario["impulsividad"] += 20

        # R3: Yucatán
        if "Hacer alto total" in r3:
            usuario["precision"] += 15
            usuario["impulsividad"] -= 10
        elif "Bajar a revisar huellas" in r3:
            usuario["adaptabilidad"] += 20
        else:
            usuario["riesgo"] += 20
            usuario["impulsividad"] += 25

        # R4: Campamento
        if "Recordar el plan" in r4:
            usuario["cooperacion"] -= 15
        elif "Poner una pausa" in r4:
            usuario["cooperacion"] += 25
            usuario["adaptabilidad"] += 10
        else:
            usuario["riesgo"] += 10

        # R5: Regreso
        if "Apretar el paso" in r5:
            usuario["impulsividad"] += 15
            usuario["riesgo"] += 10
        elif "Hacer una pausa en el andén" in r5:
            usuario["planeacion"] += 15
            usuario["precision"] += 15
        else:
            usuario["adaptabilidad"] += 15

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
# REPORTE DE RESULTADOS
# =====================================================
if st.session_state.get("procesado"):
    usuario = st.session_state["usuario"]
    cluster_idx = st.session_state["cluster"]
    
    st.divider()
    st.header("📊 Lectura de tu recorrido")

    cluster_nombres = {
        0: "Cluster Analítico",
        1: "Cluster Explorador",
        2: "Cluster Adaptativo",
        3: "Cluster Reactivo"
    }

    c_col1, c_col2 = st.columns(2)
    c_col1.metric("Estilo de toma de decisiones", st.session_state["perfil"])
    c_col2.metric("Grupo en el modelo (K-Means)", cluster_nombres[cluster_idx])

    # 1. Variables
    st.subheader("1. Balance por dimensión en la expedición")
    
    for var in VARIABLES:
        st.markdown(f"""
        <div class="interpretacion">
            <h4 style="color: #7A003C; margin-bottom: 4px;">{var.capitalize()} — Rango {nivel(usuario[var])}</h4>
            <p style="margin-bottom: 4px;"><strong>Puntaje:</strong> {round(usuario[var], 1)} / 100</p>
            <p style="color: #475569; margin: 0;">{interpretar_variable(var, usuario[var])}</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Índices integrados
    st.subheader("2. Indicadores de dinámica en ruta")

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
        st.metric("Iniciativa y Exploración", f"{indice_exploracion} / 100")
        st.progress(indice_exploracion / 100.0)
    with idx_c2:
        st.metric("Cuidado y Estabilidad", f"{indice_estabilidad} / 100")
        st.progress(indice_estabilidad / 100.0)
    with idx_c3:
        st.metric("Respuesta Bajo Presión", f"{indice_presion} / 100")
        st.progress(indice_presion / 100.0)

    # 3. Mapa PCA
    st.subheader("3. Tu posición en el mapa de perfiles (PCA)")

    colores = {0: "#7A003C", 1: "#2563EB", 2: "#16A34A", 3: "#EA580C"}
    nombres_clusters = {
        0: "Cauteloso Analítico", 1: "Explorador Estratégico",
        2: "Adaptativo Resiliente", 3: "Impulsivo Bajo Presión"
    }

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor("#F7F8FA")
    ax.set_facecolor("#FFFFFF")

    for c in sorted(base["Cluster"].unique()):
        c_data = base[base["Cluster"] == c]
        ax.scatter(
            c_data["PCA1"], c_data["PCA2"],
            color=colores[c], alpha=0.25, s=35,
            label=nombres_clusters[c]
        )

    for c in sorted(base["Cluster"].unique()):
        cx = base[base["Cluster"] == c]["PCA1"].mean()
        cy = base[base["Cluster"] == c]["PCA2"].mean()
        ax.scatter(cx, cy, color=colores[c], s=240, edgecolor="#1E293B", linewidth=1.8, zorder=4)
        ax.text(cx, cy + 0.3, nombres_clusters[c], fontsize=8.5, ha="center", weight="bold", color="#1E293B")

    u_pca = st.session_state["usuario_pca"][0]
    ax.scatter(u_pca[0], u_pca[1], color="#DC2626", s=380, marker="X", edgecolor="#000000", linewidth=2, label="Tu posición", zorder=5)

    ax.set_title("Distribución en el plano de componentes principales", fontsize=12, fontweight="bold", pad=12, color="#1E293B")
    ax.grid(alpha=0.18, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5)

    st.pyplot(fig)

    # 4. Narrativa de cierre
    st.subheader("4. Síntesis de tu perfil")

    narrativas = {
        "Cauteloso Analítico": {
            "significado": "En la ruta mostraste preferencia por la estabilidad y por tener claras las variables antes de comprometer recursos. Prefieres cuidar los márgenes de seguridad, no arriesgar la camioneta de más y asegurar que el equipo cumpla con el plan trazado.",
            "patrones": "Tus decisiones reflejan planeación, sobriedad en el gasto y prudencia ante imprevistos. Te tomas un momento para revisar opciones antes de actuar y mantienes la cabeza fría frente a la presión del reloj.",
            "implicaciones": "En proyectos de equipo y vida universitaria, este estilo es el ancla que evita desviaciones costosas y descuidos. Sin embargo, en escenarios donde todo cambia de un minuto a otro, conviene soltar un poco el libreto para no perder oportunidades por exceso de precaución."
        },
        "Explorador Estratégico": {
            "significado": "Tu estilo se orienta a descubrir y aprovechar oportunidades en el camino. Asumes la incertidumbre como parte natural del viaje y estás dispuesto a pagar el costo de explorar si la experiencia o el aprendizaje lo ameritan.",
            "patrones": "Aparece una marcada disposición a tomar rutas alternas, avanzar aun cuando la información esté incompleta y buscar novedades sin descuidar del todo la viabilidad del trayecto.",
            "implicaciones": "Este perfil aporta empuje, curiosidad e innovación en cualquier grupo de trabajo. El punto de atención radica en no sobreestimar las reservas de tiempo o presupuesto para no comprometer el regreso a casa."
        },
        "Adaptativo Resiliente": {
            "significado": "Mostraste un balance equilibrado entre sensatez, flexibilidad y escucha activa. No te aferras a planes rígidos si las circunstancias sobre el terreno cambian, pero tampoco tomas decisiones a tontas y a locas.",
            "patrones": "Destaca tu habilidad para destrabar desacuerdos, ajustar el itinerario con los recursos disponibles y encontrar salidas prácticas que mantengan al grupo unido y en marcha.",
            "implicaciones": "En contextos colaborativos y de alta demanda, eres quien mantiene la armonía operativa y saca los proyectos adelante frente a los tropiezos cotidianos."
        },
        "Impulsivo Bajo Presión": {
            "significado": "Frente a la urgencia y los imprevistos, tu reacción inmediata es resolver sobre la marcha y con determinación. Prefieres equivocarte intentando avanzar antes que quedarte varado analizando la situación.",
            "patrones": "Tus respuestas priorizan velocidad, resolución en caliente y una tolerancia alta a la falta de certezas cuando el tiempo apremia.",
            "implicaciones": "Es una cualidad muy valiosa en momentos de emergencia donde titubear cuesta caro. En decisiones complejas o de mediano plazo, una pausa breve para consultar con el equipo te ahorrará vueltas innecesarias."
        }
    }

    perfil_actual = st.session_state["perfil"]
    st.markdown(f"**¿Cómo se traduce en el terreno?**\n\n{narrativas[perfil_actual]['significado']}")
    st.markdown(f"**Tendencias de decisión observadas:**\n\n{narrativas[perfil_actual]['patrones']}")
    st.markdown(f"**Recomendaciones para el trabajo colaborativo:**\n\n{narrativas[perfil_actual]['implicaciones']}")

    st.caption("Simulación desarrollada con fines formativos y de divulgación en ciencia de datos aplicada a la conducta.")

    if st.button("Iniciar un nuevo recorrido"):
        for key in ["iniciado", "usuario", "cluster", "perfil", "usuario_pca", "procesado"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
