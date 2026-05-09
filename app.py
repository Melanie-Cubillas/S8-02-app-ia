import streamlit as st
import pymongo
from google import genai
from google.genai import types

# =======================
# CONFIGURACIÓN
# =======================

st.set_page_config(
    page_title="ArtVision IA",
    page_icon="🎨",
    layout="wide"
)

GOOGLE_API_KEY = st.secrets["app"]["GOOGLE_API_KEY"]
MONGODB_URI = st.secrets["app"]["MONGODB_URI"]

if not GOOGLE_API_KEY or not MONGODB_URI:
    st.error("Faltan GOOGLE_API_KEY o MONGODB_URI.")
    st.stop()

# =======================
# ESTILOS
# =======================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff8ef 0%, #f6eadb 50%, #ecd8c2 100%);
}

/* ===== TITULO ===== */

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #2b1408;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4a2a16;
    margin-bottom: 40px;
    font-weight: 500;
}

/* ===== CARDS ===== */

.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0px 8px 24px rgba(80, 45, 20, 0.12);
    border: 1px solid #d8c2aa;
    margin-bottom: 20px;
}

/* ===== RESULTADO ===== */

.result-card {
    background: #ffffff;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0px 8px 24px rgba(80, 45, 20, 0.14);
    border-left: 8px solid #8b4a20;
    color: #1f120b;
    font-size: 17px;
    line-height: 1.9;
    font-weight: 500;
}

/* ===== BADGE ===== */

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background-color: #ead0b3;
    color: #3a1f10;
    font-weight: 800;
    margin-bottom: 10px;
}

/* ===== TITULOS ===== */

.section-title {
    color: #2b1408;
    font-size: 30px;
    font-weight: 800;
    margin-top: 20px;
}

/* ===== TEXTOS ===== */

h1, h2, h3, h4, h5, h6 {
    color: #2b1408 !important;
}

p, label, div {
    color: #2d1a10;
}

/* ===== INPUTS ===== */

textarea {
    border-radius: 14px !important;
    color: #1f120b !important;
    background-color: #fffdf9 !important;
}

/* ===== BOTON ANALIZAR ===== */

div.stButton > button {
    background: linear-gradient(90deg, #7a3c15, #b87434);
    color: white;
    border-radius: 14px;
    height: 52px;
    font-weight: 700;
    border: none;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #5f2d0f, #9f6228);
    color: white;
}

/* ===== FILE UPLOADER ===== */

section[data-testid="stFileUploader"] {
    color: #2b1408;
}

/* Caja del uploader */

[data-testid="stFileUploader"] section {
    border: 2px dashed #8b4a20 !important;
    border-radius: 16px !important;
    background-color: #fffaf4 !important;
    padding: 18px !important;
}

/* BOTON UPLOAD */

[data-testid="stBaseButton-secondary"] {
    background: #f5f0ed !important;
    color: black !important;
    border-radius: 12px !important;
    border: 1px solid #cdb8a6 !important;
    font-weight: 700 !important;
}

/* HOVER BOTON */

[data-testid="stBaseButton-secondary"]:hover {
    background: #ebe2dc !important;
    color: black !important;
    border: 1px solid #b99d86 !important;
}

/* Texto interno uploader */

[data-testid="stFileUploader"] small {
    color: #4a2a16 !important;
    font-weight: 600;
}

/* ===== EXPANDERS ===== */

.streamlit-expanderHeader {
    color: #2b1408 !important;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# =======================
# CLIENTES
# =======================

@st.cache_resource
def get_genai_client():
    return genai.Client(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_mongo_collection():
    client = pymongo.MongoClient(MONGODB_URI)
    db = client["pdf_embeddings_db2"]
    return db["pdf_vectors"]

client_genai = get_genai_client()
collection = get_mongo_collection()

# =======================
# FUNCIONES
# =======================

def crear_embedding(texto: str):

    response = client_genai.models.embed_content(
        model="gemini-embedding-001",
        contents=texto,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    return response.embeddings[0].values


def buscar_similares(embedding, k=5):

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embedding,
                "numCandidates": 100,
                "limit": k,
            }
        },
        {
            "$project": {
                "_id": 0,
                "texto": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]

    return list(collection.aggregate(pipeline))


def describir_imagen(imagen):

    bytes_imagen = imagen.getvalue()

    prompt = """
    Analiza esta pintura como un experto en historia del arte.

    Describe:
    - colores
    - composición
    - iluminación
    - personajes
    - estilo artístico
    - símbolos visuales
    - posibles obras famosas relacionadas

    Sé preciso y detallado.
    """

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(
                data=bytes_imagen,
                mime_type=imagen.type
            )
        ]
    )

    return response.text


def generar_respuesta(pregunta, descripcion_imagen, contextos):

    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Eres ArtVision IA, un experto en pinturas famosas,
historia del arte y análisis visual de obras.

Analiza la imagen subida por el usuario y compárala
con la base de conocimiento del PDF.

No respondas en un solo párrafo.
No uses saludos.
No digas que eres una IA.
No menciones que la información lo obtuviste del pdf.

Contexto:
{contexto}

Descripción visual:
{descripcion_imagen}

Pregunta:
{pregunta}

Responde EXACTAMENTE con este formato:

### 🖼️ Obra probable
Nombre de la obra.

### 👨‍🎨 Autor probable
Nombre del autor.

### 🏛️ Movimiento artístico
Movimiento artístico.

### 📖 Explicación
Explica por qué coincide con esa obra.

"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# =======================
# INTERFAZ
# =======================

st.markdown(
    '<div class="main-title">🎨 ArtVision IA</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Reconoce pinturas famosas usando visión artificial, Gemini y MongoDB Atlas.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.1, 0.9], gap="large")

# =======================
# COLUMNA IZQUIERDA
# =======================

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🖼️ Carga una pintura")

    st.write(
        "Sube una imagen de una obra de arte para identificar "
        "su autor, estilo y características visuales."
    )

    imagen = st.file_uploader(
        "Sube imagen",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if imagen:
        st.image(
            imagen,
            caption="Pintura cargada",
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# COLUMNA DERECHA
# =======================

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🧠 Consulta al experto")

    pregunta = st.text_area(
        "Pregunta",
        value="¿Qué pintura es y quién es su autor?",
        height=180,
        label_visibility="collapsed"
    )

    analizar = st.button(
        "🔍 Analizar pintura",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =======================
# RESULTADO
# =======================

if analizar:

    if not imagen:
        st.warning("Debes subir una imagen.")
    else:

        with st.spinner("Analizando pintura..."):

            try:

                descripcion = describir_imagen(imagen)

                consulta = pregunta + "\n" + descripcion

                embedding = crear_embedding(consulta)

                similares = buscar_similares(embedding, k=5)

                if not similares:
                    respuesta = "No encontré información relevante."
                else:
                    respuesta = generar_respuesta(
                        pregunta,
                        descripcion,
                        similares
                    )

                st.markdown(
                    '<div class="section-title">🖌️ Resultado del análisis</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class="result-card">
                        <span class="badge">Análisis artístico</span>
                        <br><br>
                        {respuesta.replace(chr(10), "<br>")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                with st.expander("🧠 Descripción visual generada por Gemini"):
                    st.write(descripcion)

                with st.expander("📚 Fragmentos recuperados del PDF"):

                    for i, c in enumerate(similares, 1):

                        st.markdown(
                            f"**Fragmento {i}** — score: `{c['score']:.4f}`"
                        )

                        st.write(c["texto"][:700])

                        st.divider()

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")
