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
    st.error("Faltan GOOGLE_API_KEY o MONGODB_URI en secrets.")
    st.stop()

# =======================
# ESTILO
# =======================

st.markdown("""
<style>
.main {
    background-color: #fff8f0;
}
.titulo {
    text-align: center;
    color: #5a2d0c;
    font-size: 42px;
    font-weight: bold;
}
.subtitulo {
    text-align: center;
    color: #7a4a24;
    font-size: 18px;
}
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
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
    Analiza esta imagen como un experto en historia del arte.
    Describe la pintura, sus colores, composición, estilo artístico,
    personajes, símbolos visuales y posibles obras famosas relacionadas.
    No inventes datos definitivos si no estás seguro.
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


def generar_respuesta(pregunta: str, descripcion_imagen: str, contextos: list[dict]) -> str:
    contexto = "\n\n".join([c["texto"] for c in contextos])

    prompt = f"""
Eres ArtVision IA, un asistente experto en pinturas famosas, historia del arte,
autores, movimientos artísticos y reconocimiento visual de obras.

Tu tarea es analizar la imagen subida por el usuario y compararla con la base de conocimiento
del PDF. Responde como un experto en pinturas famosas.

Usa principalmente el contexto del PDF. Si no hay suficiente información, dilo claramente.

Contexto del PDF:
{contexto}

Descripción visual generada a partir de la imagen:
{descripcion_imagen}

Pregunta del usuario:
{pregunta}

Responde en español con este formato:

Obra probable:
Autor probable:
Movimiento artístico:
Explicación:
Nivel de seguridad:
"""

    response = client_genai.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# =======================
# INTERFAZ
# =======================

st.markdown('<p class="titulo">🎨 ArtVision IA</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Reconoce pinturas famosas usando Gemini, MongoDB Atlas y Streamlit</p>',
    unsafe_allow_html=True
)

st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("🖼️ Sube una imagen de una pintura")
    imagen = st.file_uploader(
        "Carga una imagen en formato JPG, JPEG o PNG",
        type=["jpg", "jpeg", "png"]
    )

    if imagen:
        st.image(imagen, caption="Imagen cargada", use_container_width=True)

with col2:
    st.markdown("🧠 Pregunta al experto")
    pregunta = st.text_area(
        "Escribe tu consulta",
        value="¿Qué pintura es y quién es su autor?",
        height=120
    )

    analizar = st.button("🔍 Analizar pintura", use_container_width=True)

st.divider()

if "historial" not in st.session_state:
    st.session_state.historial = []

if analizar:
    if not imagen:
        st.warning("Primero debes subir una imagen.")
    else:
        with st.spinner("Analizando la pintura con Gemini..."):
            try:
                descripcion = describir_imagen(imagen)

                consulta_busqueda = pregunta + "\n" + descripcion
                embedding = crear_embedding(consulta_busqueda)

                similares = buscar_similares(embedding, k=5)

                if not similares:
                    respuesta = "No encontré información relevante en la base de conocimiento."
                else:
                    respuesta = generar_respuesta(pregunta, descripcion, similares)

                st.markdown("## 🖌️ Resultado del análisis")
                st.success(respuesta)

                st.session_state.historial.append({
                    "pregunta": pregunta,
                    "respuesta": respuesta
                })

                with st.expander("📝 Descripción visual generada por Gemini"):
                    st.write(descripcion)

                with st.expander("📚 Fragmentos recuperados del PDF"):
                    for i, c in enumerate(similares, 1):
                        st.markdown(f"**Fragmento {i}** — score: `{c['score']:.4f}`")
                        st.write(c["texto"][:700])
                        st.divider()

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

# =======================
# HISTORIAL
# =======================

if st.session_state.historial:
    st.divider()
    st.markdown("## 🕘 Historial de análisis")

    for item in reversed(st.session_state.historial):
        with st.container():
            st.markdown("### Pregunta")
            st.write(item["pregunta"])
            st.markdown("### Respuesta")
            st.write(item["respuesta"])
            st.divider()
