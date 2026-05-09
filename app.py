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
    background: linear-gradient(135deg, #fff8ef 0%, #f7efe5 50%, #efe1d1 100%);
}

.main-title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    color: #4b260b;
    margin-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #7b4b25;
    margin-bottom: 40px;
}

.card {
    background: #ffffff;
    padding: 24px;
    border-radius: 24px;
    box-shadow: 0px 8px 24px rgba(80, 45, 20, 0.12);
    border: 1px solid #ead8c5;
    margin-bottom: 20px;
}

.result-card {
    background: #ffffff;
    padding: 28px;
    border-radius: 24px;
    box-shadow: 0px 8px 24px rgba(80, 45, 20, 0.14);
    border-left: 8px solid #b8793a;
    color: #3b2414;
    font-size: 17px;
    line-height: 1.8;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background-color: #f4dfc8;
    color: #5a2d0c;
    font-weight: 700;
    margin-bottom: 10px;
}

.section-title {
    color: #4b260b;
    font-size: 30px;
    font-weight: 800;
    margin-top: 20px;
}

div.stButton > button {
    background: linear-gradient(90deg, #8b4a20, #c08345);
    color: white;
    border-radius: 14px;
    height: 52px;
    font-weight: 700;
    border: none;
    width: 100%;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #6f3514, #a96a2f);
    color: white;
}

textarea {
    border-radius: 14px !important;
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
    Analiza esta pintura como un historiador del arte.

    Describe:
    - colores
    - estilo artístico
    - personajes
    - iluminación
    - composición
    - símbolos visuales
    - posibles pinturas famosas relacionadas

    Sé detallado y preciso.
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
Eres ArtVision IA, un experto en pinturas famosas, historia del arte,
autores, movimientos artísticos y análisis visual de obras.

Analiza la imagen subida por el usuario y compárala con la base de conocimiento.

No respondas en un solo párrafo.
No uses saludos.
No digas que eres una IA.

Contexto del PDF:
{contexto}

Descripción visual:
{descripcion_imagen}

Pregunta:
{pregunta}

Responde exactamente con este formato:

🖼️ Obra probable
Nombre de la obra.

👨‍🎨 Autor probable
Nombre del autor.

🏛️ Movimiento artístico
Movimiento artístico.

📖 Explicación
Explica por qué coincide con esa obra.

🎯 Nivel de seguridad
Indica si es Alto, Medio o Bajo.
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
    '<div class="subtitle">Reconoce pinturas famosas mediante visión artificial, Gemini y MongoDB Atlas.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1.1, 0.9], gap="large")

with col1:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### 🖼️ Carga una pintura")

    st.write(
        "Sube una imagen de una obra de arte para identificar su posible autor y estilo artístico."
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

with col2:

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("🧠 Consulta al experto")

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
