# %%
# Paso 1: Configuración de los requisitos
# Importa las librerías necesarias
import os
import json
import requests
from pymongo import MongoClient
from urllib.parse import quote_plus
from datetime import datetime
from typing import Dict, List

# Importa módulos de terceros
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm

# Importa las utilidades personalizadas
from utils import track_progress, create_index, check_index_ready

# %%
# Configuración de las variables de entorno para la conexión a MongoDB Atlas
# Si estás usando tu propio cluster, asegúrate de que MONGODB_URI esté configurado en tu entorno.
# Utiliza urllib.parse.quote_plus para codificar el nombre de usuario y la contraseña
# en caso de que contengan caracteres especiales como '@'.
# Ejemplo:
# MONGODB_USER = quote_plus("tu_usuario")
# MONGODB_PASSWORD = quote_plus("tu_contraseña_con_@")
# MONGODB_URI = f"mongodb+srv://{MONGODB_USER}:{MONGODB_PASSWORD}@cluster0.xxxxxxx.mongodb.net/"

MONGODB_URI = os.environ.get("MONGODB_URI")
SERVERLESS_URL = os.environ.get("SERVERLESS_URL")

# Inicializa un cliente de Python para MongoDB
mongodb_client = MongoClient(MONGODB_URI)

# Verifica la conexión al servidor de la base de datos
try:
    mongodb_client.admin.command("ping")
    print("Conexión a MongoDB exitosa.")
except Exception as e:
    print(f"Error al conectar a MongoDB: {e}")

# Rastrea el progreso de la creación del clúster (no cambiar)
track_progress("cluster_creation", "ai_rag_lab")

# ---

## Paso 2: Cargar el conjunto de datos

# Carga el archivo JSON con los datos de documentación de MongoDB
with open("mongodb_docs.json", "r") as data_file:
    json_data = data_file.read()

docs = json.loads(json_data)

# Muestra el número de documentos cargados
print(f"Número de documentos cargados: {len(docs)}")

# Muestra una vista previa del primer documento para entender su estructura
print("Vista previa del primer documento:")
print(docs[0])

# ---

## Paso 3: Dividir los datos en "chunks"

# Lista de separadores comunes para dividir el texto
separators = ["\n\n", "\n", " ", "", "#", "##", "###"]

# Utiliza `RecursiveCharacterTextSplitter` de LangChain para dividir el texto.
# Este método divide el texto en "chunks" más pequeños y manejables.
# `chunk_size` define el tamaño de cada fragmento y `chunk_overlap` crea un solapamiento
# entre ellos para mantener el contexto.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4", separators=separators, chunk_size=200, chunk_overlap=30
)

def get_chunks(doc: Dict, text_field: str) -> List[Dict]:
    """
    Divide un documento en "chunks" más pequeños.

    Args:
        doc (Dict): El documento padre del cual se generarán los chunks.
        text_field (str): El campo de texto a dividir.

    Returns:
        List[Dict]: Una lista de documentos divididos en chunks.
    """
    # Extrae el campo de texto a dividir
    text = doc[text_field]
    # Divide el texto en una lista de chunks
    chunks = text_splitter.split_text(text)

    # Crea una lista de nuevos documentos, uno por cada chunk
    chunked_data = []
    for chunk in chunks:
        temp = doc.copy()
        temp[text_field] = chunk
        chunked_data.append(temp)
    
    return chunked_data

# Itera sobre los documentos y los divide en chunks
split_docs = []
for doc in docs:
    chunks = get_chunks(doc, "body")
    split_docs.extend(chunks)

# Muestra el número total de chunks creados
print(f"Número total de chunks creados: {len(split_docs)}")

# Muestra una vista previa del primer chunk para verificar la estructura
print("Vista previa del primer chunk:")
print(split_docs[0])

# ---

## Paso 4: Generar embeddings

# Carga el modelo `gte-small` para generar embeddings vectoriales
embedding_model = SentenceTransformer("thenlper/gte-small")

def get_embedding(text: str) -> List[float]:
    """
    Genera el embedding para una pieza de texto.

    Args:
        text (str): El texto a incrustar.

    Returns:
        List[float]: El embedding del texto como una lista de flotantes.
    """
    # Codifica el texto utilizando el modelo de embedding
    embedding = embedding_model.encode(text)
    # Convierte el array NumPy a una lista de Python
    return embedding.tolist()

# Genera embeddings para cada chunk de documento
embedded_docs = []
for doc in tqdm(split_docs, desc="Generando embeddings"):
    doc["embedding"] = get_embedding(doc["body"])
    embedded_docs.append(doc)

# Verifica que el número de documentos con embeddings sea el mismo que el de los chunks
print(f"Número de documentos con embeddings: {len(embedded_docs)}")

# ---

## Paso 5: Ingresar los datos en MongoDB

# Define los nombres de la base de datos, colección y el índice
DB_NAME = "mongodb_genai_devday_rag"
COLLECTION_NAME = "knowledge_base"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# Conecta a la colección de MongoDB
collection = mongodb_client[DB_NAME][COLLECTION_NAME]

# Elimina todos los registros existentes en la colección
collection.delete_many({})

# Inserta los documentos con embeddings en la colección
collection.insert_many(embedded_docs)

# Muestra el número de documentos insertados
print(f"Se ingirieron {collection.count_documents({})} documentos en la colección {COLLECTION_NAME}.")

# ---

## Paso 6: Crear un índice de búsqueda vectorial

# Define la configuración del índice de búsqueda vectorial
model = {
    "name": ATLAS_VECTOR_SEARCH_INDEX_NAME,
    "type": "vectorSearch",
    "definition": {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,  # Coincide con el modelo `gte-small`
                "similarity": "cosine",
            }
        ]
    },
}

# Crea el índice de búsqueda vectorial
create_index(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME, model)

# Espera a que el índice esté listo
check_index_ready(collection, ATLAS_VECTOR_SEARCH_INDEX_NAME)

# Rastrea el progreso de la creación del índice (no cambiar)
track_progress("vs_index_creation", "ai_rag_lab")

# ---

## Paso 7: Realizar una búsqueda vectorial

def vector_search(user_query: str) -> List[Dict]:
    """
    Recupera documentos relevantes para una consulta de usuario utilizando la búsqueda vectorial.

    Args:
        user_query (str): La consulta del usuario.

    Returns:
        List[Dict]: Una lista de documentos que coinciden.
    """
    # Genera el embedding para la consulta del usuario
    query_embedding = get_embedding(user_query)

    # Define el pipeline de agregación para la búsqueda vectorial
    pipeline = [
        {
            "$vectorSearch": {
                "index": ATLAS_VECTOR_SEARCH_INDEX_NAME,
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,
                "limit": 5
            }
        },
        {
            "$project": {
                "_id": 0,
                "body": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    # Ejecuta el pipeline y devuelve los resultados
    results = collection.aggregate(pipeline)
    return list(results)

# Realiza búsquedas de ejemplo
print("Resultados de la búsqueda 'What are some best practices for data backups in MongoDB?':")
print(vector_search("What are some best practices for data backups in MongoDB?"))

print("\nResultados de la búsqueda 'How to resolve alerts in MongoDB?':")
print(vector_search("How to resolve alerts in MongoDB?"))

# ---

## Paso 8: Construir la aplicación RAG (Retrieval-Augmented Generation)

def create_prompt(user_query: str) -> str:
    """
    Crea un prompt de chat que incluye la consulta del usuario y el contexto relevante.

    Args:
        user_query (str): La consulta del usuario.

    Returns:
        str: El prompt de chat completo.
    """
    # Recupera los documentos más relevantes usando la búsqueda vectorial
    context = vector_search(user_query)
    # Une los documentos recuperados en una sola cadena
    context = "\n\n".join([doc.get('body') for doc in context])
    
    # Crea el prompt para el LLM
    prompt = f"Answer the question based only on the following context. If the context is empty, say I DON'T KNOW\n\nContext:\n{context}\n\nQuestion:{user_query}"
    return prompt

def generate_answer(user_query: str) -> None:
    """
    Genera una respuesta a la consulta del usuario.

    Args:
        user_query (str): La consulta del usuario.
    """
    # Crea el prompt con el contexto
    prompt = create_prompt(user_query)
    # Formatea el mensaje para la API del LLM
    messages = [{"role": "user", "content": prompt}]
    
    # Envía el mensaje a la función serverless para obtener una respuesta del LLM
    response = requests.post(url=SERVERLESS_URL, json={"task": "completion", "data": messages})
    
    # Imprime la respuesta final
    print(response.json()["text"])

# Consulta la aplicación RAG
print("\nRespuesta de la aplicación RAG:")
generate_answer("What are some best practices for data backups in MongoDB?")

# Demuestra la falta de memoria del modelo
print("\n¿Qué acabo de preguntarte? (sin memoria):")
generate_answer("What did I just ask you?")

# ---

## Paso 9: Agregar memoria a la aplicación RAG

# Conecta a la colección para el historial de chat
history_collection = mongodb_client[DB_NAME]["chat_history"]

# Crea un índice en el campo `session_id` para un acceso rápido
history_collection.create_index("session_id")

def store_chat_message(session_id: str, role: str, content: str) -> None:
    """
    Almacena un mensaje de chat en la colección de historial de MongoDB.
    """
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now(),
    }
    history_collection.insert_one(message)

def retrieve_session_history(session_id: str) -> List:
    """
    Recupera el historial de mensajes de chat para una sesión específica.
    """
    # Busca los mensajes para la sesión y los ordena por marca de tiempo
    cursor = history_collection.find({"session_id": session_id}).sort("timestamp", 1)
    
    if cursor:
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in cursor]
    else:
        messages = []
    
    return messages

def generate_answer_with_memory(session_id: str, user_query: str) -> None:
    """
    Genera una respuesta a la consulta del usuario, tomando en cuenta el historial de chat.
    """
    messages = []

    # Recupera los documentos relevantes para la consulta del usuario
    context = vector_search(user_query)
    context = "\n\n".join([d.get("body", "") for d in context])
    
    # Crea un prompt del sistema que incluye el contexto
    system_message = {
        "role": "user",
        "content": f"Answer the question based only on the following context. If the context is empty, say Paque quiere saber eso jajaja saludos\n\nContext:\n{context}",
    }
    messages.append(system_message)

    # Recupera el historial de la sesión y lo agrega a los mensajes
    message_history = retrieve_session_history(session_id)
    messages.extend(message_history)
    
    # Agrega el mensaje actual del usuario
    user_message = {"role": "user", "content": user_query}
    messages.append(user_message)
    
    # Envía los mensajes al servidor para obtener la respuesta del LLM
    response = requests.post(url=SERVERLESS_URL, json={"task": "completion", "data": messages})
    answer = response.json()["text"]

    # Almacena el mensaje del usuario y la respuesta del asistente en el historial
    store_chat_message(session_id, "user", user_query)
    store_chat_message(session_id, "assistant", answer)
    
    print(answer)

# Prueba la aplicación RAG con memoria
print("\nRespuesta de la aplicación RAG (con memoria):")
generate_answer_with_memory(
    session_id="1",
    user_query="What are some best practices for data backups in MongoDB?",
)

print("\n¿Qué acabo de preguntarte? (con memoria):")
generate_answer_with_memory(
    session_id="1",
    user_query="What did I just ask you?",
)