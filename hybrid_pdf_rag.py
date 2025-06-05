import streamlit as st
import os
import pickle
import hashlib
from io import BytesIO

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import tempfile

# Директории для хранения данных
CHROMA_DB_DIR = "./chroma_db"
BM25_CACHE_DIR = "./bm25_cache"
DOCS_METADATA_FILE = "./processed_docs.json"

# Создаем директории если их нет
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(BM25_CACHE_DIR, exist_ok=True)

def russian_preprocess(text):
    """Функция для предобработки русского текста"""
    import re
    text = re.sub(r'[^а-яё\s]', ' ', text.lower())
    return [word for word in text.split() if len(word) > 2]

template = """
Ты - помощник для ответов на вопросы по документу. Используй весь предоставленный контекст для полного и точного ответа.

ВАЖНО: Отвечай ТОЛЬКО на русском языке. Никаких китайских символов или иероглифов.

Анализируй весь контекст целиком. Если вопрос требует общего обзора документа - дай развернутый ответ.
Если не знаешь ответа, честно скажи об этом.

Вопрос: {question} 
Контекст: {context} 

Дай полный и структурированный ответ на русском языке:
"""

model = OllamaLLM(model="qwen2.5:14b")

def get_file_hash(uploaded_file):
    """Создает уникальный хеш для файла"""
    file_content = uploaded_file.getvalue()
    return hashlib.md5(file_content).hexdigest()

def load_docs_metadata():
    """Загружает метаданные обработанных документов"""
    import json
    try:
        with open(DOCS_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_docs_metadata(metadata):
    """Сохраняет метаданные обработанных документов"""
    import json
    with open(DOCS_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_pdf_from_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_file_path = temp_file.name
    
    try:
        loader = PDFPlumberLoader(temp_file_path)
        documents = loader.load()
        return documents
    finally:
        os.unlink(temp_file_path)

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def build_semantic_retriever(documents, collection_name):
    """Создает или загружает персистентный векторный индекс"""
    embeddings = OllamaEmbeddings(model="qwen2.5:14b")
    
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    # Добавляем документы только если коллекция пустая
    if vector_store._collection.count() == 0:
        vector_store.add_documents(documents)
    
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

def build_bm25_retriever(documents, file_hash):
    """Создает или загружает BM25 индекс"""
    # Создаем новый индекс и сохраняем его компоненты отдельно
    if documents:  # Если есть документы для обработки
        retriever = BM25Retriever.from_documents(
            documents, 
            preprocess_func=russian_preprocess,
            k=10
        )
        
        # Сохраняем внутренние данные ретривера
        bm25_data = {
            'docs': retriever.docs,
            'bm25': retriever.vectorizer,
            'k': getattr(retriever, 'k', 10),
            'preprocess_func_name': 'russian_preprocess'
        }
        
        bm25_file = os.path.join(BM25_CACHE_DIR, f"{file_hash}_bm25.pkl")
        with open(bm25_file, 'wb') as f:
            pickle.dump(bm25_data, f)
        
        return retriever
    else:
        # Загружаем сохраненный индекс
        bm25_file = os.path.join(BM25_CACHE_DIR, f"{file_hash}_bm25.pkl")
        
        if os.path.exists(bm25_file):
            with open(bm25_file, 'rb') as f:
                bm25_data = pickle.load(f)
            
            # Восстанавливаем ретривер из сохраненных данных
            retriever = BM25Retriever(
                vectorizer=bm25_data['bm25'],
                docs=bm25_data['docs'],
                k=bm25_data.get('k', 10),
                preprocess_func=russian_preprocess
            )
            return retriever
        else:
            # Если файл не найден, создаем пустой ретривер
            return BM25Retriever.from_documents([], preprocess_func=russian_preprocess, k=10)

def clean_response(response):
    import re
    cleaned = re.sub(r'[\u4e00-\u9fff]+', '', response)
    return cleaned.strip()

def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    
    system_message = """Ты русскоязычный помощник. Отвечай ТОЛЬКО на русском языке. 
    Запрещено использовать китайские символы или любые иероглифы."""
    
    user_message = f"""
    Контекст: {context}
    
    Вопрос: {question}
    
    Дай краткий ответ на русском языке (максимум 3 предложения):
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("user", user_message)
    ])
    
    chain = prompt | model
    result = chain.invoke({})
    return clean_response(result)

# Инициализируем session state
if 'current_doc_hash' not in st.session_state:
    st.session_state.current_doc_hash = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# Загружаем метаданные обработанных документов
docs_metadata = load_docs_metadata()

# Боковая панель с обработанными документами
st.sidebar.title("📚 Обработанные документы")

# Настройки поиска
st.sidebar.title("⚙️ Настройки поиска")
search_fragments = st.sidebar.slider("Количество фрагментов", 5, 20, 10)
chunk_info = st.sidebar.info(f"Размер фрагмента: 2000 символов\nПерекрытие: 400 символов")

if docs_metadata:
    selected_doc = st.sidebar.selectbox(
        "Выберите документ:",
        options=[""] + list(docs_metadata.keys()),
        format_func=lambda x: "Выберите документ..." if x == "" else f"{docs_metadata[x]['name']} ({docs_metadata[x]['fragments']} фрагментов)"
    )
    
    if selected_doc and selected_doc != st.session_state.current_doc_hash:
        # Загружаем выбранный документ
        st.session_state.current_doc_hash = selected_doc
        
        with st.spinner("Загружаю сохраненный документ..."):
            semantic_retriever = build_semantic_retriever([], selected_doc)
            bm25_retriever = build_bm25_retriever([], selected_doc)
            
            st.session_state.retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
        
        st.success(f"Документ '{docs_metadata[selected_doc]['name']}' загружен!")
else:
    st.sidebar.write("Пока нет обработанных документов")

# Загрузка нового файла
uploaded_file = st.file_uploader(
    "Загрузить новый PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    file_hash = get_file_hash(uploaded_file)
    
    if file_hash in docs_metadata:
        st.info(f"📋 Документ '{uploaded_file.name}' уже обработан!")
        if st.button("Использовать сохраненный документ"):
            st.session_state.current_doc_hash = file_hash
            
            semantic_retriever = build_semantic_retriever([], file_hash)
            bm25_retriever = build_bm25_retriever([], file_hash)
            
            st.session_state.retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            
            st.success("Сохраненный документ загружен!")
    else:
        # Обрабатываем новый документ
        st.info(f"🔄 Обрабатываю новый файл размером {uploaded_file.size / (1024*1024):.1f} МБ...")
        
        with st.spinner("Загружаю и обрабатываю документ..."):
            documents = load_pdf_from_uploaded_file(uploaded_file)
            chunked_documents = split_text(documents)
            
            st.info(f"Создано {len(chunked_documents)} фрагментов")
            
            semantic_retriever = build_semantic_retriever(chunked_documents, file_hash)
            bm25_retriever = build_bm25_retriever(chunked_documents, file_hash)
            
            st.session_state.retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
            
            # Сохраняем метаданные
            docs_metadata[file_hash] = {
                'name': uploaded_file.name,
                'fragments': len(chunked_documents),
                'size_mb': round(uploaded_file.size / (1024*1024), 1)
            }
            save_docs_metadata(docs_metadata)
            
            st.session_state.current_doc_hash = file_hash
            
        st.success("✅ Документ обработан и сохранен!")
        st.rerun()

# Чат доступен только если документ загружен
if st.session_state.retriever and st.session_state.current_doc_hash:
    current_doc_name = docs_metadata.get(st.session_state.current_doc_hash, {}).get('name', 'Неизвестный документ')
    fragments_count = docs_metadata.get(st.session_state.current_doc_hash, {}).get('fragments', 0)
    
    st.write(f"💬 **Чат с документом: {current_doc_name}**")
    st.write(f"📊 **Будет использовано {search_fragments} из {fragments_count} фрагментов (~{search_fragments * 2000} символов контекста)**")
    
    question = st.chat_input("Ваш вопрос...")

    if question:
        with st.chat_message("user"):
            st.write(question)
            
        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ в документе..."):
                # Используем настраиваемое количество фрагментов
                related_documents = st.session_state.retriever.invoke(question)
                # Берем только нужное количество (на случай если retriever вернул больше)
                related_documents = related_documents[:search_fragments]
                
                answer = answer_question(question, related_documents)
                st.write(answer)
                
                # Показываем информацию о поиске
                with st.expander("🔍 Детали поиска"):
                    st.write(f"**Найдено фрагментов:** {len(related_documents)}")
                    st.write(f"**Общий объем контекста:** ~{sum(len(doc.page_content) for doc in related_documents)} символов")
                    
                    for i, doc in enumerate(related_documents[:3], 1):
                        st.write(f"**Фрагмент {i}:** {doc.page_content[:200]}...")


