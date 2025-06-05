import streamlit as st
import os
import pickle
import hashlib
import time
from datetime import datetime
from io import BytesIO

# Настройки для постоянного удержания модели в GPU
os.environ['OLLAMA_KEEP_ALIVE'] = '-1'  # Никогда не выгружать
os.environ['OLLAMA_NUM_PARALLEL'] = '1'  # Одна модель параллельно
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Максимум 1 модель в памяти

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import tempfile
import json

# Конфигурация для библиотеки статей
ARTICLES_DB_DIR = "./articles_chroma_db"
ARTICLES_CACHE_DIR = "./articles_bm25_cache"
ARTICLES_METADATA_FILE = "./articles_metadata.json"

# Создаем директории
os.makedirs(ARTICLES_DB_DIR, exist_ok=True)
os.makedirs(ARTICLES_CACHE_DIR, exist_ok=True)

def russian_preprocess(text):
    """Функция для предобработки русского текста"""
    import re
    text = re.sub(r'[^а-яё\s]', ' ', text.lower())
    return [word for word in text.split() if len(word) > 2]

# Два разных промпта для разных источников
articles_template = """
Ты - эксперт по категорийному менеджменту. 

ВАЖНО: 
- Отвечай ТОЛЬКО на русском языке. Никаких иероглифов.
- Используй эмодзи для заголовков
- Создавай структурированные списки с подпунктами
- Делай красивое форматирование как в примере

Контекст из статей: {context}

Вопрос: {question}

Ответ в таком формате:

🧩 **Основные моменты [тема вопроса]**

1. **Первый ключевой аспект**
   - Детальное объяснение
   - Дополнительные нюансы
   - Конкретные примеры из источников

2. **Второй важный аспект**  
   - Практическое применение
   - Влияние на бизнес-процессы
   - Связь с другими элементами

3. **Третий критичный момент**
   - Технические детали
   - Результаты и эффекты
   - Рекомендации

✅ **Ключевые выводы**

> Основной вывод на основе анализа источников.

- **Практическое значение:** что это дает бизнесу
- **Критические риски:** что происходит без этого  
- **Рекомендации:** как применить на практике

📚 **Источники**

*Информация взята из статей библиотеки по категорийному менеджменту*
"""

llm_template = """
Ты - эксперт по категорийному менеджменту. В базе статей не найдено релевантной информации.

ВАЖНО: 
- Отвечай ТОЛЬКО на русском языке. Никаких иероглифов.
- Используй эмодзи и структуру как в примере
- Будь краток но информативен

Вопрос: {question}

🧠 **Общие знания по теме**

**🎯 Основное:**
- Ключевая суть вопроса
- Практическое применение в ритейле

**💼 Применение в категорийном менеджменте:**
- Как используется на практике
- Связь с другими процессами

> ⚠️ *Ответ основан на общих знаниях, не из конкретных статей библиотеки*
"""

def get_available_models():
    """Получает список доступных моделей Ollama"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
    except:
        pass
    return ["qwen2.5:14b", "deepseek-r1:14b", "qwen3:8b", "llama3:8b", "gemma3:4b"]

def unload_model(model_name):
    """Выгружает модель из памяти GPU"""
    try:
        import requests
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "keep_alive": 0}
        )
        return response.status_code == 200
    except:
        return False

def get_loaded_models():
    """Получает список загруженных в память моделей"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/ps")
        if response.status_code == 200:
            models = response.json().get("models", [])
            result = []
            for model in models:
                name = model["name"]
                size_bytes = model.get("size", 0)
                # Конвертируем байты в GB
                size_gb = size_bytes / (1024**3)
                result.append((name, size_gb))
            return result
    except:
        pass
    return []

def create_model(model_name):
    """Создает модель с выбранным именем и выгружает предыдущую"""
    # Выгружаем предыдущую модель если она была
    if 'current_model' in st.session_state and st.session_state.current_model != model_name:
        old_model = st.session_state.current_model
        if unload_model(old_model):
            st.sidebar.info(f"🗑️ Выгружена модель {old_model}")
    
    return OllamaLLM(
        model=model_name,
        keep_alive=-1,  # Держим только текущую модель
        temperature=0.1
    )

model = OllamaLLM(
    model="qwen2.5:14b",
    keep_alive=-1,  # Никогда не выгружать из памяти
    temperature=0.1  # Более детерминированные ответы
)

def extract_title(text):
    """Пытается извлечь заголовок статьи из первых строк"""
    lines = text.split('\n')[:10]
    for line in lines:
        cleaned = line.strip()
        if 20 <= len(cleaned) <= 150:  # Разумная длина заголовка
            return cleaned
    return "Без названия"

def load_articles_metadata():
    """Загружает метаданные библиотеки статей"""
    try:
        with open(ARTICLES_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "articles": [],
            "total_fragments": 0,
            "total_size_mb": 0,
            "last_updated": None,
            "collection_initialized": False
        }

def save_articles_metadata(metadata):
    """Сохраняет метаданные библиотеки статей"""
    metadata["last_updated"] = datetime.now().isoformat()
    with open(ARTICLES_METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def process_articles_batch(uploaded_files):
    """Обрабатывает пакет статей"""
    all_documents = []
    articles_info = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Обрабатываю: {uploaded_file.name}")
        
        # Загружаем PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            loader = PDFPlumberLoader(temp_file_path)
            documents = loader.load()
            
            if not documents:
                st.warning(f"Не удалось обработать {uploaded_file.name}")
                continue
            
            # Разбиваем на фрагменты (оптимизировано для статей)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Больше для статей
                chunk_overlap=300,
                add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            
            # Извлекаем заголовок статьи
            full_text = "\n".join([doc.page_content for doc in documents])
            article_title = extract_title(full_text)
            
            # Добавляем метаданные к каждому фрагменту
            for chunk in chunks:
                chunk.metadata.update({
                    "source_file": uploaded_file.name,
                    "article_title": article_title,
                    "upload_date": datetime.now().isoformat(),
                    "file_size_mb": round(uploaded_file.size / (1024*1024), 2)
                })
            
            all_documents.extend(chunks)
            articles_info.append({
                "filename": uploaded_file.name,
                "title": article_title,
                "fragments": len(chunks),
                "size_mb": round(uploaded_file.size / (1024*1024), 2)
            })
            
        except Exception as e:
            st.error(f"Ошибка при обработке {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(temp_file_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Обработка завершена!")
    return all_documents, articles_info

def get_embeddings(model_name):
    """Создает embedding функцию только при необходимости"""
    return OllamaEmbeddings(model=model_name)

def initialize_articles_retriever():
    """Инициализирует ретривер для существующих статей БЕЗ загрузки модели"""
    # BM25 ретривер (не требует embeddings)
    bm25_file = os.path.join(ARTICLES_CACHE_DIR, "articles_bm25.pkl")
    
    if os.path.exists(bm25_file):
        with open(bm25_file, 'rb') as f:
            bm25_data = pickle.load(f)
        bm25_retriever = BM25Retriever(
            vectorizer=bm25_data['bm25'],
            docs=bm25_data['docs'],
            k=8,
            preprocess_func=russian_preprocess
        )
        
        # Возвращаем только BM25 - без векторного поиска
        return bm25_retriever
    else:
        return None

def create_hybrid_retriever(embedding_model_name):
    """Создает полный гибридный ретривер с embeddings при первом поиске"""
    # Создаем embeddings только сейчас
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    
    # Векторное хранилище
    vector_store = Chroma(
        collection_name="category_management_articles",
        embedding_function=embeddings,
        persist_directory=ARTICLES_DB_DIR
    )
    
    semantic_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 16, "lambda_mult": 0.8}
    )
    
    # BM25 поиск
    bm25_file = os.path.join(ARTICLES_CACHE_DIR, "articles_bm25.pkl")
    if os.path.exists(bm25_file):
        with open(bm25_file, 'rb') as f:
            bm25_data = pickle.load(f)
        bm25_retriever = BM25Retriever(
            vectorizer=bm25_data['bm25'],
            docs=bm25_data['docs'],
            k=8,
            preprocess_func=russian_preprocess
        )
        
        # Гибридный ретривер
        return EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[0.6, 0.4]
        )
    else:
        return semantic_retriever

def add_articles_to_retriever(documents, embedding_model):
    """Добавляет новые статьи - ЗДЕСЬ создаем эмбеддинги"""
    embeddings = get_embeddings(embedding_model)  # ← Только при добавлении!
    
    vector_store = Chroma(
        collection_name="category_management_articles",
        embedding_function=embeddings,
        persist_directory=ARTICLES_DB_DIR
    )
    vector_store.add_documents(documents)
    
    # Обновляем BM25 индекс
    bm25_file = os.path.join(ARTICLES_CACHE_DIR, "articles_bm25.pkl")
    
    # Загружаем существующие документы
    existing_docs = []
    if os.path.exists(bm25_file):
        with open(bm25_file, 'rb') as f:
            existing_data = pickle.load(f)
            existing_docs = existing_data.get('docs', [])
    
    # Объединяем с новыми
    all_docs = existing_docs + documents
    bm25_retriever = BM25Retriever.from_documents(
        all_docs, 
        preprocess_func=russian_preprocess,
        k=8
    )
    
    # Сохраняем обновленный индекс
    bm25_data = {
        'docs': all_docs,
        'bm25': bm25_retriever.vectorizer,
        'k': 8
    }
    with open(bm25_file, 'wb') as f:
        pickle.dump(bm25_data, f)
    
    return initialize_articles_retriever()

def has_relevant_content(documents, threshold=0.5):
    """Проверяет есть ли релевантный контент в найденных документах"""
    if not documents or len(documents) < 2:
        return False
    
    # Проверяем общую длину контента
    total_length = sum(len(doc.page_content) for doc in documents[:5])
    return total_length > 2000  # Достаточно контента для ответа

def answer_from_articles(question, documents):
    """Генерирует ответ на основе статей"""
    start_time = time.time()
    
    # Группируем по статьям для лучшего контекста
    articles_content = {}
    for doc in documents[:6]:  # Уменьшено с 12 до 6 фрагментов
        article_title = doc.metadata.get('article_title', 'Неизвестна')
        source_file = doc.metadata.get('source_file', '')
        
        key = f"{article_title} ({source_file})"
        if key not in articles_content:
            articles_content[key] = []
        articles_content[key].append(doc.page_content)
    
    # Формируем контекст
    context_parts = []
    for article_key, contents in articles_content.items():
        combined_content = "\n".join(contents[:2])  # Максимум 2 фрагмента на статью
        context_parts.append(f"Из статьи '{article_key}':\n{combined_content}")
    
    context = "\n\n".join(context_parts)
    
    # Измеряем время генерации
    generation_start = time.time()
    prompt = ChatPromptTemplate.from_template(articles_template)
    chain = prompt | model
    result = chain.invoke({"question": question, "context": context})
    generation_time = time.time() - generation_start
    
    total_time = time.time() - start_time
    
    return {
        "answer": result,
        "generation_time": generation_time,
        "total_time": total_time,
        "fragments_used": len(documents[:6]),
        "sources_count": len(articles_content)
    }

def answer_from_llm(question):
    """Генерирует ответ на основе знаний LLM"""
    start_time = time.time()
    
    prompt = ChatPromptTemplate.from_template(llm_template)
    chain = prompt | model
    result = chain.invoke({"question": question})
    
    total_time = time.time() - start_time
    
    return {
        "answer": result,
        "generation_time": total_time,
        "total_time": total_time
    }

def clean_response(response):
    """Очищает ответ от лишних символов"""
    import re
    
    # Убираем азиатские символы
    cleaned = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+', '', response)
    
    # Убираем лишние символы ###
    cleaned = re.sub(r'#{3,}', '', cleaned)
    
    # Убираем ТОЛЬКО множественные пробелы внутри строк, сохраняя переносы
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Заменяем только пробелы и табы
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Убираем избыточные пустые строки
    
    # Убираем висящие знаки препинания
    cleaned = re.sub(r'[,，。：:]+\s*$', '', cleaned)
    cleaned = re.sub(r'^\s*[,，。：:]+', '', cleaned)
    
    return cleaned.strip()

def is_simple_list_question(question):
    """Проверяет простые вопросы про список документов"""
    simple_keywords = ["какие документы", "список статей", "что есть в базе", "какие файлы", "документы есть"]
    return any(keyword in question.lower() for keyword in simple_keywords)

def get_articles_list():
    """Возвращает список статей из метаданных"""
    metadata = load_articles_metadata()
    if not metadata["articles"]:
        return "В библиотеке пока нет статей."
    
    result = f"📚 В библиотеке {len(metadata['articles'])} статей:\n\n"
    for i, article in enumerate(metadata["articles"], 1):
        result += f"{i}. **{article['title']}**\n   📄 {article['filename']} ({article['fragments']} фрагментов)\n\n"
    
    return result

def display_answer_beautifully(answer_text, result_info, search_time=None):
    """Красиво отображает ответ с форматированием"""
    
    # Основной ответ
    st.markdown(answer_text)
    
    # Разделитель  
    st.divider()
    
    # Информационные метрики в колонках
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if search_time:
            st.metric("🔍 Поиск", f"{search_time:.2f}с")
        else:
            st.metric("🧠 Режим", "LLM")
            
    with col2:
        st.metric("⏱️ Генерация", f"{result_info['generation_time']:.2f}с")
        
    with col3:
        if 'fragments_used' in result_info:
            st.metric("📊 Фрагментов", result_info['fragments_used'])
        else:
            st.metric("📚 Источник", "Общие знания")
            
    with col4:
        if 'sources_count' in result_info:
            st.metric("📑 Статей", result_info['sources_count'])
        else:
            st.metric("🎯 Качество", "Высокое")

def display_sources_beautifully(docs, max_sources):
    """Красиво отображает источники информации"""
    with st.expander("📋 **Источники информации**", expanded=False):
        sources = {}
        for doc in docs[:max_sources]:
            article = doc.metadata.get('article_title', 'Неизвестна')
            file = doc.metadata.get('source_file', '')
            if article not in sources:
                sources[article] = {
                    'file': file,
                    'fragments': 0
                }
            sources[article]['fragments'] += 1
        
        st.markdown("#### 📚 Использованные статьи:")
        for i, (article, info) in enumerate(sources.items(), 1):
            st.markdown(f"""
            **{i}. {article}**  
            📄 *{info['file']}*  
            🔢 *{info['fragments']} фрагментов использовано*
            """)
            if i < len(sources):
                st.divider()
        
        st.info(f"📊 **Итого:** {len(docs)} фрагментов из {len(sources)} статей")

# Настройка страницы
st.set_page_config(
    page_title="Библиотека статей по категорийному менеджменту", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Библиотека статей по категорийному менеджменту")
st.markdown("*Интеллектуальный поиск по коллекции статей с fallback на общие знания*")

# Инициализация session state
if 'articles_retriever' not in st.session_state:
    st.session_state.articles_retriever = None
if 'library_loaded' not in st.session_state:
    st.session_state.library_loaded = False

# Загрузка метаданных
articles_metadata = load_articles_metadata()

# Автоматическая загрузка существующей библиотеки
if not st.session_state.library_loaded and articles_metadata["articles"]:
    with st.spinner("Загружаю существующую библиотеку статей..."):
        try:
            st.session_state.articles_retriever = initialize_articles_retriever()
            st.session_state.library_loaded = True
            st.success(f"✅ Загружена библиотека: {len(articles_metadata['articles'])} статей")
        except Exception as e:
            st.error(f"Ошибка загрузки библиотеки: {str(e)}")

# Боковая панель - Статистика
st.sidebar.title("📊 Статистика библиотеки")

if articles_metadata["articles"]:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Статей", len(articles_metadata['articles']))
        st.metric("Фрагментов", articles_metadata['total_fragments'])
    with col2:
        st.metric("Размер", f"{articles_metadata['total_size_mb']:.1f} МБ")
        last_update = articles_metadata.get('last_updated', 'Неизвестно')
        if last_update != 'Неизвестно':
            try:
                update_date = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                st.metric("Обновлено", update_date.strftime('%d.%m.%Y'))
            except:
                st.metric("Обновлено", "Недавно")
    
    with st.sidebar.expander("📑 Список статей"):
        for i, article in enumerate(articles_metadata["articles"], 1):
            st.write(f"**{i}. {article['title']}**")
            st.caption(f"📄 {article['filename']}")
            st.caption(f"💾 {article['size_mb']} МБ, {article['fragments']} фрагментов")
            st.divider()
else:
    st.sidebar.info("Библиотека пуста. Загрузите статьи ниже.")

# Боковая панель - Настройки
st.sidebar.title("⚙️ Настройки поиска")

search_strategy = st.sidebar.selectbox(
    "Стратегия ответа:",
    ["🔄 Авто (статьи → LLM)", "📚 Только статьи", "🧠 Только LLM"],
    help="Авто: сначала ищет в статьях, если не найдено - использует общие знания"
)

relevance_threshold = st.sidebar.slider(
    "Порог релевантности", 
    0.1, 1.0, 0.5, 0.1,
    help="Насколько релевантными должны быть найденные фрагменты"
)

max_articles_sources = st.sidebar.slider(
    "Макс. источников в ответе", 
    1, 10, 5,
    help="Максимальное количество статей для использования в ответе"
)

# В боковой панели после настроек поиска добавляем:
st.sidebar.title("🤖 Настройки модели")

# Показываем загруженные модели
loaded_models = get_loaded_models()
if loaded_models:
    st.sidebar.write("**💾 В памяти GPU:**")
    total_size = 0
    for name, size_gb in loaded_models:
        total_size += size_gb
        st.sidebar.write(f"• {name}: {size_gb:.1f} GB")
    st.sidebar.write(f"**Всего: {total_size:.1f} GB**")
    
    # Кнопка очистки всех моделей
    if st.sidebar.button("🗑️ Очистить память GPU"):
        for name, _ in loaded_models:
            unload_model(name)
        st.sidebar.success("✅ Память GPU очищена")
        st.rerun()

# Информация о моделях
model_info = {
    "qwen2.5:14b": "🧠 Сбалансированная (9GB) - хорошо для сложных задач",
    "deepseek-r1:14b": "🎯 Лучшее качество (9GB) - отличное рассуждение", 
    "qwen3:8b": "⚡ Быстрая (5GB) - хороший баланс скорости и качества",
    "llama3:8b": "🌟 Стабильная (5GB) - надежный русский язык",
    "gemma3:4b": "🚀 Сверхбыстрая (3GB) - экономия ресурсов"
}

# Выбор модели
available_models = get_available_models()
selected_model = st.sidebar.selectbox(
    "Выберите модель:",
    available_models,
    index=0 if "qwen2.5:14b" in available_models else 0,
    format_func=lambda x: f"{x} - {model_info.get(x, 'Модель Ollama')}"
)

# Создаем/меняем модель только при изменении
if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
    with st.spinner(f"🔄 Переключаю на модель {selected_model}..."):
        model = create_model(selected_model)
        st.session_state.current_model = selected_model
    st.sidebar.success(f"✅ Активна модель {selected_model}")
else:
    model = create_model(selected_model)

# Основной интерфейс - Загрузка файлов
st.header("📥 Загрузка статей")

uploaded_files = st.file_uploader(
    "Выберите PDF файлы статей",
    type="pdf",
    accept_multiple_files=True,
    help=f"Загрузите до 20 PDF файлов одновременно. Текущий размер библиотеки: {articles_metadata.get('total_size_mb', 0):.1f} МБ"
)

if uploaded_files:
    st.info(f"Выбрано файлов: {len(uploaded_files)} (общий размер: {sum(f.size for f in uploaded_files) / (1024*1024):.1f} МБ)")
    
    if len(uploaded_files) > 20:
        st.warning("Рекомендуется загружать не более 20 файлов за раз для стабильной работы.")
    
    if st.button("🔄 Обработать и добавить в библиотеку", type="primary"):
        with st.spinner("Обрабатываю статьи..."):
            try:
                documents, articles_info = process_articles_batch(uploaded_files)
                
                if documents:
                    st.success(f"✅ Обработано {len(uploaded_files)} статей, создано {len(documents)} фрагментов")
                    
                    # Обновляем ретривер
                    if st.session_state.articles_retriever is None:
                        st.session_state.articles_retriever = add_articles_to_retriever(documents, selected_model)
                    else:
                        st.session_state.articles_retriever = add_articles_to_retriever(documents, selected_model)
                    
                    st.session_state.library_loaded = True
                    
                    # Обновляем метаданные
                    articles_metadata["articles"].extend(articles_info)
                    articles_metadata["total_fragments"] += len(documents)
                    articles_metadata["total_size_mb"] += sum(info["size_mb"] for info in articles_info)
                    articles_metadata["collection_initialized"] = True
                    save_articles_metadata(articles_metadata)
                    
                    # Показываем детали
                    with st.expander("📋 Детали обработки"):
                        for info in articles_info:
                            st.write(f"• **{info['title']}** - {info['fragments']} фрагментов ({info['size_mb']} МБ)")
                    
                    st.rerun()
                else:
                    st.error("Не удалось обработать ни одного файла")
                    
            except Exception as e:
                st.error(f"Ошибка при обработке: {str(e)}")

# Основной интерфейс - Чат
st.header("💬 Чат с библиотекой")

if st.session_state.articles_retriever or search_strategy == "🧠 Только LLM":
    # Показываем статус в красивом формате
    if search_strategy == "🧠 Только LLM":
        st.info("🧠 **Режим работы:** Ответы на основе общих знаний модели")
    elif articles_metadata["articles"]:
        st.success(f"📚 **Готов к поиску!** {len(articles_metadata['articles'])} статей • {articles_metadata['total_fragments']} фрагментов • {articles_metadata['total_size_mb']:.1f} МБ данных")
    else:
        st.warning("💭 **Библиотека пуста** — будут использоваться общие знания")
    
    question = st.chat_input("Задайте вопрос по категорийному менеджменту...")
    
    if question:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            # Проверяем простые вопросы сначала
            if is_simple_list_question(question):
                articles_list = get_articles_list()
                st.write(articles_list)
                st.caption("📋 Список статей из библиотеки")
                
            elif search_strategy == "🧠 Только LLM":
                # Только LLM
                with st.spinner("💭 Отвечаю на основе общих знаний..."):
                    result = answer_from_llm(question)
                    answer = clean_response(result["answer"])
                    display_answer_beautifully(answer, result)
                    
            elif search_strategy == "📚 Только статьи":
                # Только статьи
                if st.session_state.articles_retriever:
                    with st.spinner("🔍 Ищу в библиотеке статей..."):
                        search_start = time.time()
                        
                        # Проверяем нужно ли создать полный ретривер
                        if isinstance(st.session_state.articles_retriever, BM25Retriever):
                            # Первый поиск - создаем полный ретривер с embeddings
                            with st.spinner("🔄 Инициализирую векторный поиск..."):
                                st.session_state.articles_retriever = create_hybrid_retriever(selected_model)
                        
                        docs = st.session_state.articles_retriever.invoke(question)
                        search_time = time.time() - search_start
                        
                        if docs and has_relevant_content(docs, relevance_threshold):
                            result = answer_from_articles(question, docs)
                            answer = clean_response(result["answer"])
                            display_answer_beautifully(answer, result, search_time)
                            
                            # Показываем источники
                            display_sources_beautifully(docs, max_articles_sources)
                        else:
                            st.warning("📭 В библиотеке статей не найдено релевантной информации по этому вопросу.")
                            st.caption(f"🔍 Поиск завершен за {search_time:.2f}с")
                else:
                    st.error("❌ **Библиотека статей не загружена**  \nЗагрузите статьи сначала.")
                    
            else:  # Авто режим
                if st.session_state.articles_retriever:
                    with st.spinner("🔍 Ищу в библиотеке статей..."):
                        search_start = time.time()
                        
                        # Проверяем нужно ли создать полный ретривер
                        if isinstance(st.session_state.articles_retriever, BM25Retriever):
                            # Первый поиск - создаем полный ретривер с embeddings
                            with st.spinner("🔄 Инициализирую векторный поиск..."):
                                st.session_state.articles_retriever = create_hybrid_retriever(selected_model)
                        
                        docs = st.session_state.articles_retriever.invoke(question)
                        search_time = time.time() - search_start
                        
                        if has_relevant_content(docs, relevance_threshold):
                            # Найдено в статьях
                            result = answer_from_articles(question, docs)
                            answer = clean_response(result["answer"])
                            display_answer_beautifully(answer, result, search_time)
                            
                            # Показываем источники
                            display_sources_beautifully(docs, max_articles_sources)
                        else:
                            # Fallback на LLM
                            st.info("📭 **В статьях не найдено релевантной информации**  \n🔄 Переключаюсь на общие знания")
                            st.caption(f"🔍 Поиск завершен за {search_time:.2f}с")
                            
                            with st.spinner("💭 Генерирую ответ на основе общих знаний..."):
                                result = answer_from_llm(question)
                                answer = clean_response(result["answer"])
                                display_answer_beautifully(answer, result)
                else:
                    # Нет ретривера - используем LLM
                    st.warning("📚 **Библиотека не загружена**  \n🧠 Использую общие знания")
                    with st.spinner("💭 Отвечаю на основе общих знаний..."):
                        result = answer_from_llm(question)
                        answer = clean_response(result["answer"])
                        display_answer_beautifully(answer, result)
else:
    st.info("👆 Загрузите статьи в библиотеку чтобы начать работу, или выберите режим 'Только LLM'")

# Футер
st.markdown("---")
st.markdown("*Система интеллектуального поиска по статьям категорийного менеджмента*") 