import os
import numpy as np    

from langchain.vectorstores.faiss import FAISS
#from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.vectorstores.base import VectorStore
from langchain_openai.embeddings.base import OpenAIEmbeddings

# Ваш API-ключ для OpenAI
OPENAI_API_KEY = "Your_Open_API_Key"

# Путь к папке с вашими PDF-файлами (база знаний)
folder_path = "Path_To_Your_Knowledge_Base"

# Путь к папке, где будет храниться (или уже хранится) FAISS-индекс
vector_store_dir = "Path_To_Your_Vector_Store"

# Инициализируем LLM и Embeddings (в данном случае OpenAI)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o")
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

DOCS_IN_RETRIEVER = 15
RELEVANCE_THRESHOLD_DOCS = 0,7
RELEVANCE_THRESHOLD_PROMPT = 0,6

def save_vector_store(vector_store, vector_store_dir: str):
    """
    :param vector_store: Экземпляр FAISS (VectorStore), который нужно сохранить
    :param vector_store_dir: Путь к папке, где будет сохранён индекс
    """
    vector_store.save_local(vector_store_dir)
    print(f"Vector store сохранён в: {vector_store_dir}")


def load_vector_store(vector_store_dir:str, embeddings):
    # Проверяем наличие файла 'index.faiss'
    index_file = os.path.join(vector_store_dir, "index.faiss")
    if not os.path.exists(index_file):
        print(f"Файл {index_file} не найден. Не удалось загрузить vector store.")
        return None
    try:
        vector_store = FAISS.load_local(vector_store_dir, embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store загружен из: {vector_store_dir}")
        return vector_store
    except Exception as e:
        print(f"Ошибка при загрузке vector store: {e}")
        return None

def load_and_index_documents(folder_path: str, vector_store_dir: str, embeddings) -> bool:
    vector_store = load_vector_store(vector_store_dir, embeddings)
    if vector_store:
        print("Существующий vector store успешно загружен.")
        return True

    documents = []
    if not os.path.isdir(folder_path):
        print(f"Папка {folder_path} не найдена.")
        return False

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(pdf_path)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                print(f"Добавлено {len(pdf_docs)} страниц из {filename}")
            except Exception as e:
                print(f"Ошибка при чтении {filename}: {e}")

    if not documents:
        print("Не найдено подходящих PDF-файлов для индексирования.")
        return False
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    #print(f"Всего получено {len(split_docs)} чанков после разбиения.")
    vector_store = FAISS.from_documents(split_docs, embeddings)
    print("Документы успешно проиндексированы в FAISS.")
    
    save_vector_store(vector_store, vector_store_dir)
    return True

def preprocess_user_prompt(user_prompt: str, chat_history: list, llm) -> str:
    """
    Упрощённая предобработка пользовательского запроса через LLM.
    """
    instructions = (
        "Your task is to refine the user prompt below, preserving its meaning.\n"
        "Steps to follow:\n"
        "1. Identify the main question or request.\n"
        "2. If there are multiple tasks, list them.\n"
        "3. Keep the text concise and clear.\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        "Chat history:\n"
        f"{chat_history}\n"
        "-----\n"
        "Now, provide the improved prompt below:\n"
    )
    response = llm(instructions)
    improved_prompt = response.content.strip()
    return improved_prompt

def retrieve_documents(
   vector_store,
   user_prompt: str,
   k: int = 20,
   metadata_filters: dict = None
):
    """
    Выполняет поиск по векторному хранилищу FAISS (similarity search).
    Возвращает список кортежей (Document, score).
    """
    if not vector_store:
        print("Vector store не загружен. Сначала загрузите индекс.")
        return []


    try:
        if metadata_filters:
            docs_with_scores = vector_store.similarity_search_with_score(
               user_prompt,
               k=DOCS_IN_RETRIEVER,
               filter=metadata_filters)
        else:
            docs_with_scores = vector_store.similarity_search_with_score(user_prompt, k=k)
            return docs_with_scores
    except Exception as e:
        print(f"Ошибка при извлечении документов: {e}")
        return []
    
def compute_embeddings_similarity(embeddings, prompt: str, documents: list):
    
    """
    Синхронная функция вычисления косинусной похожести
    между вектором запроса (prompt) и векторами документов.
    """
    
    if not documents:
        return []

    try:
        # Получаем эмбеддинг для prompt
        prompt_embedding = np.array(embeddings.embed_query(prompt))
        relevance_scores = []
        
        for doc in documents:
            doc_embedding = np.array(embeddings.embed_query(doc.page_content))
            # Проверяем валидность эмбеддингов
            if (prompt_embedding.size == 0 or doc_embedding.size == 0):
                print(f"Warning: invalid embedding for doc: {doc.metadata.get('source', 'Unknown')}")
                similarity = 0.0
            else:
                dot_product = np.dot(prompt_embedding, doc_embedding)
                norm_prompt = np.linalg.norm(prompt_embedding)
                norm_doc = np.linalg.norm(doc_embedding)
                similarity = 0.0
                # Косинусная похожесть = (A·B) / (|A| * |B|)
            if norm_prompt > 1e-9 and norm_doc > 1e-9:
                similarity = dot_product / (norm_prompt * norm_doc)
                # «Зажимаем» результат в [-1, 1] для защиты от плавающих погрешностей
                similarity = np.clip(similarity, -1.0, 1.0)


        relevance_scores.append((doc, similarity))


        return relevance_scores


    except Exception as e:
        print(f"Exception in compute_embeddings_similarity: {str(e)}")
        return [(doc, 0.0) for doc in documents]

def is_prompt_relevant_to_documents(relevance_scores, relevance_threshold=RELEVANCE_THRESHOLD_PROMPT):
    """
    Синхронная проверка: считать ли запрос релевантным документам
    по максимальной оценке (similarity) среди всех.
    """
    try:
        if not relevance_scores:
            return False

        max_similarity = max((sim for _, sim in relevance_scores), default=0.0)
        print(f"Debug: max_similarity = {max_similarity:.4f}, "
              f"threshold = {relevance_threshold}, "
              f"is_relevant = {max_similarity >= relevance_threshold}")

        return max_similarity >= relevance_threshold
    except Exception as e:
        print(f"Exception in is_prompt_relevant_to_documents: {str(e)}")
        return False


def postprocess_llm_response(
    llm_response: str,
    user_prompt: str,
    context_str: str = "",
    references: dict = None,
    is_relevant: bool = False
) -> tuple:
    """
    Упрощённая синхронная постобработка ответа от LLM:
    - улучшает стиль,
    - добавляет/структурирует список ссылок (если нужно).
    Возвращает (final_answer, processed_references).
    """
    if references is None:
        references = {}

    if not is_relevant:
        references = {}
        context_str = ""

    prompt_references = (
        "You are an advanced language model tasked with providing a final, "
        "well-structured answer based on the given content.\n\n"
        "### Provided Data\n"
        f"LLM raw response:\n{llm_response}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"Context:\n{context_str}\n\n"
        f"References:\n{references}\n\n"
        f"is_relevant: {is_relevant}\n"
        "-------------------------\n"
        "Please re-check clarity and, if references exist, list them at the end.\n"
        "Return the final improved answer now:\n"
    )

    # Вызов llm синхронно (упрощённый вариант)
    chain_response = llm(prompt_references)
    final_answer = chain_response.content.strip()

    return final_answer, references


async def generate_response(
   prompt: str,
   chat_history=None,
   metadata_filters=None,
   context=None
):
    # 1. Загрузка/создание vector_store
    load_and_index_documents(KNOWLEDGE_BASE_FOLDER, VECTOR_STORE_DIR, embeddings)
    vector_store = load_vector_store(VECTOR_STORE_DIR, embeddings)
    if not vector_store:
        return "Unable to load Vector Store.", None, None

   # 2. Предобрабатываем вопрос
   if chat_history is None:
       chat_history = []
   prepared_prompt = preprocess_user_prompt(prompt, chat_history, llm)

   # 3. Извлекаем документы из FAISS
   retrieved_docs_with_scores = retrieve_documents(
       vector_store=vector_store,
       user_prompt=prepared_prompt,
       k=5,
       metadata_filters=metadata_filters
   )
   retrieved_docs = [doc for doc, _ in retrieved_docs_with_scores]

   # 4. Подсчитываем косинусную похожесть
   relevance_scores = compute_embeddings_similarity(embeddings, prepared_prompt, retrieved_docs)

   # 5. Фильтруем документы на основе RELEVANCE_THRESHOLD_DOCS
   relevant_docs = [
       doc for (doc, similarity) in relevance_scores
       if similarity >= RELEVANCE_THRESHOLD_DOCS
   ]

   # 6. Если ничего не нашлось, выдаём «fallback»-ответ
   if not relevant_docs:
       fallback_answer = "I couldn't find relevant information to answer your question."
       final_answer, _, _ = postprocess_llm_response(
           llm_response=fallback_answer,
           user_prompt=prompt,
           context_str="",
           references=None,
           is_relevant=False
       )
       return parser_html(final_answer), None, None

   # 7. Формируем «контекст» из релевантных документов
   context_str = ""
   for doc in relevant_docs:
       source = doc.metadata.get('source', 'Unknown')
       page = doc.metadata.get('page', 'N/A')
       content = doc.page_content or 'N/A'
       context_str += f"Source: {source}, Page: {page}\nContent:\n{content}\n---\n"

   # 8. «Системный» промпт: даём модели контекст
   system_prompt = (
       "You are an expert. Provide a concise answer based on the context:\n"
       f"{context_str}\n"
       "--- End Context ---\n"
       "If the user question isn't fully answered in the provided context, "
       "use your best judgment while staying truthful.\n"
   )
   prompt_template = ChatPromptTemplate.from_messages([
       ("system", system_prompt),
       MessagesPlaceholder(variable_name="chat_history"),
       ("user", "{input}")
   ])

   # 9. Формируем финальный промпт для LLM
   chain_input = {
       "input": prepared_prompt,
       "chat_history": chat_history
   }
   chain_full_prompt = prompt_template.format(**chain_input)

   # 10. Вызываем LLM
   llm_result = llm(chain_full_prompt)
   if hasattr(llm_result, "content"):
       answer_text = llm_result.content
   else:
       answer_text = str(llm_result)

   # 11. Оцениваем «глобальную» релевантность (RELEVANCE_THRESHOLD_PROMPT)
   is_relevant = is_prompt_relevant_to_documents(relevance_scores)

   # 12. Готовим список ссылок
   references = {}
   for doc in relevant_docs:
       filename = doc.metadata.get("source", "Unknown")
       page = doc.metadata.get("page", "N/A")
       references.setdefault(filename, set()).add(page)

   # 13. Пост-обработка ответа
   final_answer, processed_refs = postprocess_llm_response(
       llm_response=answer_text,
       user_prompt=prompt,
       context_str=context_str,
       references=references,
       is_relevant=is_relevant
   )

   # 14. Итоговый форматированный текст
   if is_relevant:
       final_text = final_answer + "\n---\nAdditional references may be listed above."
       source_files = list(processed_refs.keys()) if processed_refs else None
   else:
       final_text = final_answer
       source_files = None

   return parser_html(final_text), source_files

# ========================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ (запуск через asyncio)
# ========================================================================
if __name__ == "__main__":
   async def main():
       user_query = "Какой защитный слой должен быть у фундаментов?"
       chat_hist = ["User: Привет!"]


       answer, sources, short_sum = await generate_response(
           prompt=user_query,
           chat_history=chat_hist
       )
       print("ОТВЕТ ОТ LLM:\n", answer)
       print("ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ:\n", sources)

   asyncio.run(main())