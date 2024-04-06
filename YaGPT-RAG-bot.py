# создаем простое streamlit приложение для работы с вашими pdf-файлами при помощи YaGPT

import streamlit as st
import tempfile
import os
from opensearchpy import OpenSearch
from yandex_chain import YandexEmbeddings
from yandex_chain import YandexLLM


from langchain.prompts import PromptTemplate
from langchain.vectorstores import OpenSearchVectorSearch

from langchain.chains import RetrievalQA

from langchain_community.chat_models import ChatYandexGPT
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import format_document
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import StrOutputParser

from streamlit_chat import message
from constants import MDB_OS_CA

# from dotenv import load_dotenv

# использовать системные переменные из облака streamlit (secrets)
# yagpt_api_key = st.secrets["yagpt_api_key"]
# yagpt_folder_id = st.secrets["yagpt_folder_id"]
# yagpt_api_id = st.secrets["yagpt_api_id"]
mdb_os_pwd = st.secrets["mdb_os_pwd"]
mdb_os_hosts = st.secrets["mdb_os_hosts"].split(",")
mdb_os_index_name = st.secrets["mdb_os_index_name"]

# MDB_OS_CA = st.secrets["mdb_os_ca"] # 

# это основная функция, которая запускает приложение streamlit
def main():
    # Загрузка логотипа компании
    logo_image = './images/logo.png'  # Путь к изображению логотипа

    # # Отображение логотипа в основной части приложения
    from PIL import Image
    # Загрузка логотипа
    logo = Image.open(logo_image)
    # Изменение размера логотипа
    resized_logo = logo.resize((100, 100))
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    # Указываем название и заголовок Streamlit приложения
    st.title('YaGPT-чат с вашими PDF файлами')
    st.warning('Загружайте свои PDF-файлы и задавайте вопросы по ним. Если вы уже загрузили свои файлы, то ***обязательно*** удалите их из списка загруженных и переходите к чату ниже.')

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001F917\U0001F4ACИИ-помощник')
        st.markdown('''
        ## О программе
        Данный YaGPT-помощник реализует [Retrieval-Augmented Generation (RAG)](https://github.com/yandex-cloud-examples/yc-yandexgpt-qa-bot-for-docs/blob/main/README.md) подход
        и использует следующие компоненты:
        - [Yandex GPT](https://cloud.yandex.ru/services/yandexgpt)
        - [Yandex GPT for Langchain](https://pypi.org/project/yandex-chain/)
        - [YC MDB Opensearch](https://cloud.yandex.ru/docs/managed-opensearch/)
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        ''')
        st.markdown('''
            ## Дополнительные настройки
            Можно выбрать [модель](https://cloud.yandex.ru/ru/docs/yandexgpt/concepts/models), степень креативности и системный промпт
            ''')

    model_list = [
      "YandexGPT Lite",
      "YandexGPT Pro"      
    ]    
    selected_model = 0
    selected_model = st.sidebar.radio("Выберите модель для работы:", model_list, index=selected_model, key="index")     
    
    # Добавляем виджет для выбора опции
    prompt_option = st.sidebar.selectbox(
        'Выберите какой системный промпт использовать',
        ('По умолчанию', 'Задать самостоятельно')
    )
    default_prompt = """
    Представь, что ты полезный ИИ-помощник. Твоя задача отвечать на вопросы по информации из предоставленного ниже контекста.
    Отвечай точно в рамках предоставленного контекста, даже если тебя просят придумать.
    Отвечай вежливо в официальном стиле. 
    Если ответ в контексте отсутствует, отвечай: "Я могу давать ответы только по тематике загруженных документов. Мне не удалось найти в документах ответ на ваш вопрос." даже если думаешь, что знаешь ответ на вопрос. 
    Контекст: {context}
    Вопрос: {question}
    """
     # Если выбрана опция "Задать самостоятельно", показываем поле для ввода промпта
    if prompt_option == 'Задать самостоятельно':
        custom_prompt = st.sidebar.text_input('Введите пользовательский промпт:')
    else:
        custom_prompt = default_prompt
        with st.sidebar:
            st.code(custom_prompt)
    # Если выбрали "задать самостоятельно" и не задали, то берем дефолтный промпт
    if len(custom_prompt)==0: custom_prompt = default_prompt  


    global  yagpt_folder_id, yagpt_api_key, mdb_os_ca, mdb_os_pwd, mdb_os_hosts, mdb_os_index_name    
    yagpt_folder_id = st.sidebar.text_input("YAGPT_FOLDER_ID", type='password')
    yagpt_api_key = st.sidebar.text_input("YAGPT_API_KEY", type='password')
    mdb_os_ca = MDB_OS_CA
    # в этой версии креды для доступа к MDB OS задаются в системных переменных
    # mdb_os_pwd = st.sidebar.text_input("MDB_OpenSearch_PASSWORD", type='password')
    # mdb_os_hosts = st.sidebar.text_input("MDB_OpenSearch_HOSTS через 'запятую' ", type='password').split(",")
    mdb_os_index_name = st.sidebar.text_input("MDB_OpenSearch_INDEX_NAME", type='password', value=mdb_os_index_name)
    mdb_os_index_name = f"lcel-{mdb_os_index_name}"

    # yagpt_temp = st.sidebar.text_input("Температура", type='password', value=0.01)
    rag_k = st.sidebar.text_input("Количество поисковых выдач размером с один блок", type='password', value=5)
    # rag_k = st.sidebar.slider("Количество поисковых выдач размером с один блок", 1, 10, 5)
    yagpt_temp = st.sidebar.slider("Степень креативности (температура)", 0.0, 1.0, 0.01)

    # Параметры chunk_size и chunk_overlap
    global chunk_size, chunk_overlap
    chunk_size = st.sidebar.slider("Выберите размер текстового 'окна' разметки документов в символах", 0, 2000, 1000)
    chunk_overlap = st.sidebar.slider("Выберите размер блока перекрытия в символах", 0, 400, 100)

    # Выводим предупреждение, если пользователь не указал свои учетные данные
    if not yagpt_api_key or not yagpt_folder_id or not mdb_os_pwd or not mdb_os_hosts or not mdb_os_index_name:
        st.warning(
            "Пожалуйста, задайте свои учетные данные (в secrets/.env или в раскрывающейся панели слева) для запуска этого приложения.")

    # Загрузка pdf файлов
    uploaded_files = st.file_uploader(
        "После загрузки файлов в формате pdf начнется их добавление в векторную базу данных MDB Opensearch.", accept_multiple_files=True, type=['pdf'])

    # если файлы загружены, сохраняем их во временную папку и потом заносим в vectorstore
    if uploaded_files:
        # создаем временную папку и сохраняем в ней загруженные файлы
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    # сохраняем файл во временную папку
                    with open(os.path.join(temp_dir, file_name), "wb") as f:
                        f.write(uploaded_file.read())
                # отображение спиннера во время инъекции файлов
                with st.spinner("Добавление ваших файлов в базу ..."):
                    ingest_docs(temp_dir)
                    st.success("Ваш(и) файл(ы) успешно принят(ы)")
                    st.session_state['ready'] = True
        except Exception as e:
            st.error(
                f"При загрузке ваших файлов произошла ошибка: {str(e)}")

    # Логика обработки сообщений от пользователей
    # инициализировать историю чата, если ее пока нет 
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # инициализировать состояние готовности, если его пока нет
    if 'ready' not in st.session_state:
        st.session_state['ready'] = True

    if st.session_state['ready']:

        # подключиться к векторной БД Opensearch, используя учетные данные (проверка подключения)
        # conn = OpenSearch(
        #     mdb_os_hosts,
        #     http_auth=('admin', mdb_os_pwd),
        #     use_ssl=True,
        #     verify_certs=False,
        #     ca_certs=MDB_OS_CA
        #     )
        # print(conn)

        # инициализировать модели YandexEmbeddings и YandexGPT
        embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)

        # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"
        # model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
        if selected_model==0: 
            model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt-lite/latest"
        else:
            model_uri = "gpt://"+str(yagpt_folder_id)+"/yandexgpt/latest"  
        # обращение к модели YaGPT
        llm = ChatYandexGPT(api_key=yagpt_api_key, model_uri=model_uri, temperature = yagpt_temp, max_tokens=8000)
        # model = YandexLLM(api_key = yagpt_api_key, folder_id = yagpt_folder_id, temperature = 0.6, max_tokens=8000, use_lite = False)
        # llm = YandexLLM(api_key=yagpt_api_key, folder_id=yagpt_folder_id, temperature = yagpt_temp, max_tokens=7000)
        # llm = YandexLLM(api_key = yagpt_api_key, folder_id = yagpt_folder_id, temperature = yagpt_temp.6, max_tokens=8000, use_lite = False)

        # инициализация retrival chain - цепочки поиска
        vectorstore = OpenSearchVectorSearch (
            embedding_function=embeddings,
            index_name = mdb_os_index_name,
            opensearch_url=mdb_os_hosts,
            http_auth=("admin", mdb_os_pwd),
            use_ssl = True,
            verify_certs = False,
            ca_certs = MDB_OS_CA,
            engine = 'lucene'
        )  

        _template = """Учитывая историю общения и текущий вопрос, составь из всего этого отдельный общий вопрос на русском языке.

        История общения:
        {chat_history}
        Текущий вопрос: {question}
        Отдельный общий вопрос:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        custom_prompt = """Отвечай на вопрос, основываясь только на следующем контексте:
        {context}
        Вопрос: {question}
        """
        ANSWER_PROMPT = ChatPromptTemplate.from_template(custom_prompt)
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")       

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        memory = ConversationBufferMemory(
            return_messages=True, output_key="answer", input_key="question"
        )
        # Сначала мы добавляем шаг для загрузки памяти
        # Поэтому добавляем ключ "memory" во входящий объект
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history"),
        )

        # Теперь определяем standalone_question (композитный вопрос, который учитывает историю общения)
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser(),
        }

        # Теперь извлекаем нужные документы
        retriever = vectorstore.as_retriever(search_kwargs={'k': rag_k})
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["standalone_question"],
        }
        # Конструируем вводные для финального промпта
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }

        # Теперь запускаем выдачу ответов
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | llm,
            "docs": itemgetter("docs"),
        }
        # И собираем все вместе!
        qa = loaded_memory | standalone_question | retrieved_documents | answer

        # QA_CHAIN_PROMPT = PromptTemplate.from_template(custom_prompt)
        # qa = RetrievalQA.from_chain_type(
        #     llm,
        #     retriever=vectorstore.as_retriever(search_kwargs={'k': rag_k}),
        #     return_source_documents=True,
        #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        # )

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [
                "Что бы вы хотели узнать о документе?"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Привет!"]

        # контейнер для истории чата
        response_container = st.container()

        # контейнер для текстового поля
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input(
                    "Вопрос:", placeholder="О чем этот документ?", key='input')
                submit_button = st.form_submit_button(label='Отправить')

            if submit_button and user_input:
                # отобразить загрузочный "волчок"
                with st.spinner("Думаю..."):
                    print("История чата: ", st.session_state['chat_history'])
                    output = qa(
                        {"query": user_input})
                    print(output)
                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output['result'])

                    # # обновляем историю чата с помощью вопроса пользователя и ответа от бота
                    st.session_state['chat_history'].append(
                        {"вопрос": user_input, "ответ": output['result']})
                    ## добавляем источники к ответу
                    input_documents = output['source_documents']
                    i = 0
                    for doc in input_documents:
                        source = doc.metadata['source']
                        page_content = doc.page_content
                        i = i + 1
                        with st.expander(f"**Источник N{i}:** [{source}]"):
                            st.write(page_content)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(
                        i) + '_user')
                    message(st.session_state["generated"][i], key=str(
                        i))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # st.write(f"Что-то пошло не так. Возможно, не хватает входных данных для работы. {str(e)}")
        st.write(f"Не хватает входных данных для продолжения работы. См. панель слева.")