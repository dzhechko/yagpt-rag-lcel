import tempfile
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import OpenSearchVectorSearch
from yandex_chain import YandexEmbeddings
import streamlit as st

def ingest_docs(temp_dir: str = tempfile.gettempdir()):
    """
    Инъекция ваших pdf файлов в MBD Opensearch
    """
    global chunk_size, chunk_overlap
    global  yagpt_folder_id, yagpt_api_key, mdb_os_ca, mdb_os_pwd, mdb_os_hosts, mdb_os_index_name
    try:
        # выдать ошибку, если каких-то переменных не хватает
        if not yagpt_api_key or not yagpt_folder_id or not mdb_os_pwd or not mdb_os_hosts or not mdb_os_index_name:
            raise ValueError(
                "Пожалуйста укажите необходимый набор переменных окружения")

        # загрузить PDF файлы из временной директории
        loader = DirectoryLoader(
            temp_dir, glob="**/*.pdf", loader_cls=PyPDFLoader, recursive=True
        )
        documents = loader.load()

        # разбиваем документы на блоки
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        documents = text_splitter.split_documents(documents)
        print(len(documents))

        # подключаемся к базе данных MDB Opensearch, используя наши ключи (проверка подключения)
        # conn = OpenSearch(
        #     mdb_os_hosts,
        #     http_auth=('admin', mdb_os_pwd),
        #     use_ssl=True,
        #     verify_certs=False,
        #     ca_certs=mdb_os_ca)
        # для включения проверки MDB сертификата используйте verify_certs=True, также надо будет загрузить сертификат используя инструкцию по ссылке 
        # https://cloud.yandex.ru/docs/managed-opensearch/operations/connect 
        # и положить его в папку .opensearch/root.crt
        
        # инициируем процедуру превращения блоков текста в Embeddings через YaGPT Embeddings API, используя API ключ доступа
        embeddings = YandexEmbeddings(folder_id=yagpt_folder_id, api_key=yagpt_api_key)
        embeddings.sleep_interval = 0.1 #текущее ограничение эмбеддера 10 RPS, делаем задержку 1/10 секунды, чтобы не выйти за это ограничение
        text_to_print = f"Ориентировочное время оцифровки = {len(documents)*embeddings.sleep_interval} с."
        st.text(text_to_print)

        # добавляем "документы" (embeddings) в векторную базу данных Opensearch
        OpenSearchVectorSearch.from_documents(
            documents,
            embeddings,
            opensearch_url=mdb_os_hosts,
            http_auth=("admin", mdb_os_pwd),
            use_ssl = True,
            verify_certs = False,
            ca_certs = mdb_os_ca,
            engine = 'lucene',
            index_name = mdb_os_index_name,
            bulk_size=1000000
        )
    # bulk_size - это максимальное количество embeddings, которое можно будет поместить в индекс

    except Exception as e:
        st.error(f"Возникла ошибка при добавлении ваших файлов: {str(e)}")