from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_voyageai import VoyageAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
import os

class DocumentReader:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
                # Set a really small chunk size, just to show.
                chunk_size=1024,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )
        self.embeddings = VoyageAIEmbeddings(
            voyage_api_key=os.getenv("VOYAGER_API_KEY"), model="voyage-law-2"
        )
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world"))),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def load_and_split(self, file_path, wa_id):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = ''
        for doc in docs:
            text += doc.page_content

        texts = self.text_splitter.create_documents([text],metadatas=[{"user": f"{wa_id}"}]*len([text]))
        documents = texts[:3]
        return documents
    
    def add_documents(self, documents):
        uuids = [str(uuid4()) for _ in range(len(documents))]
        self.vector_store.add_documents(documents=documents, ids=uuids)
    
    def retrieve(self, query, wa_id, k=2):
        results = self.vector_store.similarity_search(query, k=k, filter={"user": f"{wa_id}"})
        output = ''
        for res in results:
            output+=f"* {res.page_content}\n\n"

    def get_ids_by_filter(self, filter_metadata):
        matching_ids = []
        for doc_id, doc in self.vector_store.docstore._dict.items():
            if all(doc.metadata.get(key) == value for key, value in filter_metadata.items()):
                matching_ids.append(doc_id)
        return matching_ids

    def delete_documents(self, wa_id):
        ids_to_delete = self.get_ids_by_filter({"user": f"{wa_id}"})
        self.vector_store.delete(ids=ids_to_delete)