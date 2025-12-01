import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGService:
    def __init__(self, google_api_key):
        os.environ["GOOGLE_API_KEY"] = google_api_key
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = FAISS.load_local("faiss_index_local", self.embeddings, allow_dangerous_deserialization=True)

    def format_history(self, history):
        """Mengubah list pesan chat menjadi string untuk prompt."""
        formatted = ""
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"{role}: {msg['content']}\n"
        return formatted

    def chat(self, query, history=[]):
        if not self.vector_store:
            return "Index belum dimuat.", []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        template = """
        Kamu adalah asisten AI yang menjawab berdasarkan dokumen.

        Riwayat Percakapan:
        {chat_history}

        Konteks Dokumen:
        {context}

        Pertanyaan:
        {question}

        Jawaban:
        """
        
        prompt = PromptTemplate.from_template(template)
        
        history_str = self.format_history(history)

        rag_chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: history_str
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(query)
        source_docs = retriever.invoke(query)
        
        return answer, source_docs