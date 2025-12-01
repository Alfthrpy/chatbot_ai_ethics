import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import Chonkie (Pastikan sudah terinstall atau folder src ada di path)
try:
    from chonkie import SentenceChunker
except ImportError:
    print("⚠️ Library Chonkie belum terdeteksi. Pastikan sudah 'pip install .' di folder repo chonkie.")
    exit()

PDF_SOURCE = "D:/CODING/PYTHON/chatbot_ai_ethics/dataset/kemendikbud.pdf"    
TXT_RAW_OUTPUT = "temp_raw.txt"        
TXT_CLEAN_OUTPUT = "temp_clean.txt"   
DB_OUTPUT_FOLDER = "faiss_index_local"

def step_1_extract():
    print(f"\n[1/4] 📄 Extracting {PDF_SOURCE}...")
    if not os.path.exists(PDF_SOURCE):
        print("❌ File PDF tidak ditemukan!")
        return False

    loader = PyPDFLoader(PDF_SOURCE)
    docs = loader.load()

    full_text = "\n".join([d.page_content for d in docs])
    
    with open(TXT_RAW_OUTPUT, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"✅ Tersimpan di: {TXT_RAW_OUTPUT}")
    return True

def step_2_preprocess():
    print(f"\n[2/4] 🧹 Preprocessing Text...")
    
    with open(TXT_RAW_OUTPUT, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Ganti newline dengan spasi
    text = text.replace("\n", " ")
    
    # 2. Hapus simbol-simbol aneh selain tanda baca dasar (.,?!) dan angka/huruf

    text = re.sub(r'[^\w\s.,?!-]', '', text)
    
    # 3. Hapus spasi berlebih (multiple spaces -> single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    with open(TXT_CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        f.write(text)
        
    return True

def step_3_chonker():
    print(f"\n[3/4] 🔪 Chunking with Chonkie...")
    
    with open(TXT_CLEAN_OUTPUT, "r", encoding="utf-8") as f:
        clean_text = f.read()


    chunker = SentenceChunker(
        chunk_size=512,
        chunk_overlap=128
    )
    
    chunks = chunker.chunk(clean_text)
    
    lc_documents = []
    for i, c in enumerate(chunks):
        doc = Document(
            page_content=c.text,
            metadata={
                "source": PDF_SOURCE,
                "chunk_id": i,
                "token_count": c.token_count
            }
        )
        lc_documents.append(doc)
        
    print(f"✅ Berhasil memecah menjadi {len(lc_documents)} chunks.")
    return lc_documents

def step_4_embed(documents):
    print(f"\n[4/4] 🧠 Embedding & Saving to FAISS...")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = FAISS.from_documents(documents, embeddings)
    
    vector_store.save_local(DB_OUTPUT_FOLDER)

if __name__ == "__main__":
    if step_1_extract():
        if step_2_preprocess():
            final_docs = step_3_chonker()
            step_4_embed(final_docs)