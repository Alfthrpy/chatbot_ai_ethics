import streamlit as st
from service.chat import RAGService
st.set_page_config(page_title="Edu-AI Ethics Guard", page_icon="⚖️", layout="centered")

st.title("⚖️ Asisten Etika AI Pendidikan")
st.markdown("""
    *Panduan cerdas untuk penggunaan AI yang bertanggung jawab di lingkungan akademik.*
    """)

with st.sidebar:
    st.header("🎓 Panel Kontrol")
    api_key = "AIzaSyD7Fht7mQvV7JREt4UD8j0K4DhuVM_ClSk"
    
    st.divider()
    
    st.subheader("💡 Contoh Pertanyaan")
    st.info("""
    - "Apakah menggunakan ChatGPT untuk esai itu plagiarisme?"
    - "Bagaimana cara sitasi konten hasil AI?"
    - "Apa risiko privasi data siswa saat pakai AI?"
    - "Berikan pedoman etis untuk dosen."
    """)

    st.divider()
    if st.button("🔄 Reset Percakapan"):
        st.session_state.messages = []
        st.rerun()

# Inisialisasi Service
if "rag_service" not in st.session_state and api_key:
    try:
        st.session_state.rag_service = RAGService(google_api_key=api_key)
    except Exception as e:
        st.error(f"Gagal memuat sistem: {e}")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya di sini untuk membantu Anda memahami etika penggunaan AI dalam pendidikan. Apa yang ingin Anda diskusikan hari ini?"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ketik pertanyaan etika akademik di sini..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "rag_service" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Menganalisis pedoman etika..."):
                try:
                    answer, sources = st.session_state.rag_service.chat(
                        prompt, 
                        st.session_state.messages[:-1]
                    )
                    
                    st.markdown(answer)
                    
                    if sources:
                        with st.expander("📖 Referensi Pedoman"):
                            for i, doc in enumerate(sources):
                                st.caption(f"**Kutipan {i+1}:** \"{doc.page_content[:250]}...\"")
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Sistem sedang memuat, harap tunggu sebentar.")