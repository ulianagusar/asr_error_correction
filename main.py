import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
import re
import kenlm
import tempfile
import os
from io import BytesIO
import soundfile as sf
import numpy as np
from audio_recorder_streamlit import audio_recorder
from prep_data import add_sim_panphon  


st.set_page_config(
    page_title="Медична ASR корекція",
    page_icon="🩺",
    layout="wide"
)


@st.cache_resource
def load_models():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
  
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model_path = "/Users/ulanagusar/Desktop/4_курс/diplom/Uni_Syn_Med/model_no_nlt_8ep_final_promt/medical_asr_correction_model_best"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30
    )
    

    kenlm_model = kenlm.Model("3gram.bin")
    
    return tokenizer, model, asr_pipe, kenlm_model, device

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_single(text, tokenizer, model, device, max_length=128):
    
    PROMPT = "correct the ASR output, use phonetically similar words in brackets: "

    inputs = tokenizer(
        PROMPT + text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
    
    prediction = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return prediction

def process_audio_file(audio_file, asr_pipe):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name
    
    try:

        result = asr_pipe(tmp_path)
        return result["text"]
    finally:

        os.unlink(tmp_path)

def process_recorded_audio(audio_bytes, asr_pipe):

    if audio_bytes:

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:

            result = asr_pipe(tmp_path)
            return result["text"]
        finally:

            os.unlink(tmp_path)
    return None


with st.spinner("Завантаження моделей..."):
    tokenizer, model, asr_pipe, kenlm_model, device = load_models()


st.title("Система корекції розпізнавання мовлення")



with st.sidebar:
    st.header("Інформація")
    st.info("""
    **Як використовувати:**
    1. Оберіть спосіб введення аудіо
    2. Запишіть або завантажте аудіо
    3. Отримайте розпізнаний текст
    4. Переглядьте результати корекції
    """)
    
    st.header(" Параметри")
    show_intermediate = st.checkbox("Показати проміжні кроки", True)


col1, col2 = st.columns([1, 1])

with col1:
    st.header("Введення аудіо")
    

    input_method = st.radio(
        "Оберіть спосіб:",
        ["Записати аудіо", "Завантажити файл"]
    )
    
    recognized_text = None
    
    if input_method == "Записати аудіо":
        st.subheader("Запис аудіо")
        audio_bytes = audio_recorder(
            text="Натисніть для запису",
            recording_color="#e74c3c",
            neutral_color="#34495e",
            icon_name="microphone",
            icon_size="2x"
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            
            if st.button("Розпізнати мовлення", type="primary"):
                with st.spinner("Розпізнавання мовлення..."):
                    try:
                         recognized_text = process_recorded_audio(audio_bytes, asr_pipe)
                    except ValueError as e:
                          recognized_text = "no_data"
                    
    
    else:  
        st.subheader("Завантаження аудіофайлу")
        uploaded_file = st.file_uploader(
            "Оберіть аудіофайл",
            type=["wav", "mp3",  "flac"],
            help="Підтримувані формати: WAV, MP3, FLAC"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format="audio/wav")
            
            if st.button("Розпізнати мовлення", type="primary"):
                with st.spinner("Розпізнавання мовлення..."):
                    try:
                         recognized_text = process_audio_file(uploaded_file, asr_pipe)
                    except ValueError as e:
                          recognized_text = "no_data"
                         
with col2:
    st.header("Результати")
    if recognized_text == "no_data":
        st.error(f"Помилка при обробці аудіо , спробуйте записати ще раз")
    elif recognized_text:
        cleaned_text = clean_text(recognized_text)
        
        st.subheader("Розпізнаний текст:")
        st.text_area("", value=recognized_text, height=100, disabled=True)
        
        if show_intermediate:
            st.subheader("Очищений текст:")
            st.text_area("", value=cleaned_text, height=80, disabled=True)
        

        with st.spinner("Обробка та корекція тексту..."):
            try:
   
                text_phon_add = add_sim_panphon(cleaned_text, kenlm_model)
                
                if show_intermediate:
                    st.subheader("Текст з фонетичними варіантами:")
                    st.text_area("", value=text_phon_add, height=100, disabled=True)
                

                corrected_text = predict_single(
                    text_phon_add, tokenizer, model, device, 128
                )
                
  
                st.subheader("Виправлений текст:")
                st.text_area(
                    "", 
                    value=corrected_text, 
                    height=100, 
                    disabled=True,
                    key="final_result"
                )
                
                
            except Exception as e:
                st.error(f"Помилка при обробці: {str(e)}")
    
    else:
        st.info("👆 Спочатку запишіть або завантажте аудіо")

