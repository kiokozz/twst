import streamlit as st
import openai
import os
import tempfile

# Функция для транскрипции с Whisper
def transcribe_audio(file_path, openai_api_key):
    client = openai.OpenAI(api_key=openai_api_key)
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription.text

# Функция для исправления текста (промпт)
def correct_text(text, api_client):
    response = api_client.chat.completions.create(
        model="gpt-4o-mini",  # Или "deepseek-chat" для DeepSeek
        messages=[
            {"role": "system", "content": "You are a helpful assistant that corrects text."},
            {"role": "user", "content": f"Исправь грамматические ошибки, опечатки и сделай текст coherent и естественным: {text}"}
        ]
    )
    return response.choices[0].message.content

# Функция для анализа текста (промпт)
def analyze_text(text, api_client):
    response = api_client.chat.completions.create(
        model="gpt-4o-mini",  # Или "deepseek-chat" для DeepSeek
        messages=[
            {"role": "system", "content": "You are an analyst that summarizes and analyzes text."},
            {"role": "user", "content": f"Проанализируй текст: опиши ключевые темы, эмоции, возможные улучшения и дай краткий summary: {text}"}
        ]
    )
    return response.choices[0].message.content

# Основное приложение Streamlit
def main():
    st.title("Аудио/Видео Транскриптор и Анализатор")
    st.write("Загрузите файл WAV или MP4, и приложение транскрибирует, исправит и проанализирует текст.")

    # Ввод API-ключей
    openai_api_key = st.text_input("OpenAI API Key (для Whisper и ChatGPT)", type="password")
    deepseek_api_key = st.text_input("DeepSeek API Key (опционально, для альтернативы ChatGPT)", type="password")

    # Загрузка файла
    uploaded_file = st.file_uploader("Загрузите файл (WAV или MP4)", type=["wav", "mp4"])

    if uploaded_file is not None and openai_api_key:
        # Сохраняем файл временно
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        try:
            # Транскрипция
            st.write("Транскрипция...")
            original_text = transcribe_audio(tmp_file_path, openai_api_key)
            st.subheader("Оригинальный текст:")
            st.text_area("Текст", original_text, height=200)

            # Выбор API для анализа (DeepSeek или ChatGPT)
            if deepseek_api_key:
                client = openai.OpenAI(
                    api_key=deepseek_api_key,
                    base_url="https://api.deepseek.com"
                )
                st.write("Используем DeepSeek API для анализа и исправления.")
            else:
                client = openai.OpenAI(api_key=openai_api_key)
                st.write("Используем ChatGPT API для анализа и исправления.")

            # Исправление
            st.write("Исправление текста...")
            corrected_text = correct_text(original_text, client)
            st.subheader("Исправленный текст:")
            st.text_area("Исправленный", corrected_text, height=200)

            # Анализ
            st.write("Анализ текста...")
            analysis = analyze_text(original_text, client)
            st.subheader("Анализ:")
            st.text_area("Анализ", analysis, height=200)

        except Exception as e:
            st.error(f"Ошибка: {str(e)}")
        finally:
            # Удаляем временный файл
            os.unlink(tmp_file_path)

if __name__ == "__main__":
    main()