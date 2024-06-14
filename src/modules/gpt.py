from openai import OpenAI
from io import BytesIO
import tempfile
import os

class BaseModel:
    def __init__(self):
        self.client = OpenAI()
# esta es la nueva branch
class WhisperModel(BaseModel):
    def __init__(self):
        super().__init__()

    def transcribe_audio(self, audio_file):
        file_extension = os.path.splitext(audio_file.name)[1]
        audio_bytes = audio_file.read()
        audio_buffer = BytesIO(audio_bytes)  # Crear un buffer en memoria con los bytes del audio

        # Crear un archivo temporal para almacenar los bytes del audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio_file:
            temp_audio_file.write(audio_buffer.read())  # Escribir el contenido del buffer al archivo temporal
            temp_audio_file.seek(0)  # Volver al inicio del archivo para lectura

            # Abrir el archivo temporal en modo binario para pasarlo a la API
            with open(temp_audio_file.name, "rb") as file_handle:
                transcription = self.client.audio.transcriptions.create(
                    file=file_handle,
                    model="whisper-1"
                )

        # Asegurar que todos los manejos del archivo se han cerrado antes de eliminarlo
        os.unlink(temp_audio_file.name)  # Eliminar el archivo temporal

        return transcription.text
class GPT(BaseModel):
    def __init__(self):
        super().__init__()

    def create_response(self,  prompt:str,model='gpt-3.5-turbo-16k-0613',temperature =0):
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat.completions.create(
            model = model,
            messages = messages,
            temperature = temperature
        )
        return response.choices[0].message.content


