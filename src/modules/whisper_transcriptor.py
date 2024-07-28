import tempfile
import os
from openai import OpenAI
import logging
from typing import BinaryIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WhisperModel:
    def __init__(self):
        self.client = OpenAI()

    def transcribe_audio(self, audio_file: BinaryIO) -> str:
        """
        Transcribe el audio usando el modelo Whisper de OpenAI.

        :param audio_file: Archivo de audio a transcribir
        :return: Texto transcrito
        """
        try:
            file_extension = os.path.splitext(audio_file.name)[1]
            audio_bytes = audio_file.read()

            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio_file:
                temp_audio_file.write(audio_bytes)
                temp_audio_file.flush()

                with open(temp_audio_file.name, "rb") as file_handle:
                    transcription = self.client.audio.transcriptions.create(
                        file=file_handle,
                        model="whisper-1"
                    )

            os.unlink(temp_audio_file.name)
            return transcription.text
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            raise
