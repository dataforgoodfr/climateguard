from typing import Dict
from faster_whisper import WhisperModel
import os
import tqdm
import pandas as pd


class Whisperizer:
    def __init__(self, name: str = "large-v3"):
        self.model = WhisperModel(name, device="cuda", compute_type="int8")

    def process_file(self, file_path) -> str:
        """Call whisper to get transcription"""
        segments, info = self.model.transcribe(
            file_path, language="fr"
        )  # beam_size=5, , condition_on_previous_text=False
        return " ".join([segment.text for segment in segments])

    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, str]:

        train_audio_files = [file for file in os.listdir(input_dir) if file.endswith(".mp3")]

        results = {}
        try:
            for file in tqdm.tqdm(train_audio_files):
                file_id = file.split(".")[0]
                results[file_id] = self.process_file(input_dir + "/" + file)

        except Exception as e:
            with open(output_dir + "/" + "save" + ".csv", "w") as f:
                f.write(results)
            raise e

        return results


if __name__ == "__main__":

    whisper = Whisperizer(name="tiny")

    train_dir = "data/MediatreeMP3/testAudio"
    output_dir = "data/results/test-large"

    reference_file = "data/test_ReAnnotatedData.csv"

    results = whisper.process_directory(input_dir=train_dir, output_dir=output_dir)

    ref_csv = pd.read_csv(reference_file)
    ref_csv["whisper_tiny"] = ref_csv["id"].apply(lambda x: results.get(x, ""))

    ref_csv.to_csv("data/results/result.csv")
