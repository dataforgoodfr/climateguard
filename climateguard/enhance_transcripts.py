import os
import time
import torch
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel
from typing import Dict, List, Tuple
from evaluate import load


class Whisperizer:
    def __init__(self, name: str, compute_type: str = "int8", device: str = "cuda"):
        self.model = WhisperModel(name, device=device, compute_type=compute_type)

    def process_file(self, file_path: str, beam_size: int = 5) -> str:
        """Transcribe an audio file."""
        torch.cuda.empty_cache()
        segments, _ = self.model.transcribe(file_path, language="fr", beam_size=beam_size)
        return " ".join(segment.text for segment in segments)

    def process_directory(
        self, audio_dir: str, output_dir: str, beam_size: int = 5
    ) -> Tuple[Dict[str, str], float]:
        """Process all audio files in a directory."""
        audio_files = [
            file.split(".")[0] for file in os.listdir(audio_dir) if file.endswith(".mp3")
        ]

        results: Dict[str, str] = {}

        start_time = time.time()
        try:
            for file_id in tqdm(audio_files, desc="Processing Audio Files"):
                results[file_id] = self.process_file(
                    os.path.join(audio_dir, file_id + ".mp3"), beam_size
                )
        except Exception as e:
            pd.DataFrame.from_dict(results, orient="index", columns=["transcript"]).to_csv(
                os.path.join(output_dir, "save.csv")
            )
            raise e

        total_time = time.time() - start_time
        return results, total_time


def evaluate_models(
    models: List[str],
    compute_types: List[str],
    beam_sizes: List[int],
    audio_dir: str,
    output_dir: str,
    reference_file: str,
):
    """Evaluate different model configurations and record results."""
    ref_csv = pd.read_csv(reference_file)
    results_summary = []

    try:
        for model in models:
            for compute in compute_types:
                for beam in beam_sizes:
                    torch.cuda.empty_cache()
                    whisper = Whisperizer(name=model, compute_type=compute)
                    print(f"Evaluating: Model={model}, Compute={compute}, Beam={beam}")
                    results, exec_time = whisper.process_directory(audio_dir, output_dir, beam)
                    ref_csv[f"whisper_{model}_{compute}_beam{beam}"] = ref_csv["id"].apply(
                        lambda x: results.get(x, "")
                    )
                    results_summary.append(
                        {
                            "model": model,
                            "compute_type": compute,
                            "beam_size": beam,
                            "execution_time": exec_time,
                        }
                    )
    except Exception as e:
        pd.DataFrame(results_summary).to_csv(
            os.path.join(output_dir, "evaluation_results.csv"), index=False
        )
        ref_csv.to_csv(os.path.join(output_dir, "transcriptions.csv"), index=False)
        raise e

    pd.DataFrame(results_summary).to_csv(
        os.path.join(output_dir, "evaluation_results.csv"), index=False
    )
    ref_csv.to_csv(os.path.join(output_dir, "transcriptions.csv"), index=False)


def compute_wer(predictions, references):
    """https://huggingface.co/spaces/evaluate-metric/wer"""
    wer = load("wer")
    wer_score = wer.compute(predictions=predictions, references=references)
    return wer_score


if __name__ == "__main__":
    # Eval models
    audio_dir = "data/pos_AxvaFzQm_31/pos_AxvaFzQm_Audio"
    ref_sheet = "data/pos_AxvaFzQm_31/pos_AxvaFzQm_annotated.csv"
    output_dir = "data/results"
    output_path = "data/results/pos_AxvaFzQm_annotated_whisper.csv"
    # reference_file = "data/test_ReAnnotatedData.csv"

    # model_variants = ["tiny", "base", "medium", "large-v3"]
    # compute_variants = ["int8", "float16"]
    # beam_sizes = [1, 5, 10]
    # evaluate_models(
    #     model_variants, compute_variants, beam_sizes, audio_dir, output_dir, reference_file
    # )

    whisper = Whisperizer(name="large-v3", compute_type="int8")
    transcripts, total_time = whisper.process_directory(
        audio_dir=audio_dir, output_dir=output_dir, beam_size=5
    )
    print(f"Time for processing: {total_time}")

    ref_df = pd.read_csv(ref_sheet)
    transcripts_df = (
        pd.DataFrame.from_dict(transcripts, orient="index", columns=["whisper-largev3"])
        .reset_index()
        .rename(columns={"index": "id"})
    )

    results = pd.merge(left=ref_df, right=transcripts_df, on="id", how="left")

    results.to_csv(output_path)
