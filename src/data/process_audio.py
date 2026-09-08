# src/data/process_audio.py
from pathlib import Path
from typing import Optional, Tuple, Union

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_mel_spectrogram_data(
    audio_path: Union[str, Path],
    sr: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Loads an audio file and generates its Mel spectrogram.

    Returns:
        (mel_db, sr_loaded) or (None, None) on error.
    """
    try:
        y: np.ndarray
        sr_loaded: int
        y, sr_loaded = librosa.load(audio_path, sr=sr)

        mel_spectrogram: np.ndarray = librosa.feature.melspectrogram(
            y=y, sr=sr_loaded, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        mel_spectrogram_db: np.ndarray = librosa.power_to_db(
            mel_spectrogram, ref=np.max
        )

        return mel_spectrogram_db, sr_loaded
    except Exception as e:
        print(f"Error generating Mel spectrogram data for {audio_path}: {e}")
        return None, None


def save_spectrogram_npy(
    spectrogram_data: np.ndarray, out_path: Union[str, Path]
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, spectrogram_data)
    return out_path


def preprocess_and_save(
    audio_path: Union[str, Path],
    out_path: Union[str, Path],
    sr: int = 32000,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> bool:
    """Load audio -> mel -> save.npy. Returns True on success."""
    mel_db, _ = generate_mel_spectrogram_data(
        audio_path, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    if mel_db is None:
        return False

    save_spectrogram_npy(mel_db, out_path)
    return True


def load_local_spectrogram(npy_path: Union[str, Path]) -> np.ndarray:
    npy_path = Path(npy_path)
    if not npy_path.exists():
        raise FileNotFoundError(f"Spectrogram not found: {npy_path}")
    return np.load(npy_path)


def visualize_mel_spectrogram(
    spectrogram_data: np.ndarray,
    sr: int,
    title: str = "Mel Spectrogram",
    hop_length: int = 512,
) -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram_data, sr=sr, x_axis="time", y_axis="mel", hop_length=hop_length
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def save_spectrogram_image(
    spectrogram_data: np.ndarray,
    sr: int,
    output_path: Union[str, Path],
    hop_length: int = 512,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        spectrogram_data, sr=sr, x_axis="time", y_axis="mel", hop_length=hop_length
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
