import json
from faster_whisper import WhisperModel
from jiwer import wer
import tempfile

model = WhisperModel("small", device="cpu", compute_type="int8")

# Danh sách file test và ground truth
tests = [
    {"audio": "samples/1.mp3", "truth": "good morning everyone"},
    {"audio": "samples/2.mp3", "truth": "this is an english test"},
    {"audio": "samples/3.mp3", "truth": "whisper model works perfectly"},
]

total_wer = 0
for t in tests:
    segments, _ = model.transcribe(t["audio"], language="en")
    pred_text = " ".join([seg.text.strip().lower() for seg in segments])
    truth_text = t["truth"].strip().lower()

    score = wer(truth_text, pred_text)
    print(f"{t['audio']} — WER: {score:.3f}")
    total_wer += score

print("\nAverage WER:", total_wer / len(tests))
