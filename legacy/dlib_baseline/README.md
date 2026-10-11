# Legacy: dlib EAR baseline

This is the original version of the project, before the move to MediaPipe. It detects drowsiness from the Eye Aspect Ratio (EAR) alone, using dlib's 68-point landmark model and a fixed threshold. It is kept as the prior-work reference and is not maintained.

Run it from this folder with its own dependencies (Python 3.10–3.12):

```bash
pip install -r requirements.txt
streamlit run Drowsiness_Detection_w_UI.py
python Drowsiness_Detection.py
```
