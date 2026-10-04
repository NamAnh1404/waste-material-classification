# Waste Material Classification

A data-science project that classifies waste materials into nine categories and
provides practical recycling guidance through a Streamlit interface.

## Live demo

https://namanh-waste-classification.streamlit.app/

## Highlights

- Nine material classes: aluminium, carton, foam box, milk box, paper, paper
  cup, PET, plastic cup, and other waste
- EfficientNetB0 inference in the public demo
- Image upload and camera input
- Confidence scores, probability chart, and disposal recommendations
- Training and evaluation source code for CNN, EfficientNetB0, and ResNet50

The public repository excludes the 2.7 GB training dataset, virtual environment,
and large experimental checkpoints. It includes the compact EfficientNet model
required by the demo.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Tech stack

Python, TensorFlow, EfficientNetB0, Streamlit, NumPy, pandas, Altair, and Pillow.
