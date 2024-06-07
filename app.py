import streamlit as st
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from openai import OpenAI
import os
from PIL import Image
# Set the page configuration as the first Streamlit command
st.set_page_config(layout='wide')

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Apply custom CSS styles
st.markdown(
    """
    <style>
    /* Set the background color and text color for the main container */
    [data-testid="stAppViewContainer"] {
        background-color: #228B22;  /* Planta verde */
    }
    /* Set the background color and text color for the header */
    [data-testid="stHeader"] {
        background-color: #228B22;
    }
    /* Set the background color and text color for the sidebar */
    [data-testid="stSidebar"] {
        background-color: #228B22;
    }

    /* Set the text color for all text elements */
    body, p, div, info {
        color: white}
    </style>
    """,
    unsafe_allow_html=True
)

# Function to classify the plant image
def classify_plant(img):
    np.set_printoptions(suppress=True)
    model_path = os.path.join(script_dir, "modelo", "keras_model.h5")
    model = load_model(model_path, compile=False)
    labels_path = os.path.join(script_dir, "modelo", "labels.txt")
    class_names = open(labels_path, "r").readlines()
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img.convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    return class_name, confidence_score

# Function to generate advice based on the plant's condition
def generate_advice(label):
    client = OpenAI(api_key="YOUR_API_KEY_HERE")
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Sos un experto en jardinería y tenes que recomendar solo 3 consejos para cuidar una planta que está {label}.",
        temperature=0.5,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text

# Streamlit App layout
# Agregar el logo al lado del título
st.image("logo.png", width=850) 










st.subheader("""Cargá una foto de una hoja de tu planta y determiná su estado.""")

st.subheader("""También podés generar consejos de cuidado 🌿""")

input_img = st.file_uploader("Elegir imagen", type=['jpg', 'png', 'jpeg'])

if input_img is not None:
    if st.button("Determinar estado de la planta"):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.info("Imagen cargada")
            st.image(input_img, use_column_width=True)
        with col2:
            st.info("Resultado")
            image_file = Image.open(input_img)
            with st.spinner('Analizando imagen...'):
                label, confidence_score = classify_plant(image_file)
                label_description = label.split(maxsplit=1)[1]  # Divide la etiqueta por el primer espacio y toma el segundo elemento
                label2 = label_description  # Guarda la descripción en label2
                st.success(label2)  # Muestra la etiqueta sin el número
        with col3:
            st.info("Consejos de cuidado")
            result = generate_advice(label2)
            st.success(result)
