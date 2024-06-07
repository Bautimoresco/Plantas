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
  h1, h2, h3, h4, h5, h6 {
    color: white; /* Cambia el color de los subtítulos */
}


    [data-testid="stAppViewContainer"] {
        background-color: #6AB66A;  /* Planta verde */
    }
   
    [data-testid="stHeader"] {
        background-color:#6AB66A;
    }

    /* Set the text color for all text elements */
    body, p, div, info, {
        color: white}
    
        /* Custom style for the plantIA information box */
    .info-box {
        background-color: rgba(255, 255, 255, 0.8);  /* Transparent white background */
        color: black;  /* Black text color */
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
.stTextInput>div>label,
.stTextArea>label,
.stSelectbox>label {
    color: white;
}

    

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

# Streamlit App layout
# Agregar el logo al lado del título
st.image("logo.png", width=850) 
# Streamlit App layout


col1, col2 = st.columns(2)
with col1:
    st.subheader("Acerca de la app")
    st.markdown(
        """
        <div class="info-box">
        plantIA es una aplicación diseñada para ayudarte a identificar el estado de salud de tus plantas y proporcionarte consejos útiles para su cuidado. Utilizando inteligencia artificial y aprendizaje automático, plantIA puede analizar imágenes de plantas y ofrecer diagnósticos precisos junto con recomendaciones de expertos.                                                                                                                                                                                                                                       
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("¿Cómo Funciona?")
    st.markdown(
        """
    <div class="info-box">
    
    1. Cargá una foto de tu planta.
    
    2. La IA analiza la imagen.
    
    3. Vas a recibir un diagnostico del estado de tu planta.
    </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.subheader("Beneficios")
    st.markdown(
        """
        <div class="info-box">
        
        - Diagnósticos Precisos: La IA ofrece resultados precisos y confiables.
        
        - Fácil de Usar: Cargá una foto y obtené resultados en cuestión de segundos.
       
        - Consejos para vos: Recibí recomendaciones específicas para el cuidado de tus plantas.
        </div>
        """,
        unsafe_allow_html=True
    )



st.subheader("""Cargá la foto🌿""")

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






def vista_consejos():
    st.subheader("Consejos de Cuidado")
    st.markdown(
        """
        <div class="info-box">

Cuidar de las plantas puede ser sencillo si sigues algunas pautas básicas:

- Riego: Riega tus plantas regularmente, pero evita el exceso de agua. Asegúrate de que el agua drene bien para prevenir la pudrición de raíces.
  
- Luz: Proporciona suficiente luz solar según las necesidades de cada planta. Algunas plantas prefieren luz indirecta, mientras que otras necesitan luz directa.
  
- Suelo: Utiliza un sustrato adecuado para cada tipo de planta. Añade compost o fertilizantes para mejorar la calidad del suelo.
  
- Poda: Poda las partes muertas o enfermas de la planta para fomentar un crecimiento saludable. La poda regular ayuda a mantener la forma y tamaño deseado de la planta.
  
- Humedad: Algunas plantas requieren alta humedad; puedes usar un humidificador o colocar un plato con agua cerca de la planta.
  
- Plagas: Inspecciona regularmente tus plantas para detectar plagas. Utiliza pesticidas naturales o soluciones caseras para mantener las plagas bajo control.

Siguiendo estos consejos, tus plantas estarán saludables y vibrantes.
        """,
        unsafe_allow_html=True
    )



# Agregar botón para cambiar a la vista de los consejos de cuidado
if st.button("Ver Consejos de Cuidado"):
    st.sidebar.write("Consejos de Cuidado")
    vista_consejos()