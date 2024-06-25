import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import webbrowser


# Set the page configuration as the first Streamlit command
st.set_page_config(layout='wide')

# Get the absolute path to the directory containing the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Apply custom CSS styles
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 {
        color: black; /* Títulos en negro */
        font-weight: bold; /* Títulos en negrita */
    }

    body, p, div, info, {
        color: black; /* Texto en negro */
    }

    .info-box {
         background-color: rgba(0, 255, 0, 0.25); /* Fondo ligeramente verde */
        color: black; /* Texto en negro */
        padding: 20.5px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .info-box1 {
         background-color: rgba(0, 255, 0, 0.25); /* Fondo ligeramente verde */
        color: black; /* Texto en negro */
        padding: 20.5px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .stTextInput>div>label,
    .stTextArea>label,
    .stSelectbox>label {
        color: black; /* Etiquetas de entrada en negro */
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: white; /* Fondo blanco */
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





# Streamlit App layout
col1, col2, col3 = st.columns(3)

with col2:
   
   st.image("logo.png", width=350)



col1, col2 = st.columns(2)
with col1:
    st.subheader("Acerca de la app")
    st.markdown(
        """
        <div class="info-box1">
    plantIA es una aplicación diseñada para ayudarte a identificar el estado de salud de tus plantas y proporcionarte consejos útiles para su cuidado. Utilizando inteligencia artificial y aprendizaje automático, plantIA puede analizar imágenes de plantas y ofrecer diagnósticos precisos junto con recomendaciones de expertos. Además, plantIA te permite encontrar viveros cercanos para adquirir todo lo necesario para el cuidado de tus plantas.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.subheader("¿Cómo Funciona?")
    st.markdown(
        """
        <div class="info-box1">
    
    1. Cargá una foto de una hoja de tu planta.
    
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

st.subheader("Cargá la foto🌿")

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
                label_description = label.split(maxsplit=1)[1] if len(label.split()) > 1 else label
                st.success(f"{label_description} (Confianza: {confidence_score * 100:.2f}%)")
                st.write(label_description)
        with col3:
            st.info("Recomendaciones")
            if label_description == "Mal estado":
                st.write("Cuida tu planta")
            else:
                st.write("Tu planta esta bien")
            

                    
                

        with st.expander("Mapa de Viveros Cercanos 🗺️🌿"):
                    # Obtener ubicación del usuario
                    user_location = st.text_input("Ingrese su dirección para encontrar viveros cercanos:")

                    if user_location:
                        st.write(f"Tu Ubicación: {user_location}")
                        if st.button("Ver Viveros Cercanos"):
                            google_maps_url = f"https://www.google.com/maps/search/?api=1&query=viveros+cercanos+{user_location}"
                            webbrowser.open_new_tab(google_maps_url)

        with st.expander("Recomendaciones para el Cuidado de la Planta 🌱"):
                if st.button("Generar Recomendaciones"):
                        with st.spinner('Generando recomendaciones...'):
                            recommendation = generate_plant_recommendation(label_description)
                            st.write(recommendation)

 
