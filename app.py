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
         background-color: rgba(0, 128, 0, 0.55); /* Fondo ligeramente verde */
        color: black; /* Texto en negro */
        padding: 20.5px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    .info-box1 {
         background-color: rgba(0, 128, 0, 0.55); /* Fondo ligeramente verde */
        color: black; /* Texto en negro */
        padding: 28.5px;
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
        background-color: #f5deb3; /* Fondo blanco */
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
    st.subheader("*Acerca de la app*")
    st.markdown(
        """
        <div class="info-box1">
        <b>🌿 Identificación precisa</b>: plantIA es una aplicación diseñada para ayudarte a identificar el estado de salud de tus plantas y proporcionarte consejos útiles para su cuidado.
        
        <br><b>🤖 Tecnología avanzada</b>: Utilizando inteligencia artificial y aprendizaje automático, plantIA puede analizar imágenes de plantas y ofrecer diagnósticos precisos junto con recomendaciones de expertos.
        
        <br><b>📍 Encuentra viveros</b>: plantIA te permite encontrar viveros cercanos para adquirir todo lo necesario para el cuidado de tus plantas.
        
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.subheader("*Beneficios*")
    st.markdown(
        """
        <div class="info-box">
        <b>🌟 Diagnósticos Precisos</b>: La IA ofrece resultados precisos y confiables.
       
       <br><b>📸 Fácil de Usar</b>: Cargá una foto y obtené resultados en cuestión de segundos.
        
        <br><b>💡 Consejos para vos</b>: Recibí recomendaciones específicas para el cuidado de tus plantas.

         <br><b>🧑‍🔬 Ayuda profesional</b>: Alianzas con una gran cantidad de viveros, para brindarte el mejor asesoramiento.
        </div>
        """,
        unsafe_allow_html=True
    )

col1, col2, col3 = st.columns(3)

with col2:
    st.subheader("*¿Cómo Funciona?*")
    st.markdown(
        """
        <div class="info-box1">
        <b>📸 Paso 1</b>: Cargá una foto de una hoja de tu planta.

        <br><b>🤖 Paso 2</b>: La IA analiza la imagen.

        <br><b>📊 Paso 3</b>: Vas a recibir un diagnóstico del estado de tu planta.
        </div>
        """,
        unsafe_allow_html=True
    )

st.subheader("*Cargá la foto🌿*")

input_img = st.file_uploader("Sube una imagen de una hoja de tu planta", type=["jpg", "jpeg", "png"])

# Recomendaciones para plantas en buen estado
good_recommendations = [
    "*🌿 Proporciona luz adecuada y riego regular según las necesidades de tu planta.*",
    "*🌞 Asegúrate de que tu planta reciba suficiente luz solar diariamente.*",
    "*💧 Mantén el nivel de humedad apropiado en la tierra para el tipo de planta que tienes.*",
    "*🌱 Fertiliza tu planta de manera regular para mantenerla saludable y en crecimiento.*",
    "*🌸 Revisa regularmente si hay signos de plagas o enfermedades en tu planta.*"
]

# Recomendaciones para plantas en mal estado
bad_recommendations = [
    "*⚠️ Proporciona más agua y asegúrate de que reciba suficiente luz solar.*",
    "*🔍 Revisa si hay plagas y trátalas adecuadamente.*",
    "*💡 Asegúrate de que tu planta reciba al menos 6 horas de luz al día.*",
    "*🌾 Comprueba si la tierra está demasiado seca o demasiado húmeda.*",
    "*🛠️ Podar las hojas dañadas puede ayudar a la planta a recuperarse.*"
]

# Dentro del bloque if input_img is not None:
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
                
                if confidence_score < 0.80:
                    st.warning(f"La confianza en el resultado es baja ({confidence_score * 100:.2f}%). Por favor, intenta con otra foto.")
                else:
                    st.success(f"{label_description} (Confianza: {confidence_score * 100:.2f}%)")

        with col3:
            st.info("Recomendaciones")
            if label_description.strip().lower() == "mal estado":
                for rec in bad_recommendations:
                    st.markdown(rec)
            else:
                for rec in good_recommendations:
                    st.markdown(rec)


    st.subheader("*Necesitas la ayuda de un profesional?🧑‍🔬🌱*")


    with st.expander("Mapa de Viveros Cercanos 🗺️🌿"):
        user_location = st.text_input("Ingrese su dirección para encontrar viveros cercanos:")

        if user_location:
            st.write(f"Ubicación ingresada: {user_location}")
            if st.button("Ver viveros cercanos en Google Maps"):
                google_maps_url = f"https://www.google.com/maps/search/?api=1&query=viveros+cercanos+{user_location}"
                webbrowser.open_new_tab(google_maps_url)