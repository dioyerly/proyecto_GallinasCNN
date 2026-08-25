import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# 1. Configuración de página
st.set_page_config(
    page_title="Guardián Avícola IA - Detector Inteligente",
    page_icon="🐔",
    layout="wide"
)

# Estilos CSS
st.markdown("""
<style>
.stApp {
    background-color: #FCFBF7;
    color: #2E3B33;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    color: #2C5E3B !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    text-align: center;
    margin-bottom: 0px !important;
}
.subtitle {
    text-align: center;
    color: #7A6F5D;
    font-size: 1.05rem !important;
    margin-bottom: 20px;
    font-style: italic;
}
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #2C5E3B !important;
    background-color: #F4F1EA !important;
    border-radius: 10px !important;
    padding: 15px !important;
}

/* Control de imagen a 350px */
div[data-testid="stImage"] img {
    max-height: 500px !important;
    width: auto !important;
    object-fit: contain !important;
    border-radius: 10px;
    display: block;
    margin: 0 auto 5px auto;
}

div.stButton > button:first-child {
    background-color: #2C5E3B !important;
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: bold !important;
    font-size: 15px !important;
    border: none !important;
    width: 100% !important;
    margin-top: 10px;
}
div.stButton > button:first-child:hover {
    background-color: #8B2500 !important;
    color: #FFF !important;
}

.result-card {
    padding: 15px !important;
    border-radius: 12px !important;
    text-align: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-top: 5px;
}
.threat {
    background-color: #FDF2F2;
    border: 2px solid #EC5B5B;
    color: #9B1C1C;
}
.safe {
    background-color: #F2FDF5;
    border: 2px solid #5BEC8C;
    color: #1C5A27;
}
.info-section {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px solid rgba(0,0,0,0.1);
    text-align: left;
}
.info-section p {
    font-size: 0.9rem !important;
    margin: 3px 0 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Guardián Avícola IA 🐔</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Sistema de Detección de Amenazas en Corrales con Redes Neuronales Convolucionales (CNN)</p>", unsafe_allow_html=True)

HUMAN_KEYWORDS = [
    "groom", "bridegroom", "apron", "trench_coat", "coat", "suit", "jersey", "sweatshirt", 
    "t-shirt", "jeans", "jean", "sombrero", "cowboy_hat", "academic_gown", "scuba_diver", 
    "ballplayer", "swimming_trunks", "military_uniform", "lab_coat", "overcoat", "cardigan", 
    "pajamas", "bikini", "miniskirt", "poncho", "stole", "diaper", "bonnet", "mortarboard", 
    "gasmask", "wig", "sunglasses", "eyeglasses", "person", "human", "man", "woman", 
    "boy", "girl", "pedestrian", "bather", "doctor", "soldier", "police", "fireman", 
    "gardener", "farmer", "necktie", "bow_tie", "hair_slide", "safety_pin"
]

@st.cache_resource
def cargar_modelo_personalizado():
    ruta_modelo = "modelo_procesado/depredadores_modelo.keras"
    if os.path.exists(ruta_modelo):
        return tf.keras.models.load_model(ruta_modelo)
    return None

@st.cache_resource
def cargar_modelo_general():
    try:
        return tf.keras.applications.MobileNetV2(weights="imagenet")
    except Exception as e:
        return None

def generar_mapa_calor(modelo, img_array):
    try:
        base_model = modelo.get_layer("xception")
    except Exception:
        base_model = modelo.layers[1]

    grad_model = tf.keras.models.Model(
        inputs=[base_model.inputs],
        outputs=[base_model.get_layer("block14_sepconv2_act").output, base_model.output]
    )

    with tf.GradientTape() as tape:
        inputs_normalized = (img_array / 127.5) - 1.0
        conv_outputs, predictions = grad_model(inputs_normalized)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def superponer_mapa(imagen_pil, heatmap):
    img_np = np.array(imagen_pil)
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superposicion = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)
    return superposicion

# Función para generar el gráfico de probabilidades estilizado
def generar_grafico_estilizado(prob_amenaza, prob_seguro):
    categories = ['Amenaza', 'No Amenaza']
    values = [prob_amenaza, prob_seguro]
    colors = ['#EC5B5B', '#5BEC8C']
    
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(categories, values, color=colors, height=0.55, edgecolor='black', linewidth=0.8)
    
    ax.set_xlim(0, 100)
    ax.set_xlabel('Probabilidad (%)', fontsize=10, fontweight='bold', color='#2E3B33')
    ax.set_title('Ponderación de Salida de la Red (Capa Sigmoide)', fontsize=11, fontweight='bold', color='#2C5E3B')
    
    # Agregar etiquetas numéricas en cada barra
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', ha='left', fontsize=10, fontweight='bold', color='#2E3B33')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7A6F5D')
    ax.spines['bottom'].set_color('#7A6F5D')
    ax.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    return fig

modelo_custom = cargar_modelo_personalizado()
modelo_general = cargar_modelo_general()

if modelo_custom is None:
    st.error("⚠️ No se encontró el modelo en `modelo_procesado/depredadores_modelo.keras`.")
else:
    tab1, tab2 = st.tabs(["🔍 **Detector en Vivo**", "📊 **Atención de la Red y Métricas (TP)**"])

    with tab1:
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("1. Cargar Fotografía")
            archivo_subido = st.file_uploader(
                "Arrastra o selecciona una foto para inspeccionar el corral", 
                type=["jpg", "jpeg", "png"],
                key="uploader_tab1"
            )
            
            btn_escanear = False
            if archivo_subido is not None:
                btn_escanear = st.button("🔍 Escanear Corral")

        with col2:
            st.subheader("2. Resultado del Análisis")
            if archivo_subido is not None:
                imagen = Image.open(archivo_subido)
                st.image(imagen, caption="Imagen cargada", use_container_width=False)

                if btn_escanear:
                    with st.spinner("Procesando imagen con la Red Neuronal..."):
                        if imagen.mode != "RGB":
                            imagen = imagen.convert("RGB")
                        
                        imagen_redimensionada = imagen.resize((299, 299))
                        imagen_array_custom = np.array(imagen_redimensionada)
                        imagen_array_custom = np.expand_dims(imagen_array_custom, axis=0)
                        
                        prediccion_custom = modelo_custom.predict(imagen_array_custom)
                        probabilidad_custom = prediccion_custom[0][0]
                        
                        prob_seguro = probabilidad_custom * 100
                        prob_amenaza = (1 - probabilidad_custom) * 100
                        
                        # Guardar valores en sesión para usarlos en el gráfico de la pestaña 2
                        st.session_state['prob_amenaza'] = prob_amenaza
                        st.session_state['prob_seguro'] = prob_seguro
                        
                        es_humano = False
                        confianza_humano = 0.0
                        
                        if modelo_general is not None:
                            imagen_general = imagen.resize((224, 224))
                            imagen_array_general = np.array(imagen_general)
                            imagen_array_general = tf.keras.applications.mobilenet_v2.preprocess_input(imagen_array_general)
                            imagen_array_general = np.expand_dims(imagen_array_general, axis=0)
                            
                            predicciones_general = modelo_general.predict(imagen_array_general)
                            decodificadas = tf.keras.applications.mobilenet_v2.decode_predictions(predicciones_general, top=5)[0]
                            
                            for pred in decodificadas:
                                clase_ingles = pred[1].lower()
                                if any(kw in clase_ingles for kw in HUMAN_KEYWORDS):
                                    es_humano = True
                                    confianza_humano = pred[2] * 100
                                    prob_seguro = 100.0
                                    prob_amenaza = 0.0
                                    st.session_state['prob_amenaza'] = 0.0
                                    st.session_state['prob_seguro'] = 100.0
                                    break

                        if es_humano:
                            st.markdown(f"""
                            <div class="result-card safe" style="background-color: #F0F4FD; border-color: #5B9BEC; color: #1C3B5E;">
                                <h3 style="margin:0; color:#1C3B5E;">🧑‍🌾 BIENVENIDA: GRANJERO DETECTADO</h3>
                                <p style="font-size: 0.95rem; margin: 5px 0;">Personal autorizado (Confianza: <strong>{confianza_humano:.2f}%</strong>).</p>
                                <div class="info-section">
                                    <p style="margin: 2px 0; color:#2E456A;"><strong>Estado del Sistema:</strong> Alarmas pausadas temporalmente para el desarrollo de tareas autorizadas.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        elif probabilidad_custom < 0.5:
                            st.markdown(f"""
                            <div class="result-card threat">
                                <h3 style="margin:0; color:#9B1C1C;">⚠️ ALERTA: AMENAZA DETECTADA</h3>
                                <p style="font-size: 0.95rem; margin: 5px 0;">Nivel de amenaza: <strong>{prob_amenaza:.2f}%</strong> de confianza.</p>
                                <div class="info-section">
                                    <p style="margin: 2px 0; color:#4B1C1C;"><strong>Diagnóstico del Modelo:</strong> La imagen contiene características visuales asociadas a depredadores (zorros, félidos, roedores u otros peligros silvestres).</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-card safe">
                                <h3 style="margin:0; color:#1C5A27;">✅ ENTORNO SEGURO</h3>
                                <p style="font-size: 0.95rem; margin: 5px 0;">Sin amenazas detectadas (Confianza: <strong>{prob_seguro:.2f}%</strong>).</p>
                                <div class="info-section">
                                    <p style="margin: 2px 0; color:#2E6A3B;"><strong>Diagnóstico del Modelo:</strong> La imagen corresponde a un entorno pacífico, animales no cazadores o aves de corral autorizadas.</p>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("👈 Sube una imagen en la izquierda para ver el análisis aquí.")

    with tab2:
        st.header("📊 Métricas de Rendimiento e Interpretación (Grad-CAM)")
        
        # Muestra el gráfico de la última predicción escaneada en la Tab 2
        st.subheader("1. Distribución de Probabilidades de la Última Predicción")
        if 'prob_amenaza' in st.session_state and 'prob_seguro' in st.session_state:
            fig_barras = generar_grafico_estilizado(st.session_state['prob_amenaza'], st.session_state['prob_seguro'])
            st.pyplot(fig_barras)
        else:
            st.info("💡 Realiza un análisis en el 'Detector en Vivo' (Tab 1) para visualizar aquí el desglose gráfico de la predicción.")

        st.markdown("---")
        st.subheader("2. Mapa de Calor Visual (¿Qué mira la IA?)")
        archivo_explicacion = st.file_uploader(
            "Carga una imagen para visualizar las zonas que activan la red neuronal", 
            type=["jpg", "jpeg", "png"],
            key="uploader_tab2"
        )

        if archivo_explicacion is not None:
            img_exp = Image.open(archivo_explicacion)
            if img_exp.mode != "RGB":
                img_exp = img_exp.convert("RGB")

            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.image(img_exp, caption="Foto Original", use_container_width=False)

            with col_exp2:
                with st.spinner("Generando mapa de activación convolucional..."):
                    img_resize = img_exp.resize((299, 299))
                    img_arr = np.array(img_resize)
                    img_arr = np.expand_dims(img_arr, axis=0)
                    
                    try:
                        mapa = generar_mapa_calor(modelo_custom, img_arr)
                        imagen_superpuesta = superponer_mapa(img_exp, mapa)
                        st.image(imagen_superpuesta, caption="Mapa de Atención (Rojo = Alta Importancia)", use_container_width=False)
                        st.caption("Las regiones resaltadas en colores cálidos muestran los patrones analizados por la red.")
                    except Exception as e:
                        st.warning(f"No se pudo generar el mapa térmico: {e}")

        st.markdown("---")
        st.subheader("3. Matriz de Confusión y Reporte Global")
        if st.button("📈 Evaluar Dataset Completo (datos/test)"):
            dir_test = "datos/test"
            if not os.path.exists(dir_test):
                st.error("No se encontró la carpeta `datos/test`.")
            else:
                with st.spinner("Evaluando conjunto de prueba..."):
                    test_ds = tf.keras.utils.image_dataset_from_directory(
                        directory=dir_test,
                        labels="inferred",
                        label_mode="binary",
                        image_size=(299, 299),
                        batch_size=32,
                        shuffle=False
                    )
                    
                    nombres_clases = test_ds.class_names
                    etiquetas_reales = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
                    
                    predicciones_raw = modelo_custom.predict(test_ds)
                    predicciones_clases = (predicciones_raw >= 0.5).astype(int).flatten()
                    
                    evaluacion = modelo_custom.evaluate(test_ds, verbose=0)
                    loss_test = evaluacion[0]
                    accuracy_test = evaluacion[1]
                    
                    m1, m2 = st.columns(2)
                    m1.metric(label="Precisión General (Accuracy)", value=f"{accuracy_test * 100:.2f}%")
                    m2.metric(label="Pérdida (Loss)", value=f"{loss_test:.4f}")
                    
                    cm = confusion_matrix(etiquetas_reales, predicciones_clases)
                    fig, ax = plt.subplots(figsize=(5, 3.5))
                    sns.heatmap(
                        cm, 
                        annot=True, 
                        fmt="d", 
                        cmap="Blues", 
                        xticklabels=nombres_clases, 
                        yticklabels=nombres_clases,
                        cbar=False,
                        ax=ax
                    )
                    ax.set_title(f"Matriz de Confusión\nAccuracy: {accuracy_test * 100:.2f}%", fontsize=11, fontweight='bold')
                    ax.set_xlabel("Predicción de la IA", fontsize=9)
                    ax.set_ylabel("Clase Real del Test", fontsize=9)
                    plt.tight_layout()
                    
                    c_g1, c_g2 = st.columns([1, 1])
                    with c_g1:
                        st.pyplot(fig)
                    with c_g2:
                        st.write("**Reporte de Clasificación:**")
                        reporte_dict = classification_report(etiquetas_reales, predicciones_clases, target_names=nombres_clases, output_dict=True)
                        st.json(reporte_dict)