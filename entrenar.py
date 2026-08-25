import os
import tensorflow as tf
import keras
from keras import layers
from keras.applications import Xception

# 1. Configuración de parámetros básicos
ANCHO_ALTO_IMAGEN = (299, 299)  # El modelo Xception requiere este tamaño exacto de imagen
TAMANIO_LOTE = 32
EPOCAS = 5  # Cuántas veces el modelo repasará todo tu set de fotos

print("1. Cargando y procesando las imágenes de las carpetas...")

# Cargamos las imágenes de entrenamiento
train_ds = keras.utils.image_dataset_from_directory(
    directory="datos/train",
    labels="inferred",
    label_mode="binary",  # Clasificación Binaria: Amenaza (1) o No Amenaza (0)
    image_size=ANCHO_ALTO_IMAGEN,
    batch_size=TAMANIO_LOTE
)

# Cargamos las imágenes de pruebas
test_ds = keras.utils.image_dataset_from_directory(
    directory="datos/test",
    labels="inferred",
    label_mode="binary",
    image_size=ANCHO_ALTO_IMAGEN,
    batch_size=TAMANIO_LOTE
)

# Guardamos los nombres de las categorías ANTES de optimizar el conjunto de datos
nombres_categorias = train_ds.class_names
print(f"¡Imágenes cargadas! Categorías detectadas: {nombres_categorias}")

# Optimizamos el flujo de datos para que tu computadora procese rápido
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 2. Transfer Learning: Cargamos el modelo Xception de Google
print("\n2. Cargando modelo base Xception pre-entrenado...")
base_model = Xception(
    weights="imagenet",       # Carga el conocimiento previo de millones de imágenes
    input_shape=(299, 299, 3), # Entrada en color (RGB) de 299x299 píxeles
    include_top=False         # Quitamos la capa de salida genérica para poner la nuestra
)

# Congelamos el conocimiento del modelo base para no destruirlo
base_model.trainable = False

# 3. Construcción de nuestra red neuronal personalizada
print("\n3. Ensamblando la red neuronal personalizada...")
inputs = keras.Input(shape=(299, 299, 3))
# Normalizamos los colores de las fotos (valores de los píxeles entre -1 y 1)
x = layers.Rescaling(scale=1./127.5, offset=-1.)(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x) # Reduce la complejidad de la información
x = layers.Dense(128, activation="relu")(x) # Capa oculta para aprender patrones intermedios
outputs = layers.Dense(1, activation="sigmoid")(x) # Capa final: da un resultado entre 0 y 1

modelo_final = keras.Model(inputs, outputs)

# 4. Compilación (Configuramos el optimizador, la pérdida y la métrica de precisión)
modelo_final.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy", # Ideal para clasificar entre dos opciones
    metrics=["accuracy"]        # Mediremos el porcentaje de aciertos del modelo
)

# 5. ¡A ENTRENAR!
print("\n4. Iniciando el entrenamiento de la Inteligencia Artificial...")
historia = modelo_final.fit(
    train_ds,
    epochs=EPOCAS,
    validation_data=test_ds
)

# 6. Guardar el "cerebro" entrenado en tu disco local
os.makedirs("modelo_procesado", exist_ok=True)
modelo_final.save("modelo_procesado/depredadores_modelo.keras")
print("\n🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO! El modelo se guardó en 'modelo_procesado/depredadores_modelo.keras'")