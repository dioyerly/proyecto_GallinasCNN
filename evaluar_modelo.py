import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# 1. Configuración de parámetros y rutas
ANCHO_ALTO_IMAGEN = (299, 299)
TAMANIO_LOTE = 32
RUTA_MODELO = "modelo_procesado/depredadores_modelo.keras"
DIR_TEST = "datos/test"

print("1. Cargando el modelo entrenado...")
if not os.path.exists(RUTA_MODELO):
    raise FileNotFoundError(f"No se encontró el modelo en {RUTA_MODELO}. Asegúrate de haberlo entrenado.")

modelo = tf.keras.models.load_model(RUTA_MODELO)

# 2. Cargar el dataset de prueba (sin mezclar para mantener el orden exacto)
print("\n2. Cargando imágenes de la carpeta datos/test...")
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory=DIR_TEST,
    labels="inferred",
    label_mode="binary",
    image_size=ANCHO_ALTO_IMAGEN,
    batch_size=TAMANIO_LOTE,
    shuffle=False  # Importante: Mantener en False para alinear etiquetas reales con las predicciones
)

nombres_clases = test_ds.class_names
print(f"Clases identificadas: {nombres_clases}")

# 3. Obtener etiquetas reales y calcular predicciones
print("\n3. Realizando predicciones sobre el set de prueba...")
etiquetas_reales = np.concatenate([y.numpy() for x, y in test_ds], axis=0)

predicciones_raw = modelo.predict(test_ds)
# Convertir las probabilidades sigmoidales (0 a 1) en clases binarias (0 o 1) mediante umbral 0.5
predicciones_clases = (predicciones_raw >= 0.5).astype(int).flatten()

# 4. Cálculo de métricas
evaluacion = modelo.evaluate(test_ds, verbose=0)
loss_test = evaluacion[0]
accuracy_test = evaluacion[1]

print("\n" + "="*40)
print(f" RESULTADOS EN EL DATASET DE PRUEBA ")
print("="*40)
print(f"Pérdida (Loss):     {loss_test:.4f}")
print(f"Precisión (Accuracy): {accuracy_test * 100:.2f}%")
print("="*40 + "\n")

print("Reporte detallado de clasificación:")
print(classification_report(etiquetas_reales, predicciones_clases, target_names=nombres_clases))

# 5. Graficar la Matriz de Confusión
cm = confusion_matrix(etiquetas_reales, predicciones_clases)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Blues", 
    xticklabels=nombres_clases, 
    yticklabels=nombres_clases,
    cbar=False
)
plt.title(f"Matriz de Confusión\nAccuracy General: {accuracy_test * 100:.2f}%", fontsize=14, fontweight='bold')
plt.xlabel("Predicción del Modelo", fontsize=12)
plt.ylabel("Clase Real (Terreno)", fontsize=12)
plt.tight_layout()

# Guardar la imagen en el disco para la presentación o informe
os.makedirs("reportes", exist_ok=True)
plt.savefig("reportes/matriz_de_conexion.png", dpi=300)
print("📸 La matriz de confusión se guardó en 'reportes/matriz_de_conexion.png'")
plt.show()