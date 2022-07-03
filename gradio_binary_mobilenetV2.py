import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3, MobileNetV2

model = load_model(
    "cropsmodel_mobilenetV2_ckpt.h5"
)

class_names = ["damage", "no_damage"]


def function(image):
    image = image.reshape((-1, 224, 224, 3))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    prediction = model.predict(image).flatten()
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}


image = gr.inputs.Image(shape=(224, 224))
label = gr.outputs.Label(num_top_classes=len(class_names))


gr.Interface(
    fn=function,
    inputs=image,
    outputs=label,
    capture_session=True,
    title="Caso 2: Clasificación binaria de daños (modelo con MobilenetV2)",
    description="A partir de una imagen de un coche, la app indicará si hay o no daños.",
).launch(share=True)
