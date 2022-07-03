import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import InceptionV3

model = load_model(
    "model_inceptionv3_ckpt.h5"
)

class_names = [
    "car_dashboard",
    "car_internal",
    "car_outer",
    "chassis_engine_number",
    "document",
    "other",
    "speedometer_odometer",
]


def function(image):
    image = image.reshape((-1, 299, 299, 3))
    image = tf.keras.applications.inception_v3.preprocess_input(image)
    prediction = model.predict(image).flatten()
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}


image = gr.inputs.Image(shape=(299, 299))
label = gr.outputs.Label(num_top_classes=len(class_names))


gr.Interface(
    fn=function,
    inputs=image,
    outputs=label,
    capture_session=True,
    title="Caso 1: Clasificación de partes de un coche",
    description="Clasificación de partes de un coche",
).launch(share=True)

