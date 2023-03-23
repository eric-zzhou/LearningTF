import numpy as np
import tensorflow as tf
import cv2

interpreter = tf.lite.Interpreter(model_path="detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
for part in output_details:
    print(part)

cam_port = 0
cam = cv2.VideoCapture(cam_port)

while True:
    # reading the input using the camera
    result, image = cam.read()
    image = image[90:390, 90:390]
    interpreter.set_tensor(input_details[0]['index'], [image])
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    print("\n\n")
    for part in output_details:
        print(part)
    input("test: ")

# # Test model on random input data.
# interpreter.set_tensor(input_details[0]['index'], [new_img])
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)
#
# interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)
