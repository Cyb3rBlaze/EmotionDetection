import tensorflow as tf
import cv2

def run_inference(model, sample):
	return model.predict((sample))

def preprocess(image_path, save_path):
	img = cv2.imread(image_path)
	img = cv2.resize(img, (48, 48))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(save_path, img)
	return img

image = preprocess("test.jpg", "output.jpg")
image = image.reshape((1, 48, 48, 1))
image = image/255.

model = tf.keras.models.load_model("saved_model/model3_final")
model.summary()

output = run_inference(model, image)
classes = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}

max_val = max(output[0])
max_val_index = list(output[0]).index(max_val)

true_class = classes[max_val_index]

print(true_class)


