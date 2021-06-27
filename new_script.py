import tensorflow as tf
import numpy as np
from os import listdir
from tqdm import tqdm
from PIL import Image

def create_model():
	model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=[256, 256, 3], pooling=None)
	for i in range(len(model.layers)):
		model.layers[i].trainable = False
	output_layer = tf.keras.layers.Flatten()(model.output)
	output_layer = tf.keras.layers.Dense(8, activation="softmax")(output_layer)
	return tf.keras.Model(inputs=model.input, outputs=output_layer)

def get_loss_object():
    return tf.keras.losses.CategoricalCrossentropy()

def model_loss(model, inputs, outputs):
        y_ = model(inputs, training=True)
        loss = get_loss_object()(y_true=outputs, y_pred=y_)

        return loss

def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

def compute_model_gradients(model, inputs, outputs):
        with tf.GradientTape() as tape:
                loss_value = model_loss(model, inputs, outputs)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

def load_batch(directory, starting_index, batch_size, dims, num_labels):
	batch_x = np.zeros((batch_size, dims[0], dims[1], 3))
	batch_y = []
	files = []
	for i in listdir(directory):
		count = 0
		full_count = 0
		for j in listdir(directory + "/" + i):
			if count >= batch_size:
				break
			if i == "annotations" and j[len(j)-7:len(j)-4] == "exp" and full_count > starting_index:
				files += [j[:len(j)-8]]
				empty_one_hot = np.zeros((num_labels))
				empty_one_hot[int(np.load(directory + "/" + i + "/" + j))] = 1
				batch_y += [empty_one_hot]
				count += 1
			elif i == "images" and j[:len(j)-4] in files:
				batch_x[files.index(j[:len(j)-4])] = np.asarray(Image.open(directory + "/" + i + "/" + j).resize(dims))
				count += 1
			full_count += 1
	return np.array(batch_x), np.array(batch_y)

def train(model, directory, batch_size, dims, num_labels, epochs, data_bound):
	optimizer = get_optimizer()
	train_accuracy = tf.keras.metrics.CategoricalAccuracy()
	test_accuracy = tf.keras.metrics.CategoricalAccuracy()

	for epoch in range(epochs):
		print("Epoch: " + str(epoch))
		index = 0
		epoch_loss_avg = tf.keras.metrics.Mean()
		for j in tqdm(range(data_bound)):
			batch_x, batch_y = load_batch(directory, index, batch_size, dims, num_labels)
			loss_value, grads = compute_model_gradients(model, batch_x, batch_y)
			epoch_loss_avg.update_state(loss_value)
			train_accuracy.update_state(batch_y, model.predict(batch_x).reshape((int(batch_y.size/8), 8)))
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			
			if j % 10 == 0:
				print("Train Accuracy: " + str(train_accuracy.result().numpy()))

			index += batch_size
		train_accuracy.reset_states()

model = create_model()

train(model, "data/train_set", 64, (256, 256), 8, 10, 100)
