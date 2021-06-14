import tensorflow as tf
from tqdm import tqdm

def create_model():
	model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=[96, 96, 3], pooling=None, classes=6)
	for i in range(len(model.layers)):
		model.layers[i].trainable = False
	output_layer = tf.keras.layers.Flatten()(model.output)
	output_layer = tf.keras.layers.Dense(6, activation="softmax")(output_layer)
	return tf.keras.Model(inputs=model.input, outputs=output_layer)

def create_data_iterator(directory):
	return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0).flow_from_directory(directory, target_size=(96, 96), batch_size=32)

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

def train_model(model, dataset_train, dataset_test, epochs):
	optimizer = get_optimizer()
	train_accuracy = tf.keras.metrics.CategoricalAccuracy()
	test_accuracy = tf.keras.metrics.CategoricalAccuracy()	

	for epoch in range(epochs):
		print("Epoch: " + str(epoch))
		epoch_loss_avg = tf.keras.metrics.Mean()
		count = 0
		for batch in tqdm(dataset_train):
			count += 1 
			loss_value, grads = compute_model_gradients(model, batch[0], batch[1])
			epoch_loss_avg.update_state(loss_value)
			train_accuracy.update_state(batch[1], model.predict(batch[0]).reshape((int(batch[1].size/6), 6)))
			optimizer.apply_gradients(zip(grads, model.trainable_variables))
			if count % 200 == 0:
				print("Train Accuracy: " + str(train_accuracy.result().numpy()))
			if count > len(dataset_train):
				break
		count = 0
		for batch in tqdm(dataset_test):
			count += 1
			test_accuracy.update_state(batch[1], model.predict(batch[0]).reshape((int(batch[1].size/6), 6)))
			if count > len(dataset_test):
				print("Test Accuracy: " + str(test_accuracy.result().numpy()))
				break
		
		train_accuracy.reset_states()
		test_accuracy.reset_states()

model = create_model()
model.summary()

dataset_train = create_data_iterator("./data/train")
dataset_test = create_data_iterator("./data/test")

train_model(model, dataset_train, dataset_test, 50)
