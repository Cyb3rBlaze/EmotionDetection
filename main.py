import tensorflow as tf
from tqdm import tqdm

def create_model():
	input_layer = tf.keras.layers.Input((48, 48, 1))
	intermediate_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_1')(input_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_1')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_2')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_2')(intermediate_layer)
	intermediate_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dropout(0.4, name='dropout_1')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(5,5), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_3')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_3')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_4')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_4')(intermediate_layer)	
	intermediate_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dropout(0.4, name='dropout_2')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_5')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_5')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_6')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_6')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dropout(0.4, name='dropout_3')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_7')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_7')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='elu', padding='same', kernel_initializer='he_normal', name='conv2d_8')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_8')(intermediate_layer)
	intermediate_layer = tf.keras.layers.MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dropout(0.5, name='dropout_4')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Flatten(name='flatten')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dense(128, activation='elu', kernel_initializer='he_normal', name='dense_1')(intermediate_layer)
	intermediate_layer = tf.keras.layers.BatchNormalization(name='batchnorm_9')(intermediate_layer)
	intermediate_layer = tf.keras.layers.Dropout(0.6, name='dropout_5')(intermediate_layer)
	output_layer = tf.keras.layers.Dense(6, activation='softmax', name='output')(intermediate_layer)
	

	return tf.keras.Model(inputs=input_layer, outputs=output_layer)

def create_data_iterator(directory):
	return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True, width_shift_range=[-5,5], height_shift_range=[-5,5]).flow_from_directory(directory, color_mode="grayscale", target_size=(48, 48), batch_size=32)

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
		if epoch % 5 == 0:
			model.save('saved_model/model')

model = create_model()
model.summary()

dataset_train = create_data_iterator("./data/train")
dataset_test = create_data_iterator("./data/test")

train_model(model, dataset_train, dataset_test, 200)
