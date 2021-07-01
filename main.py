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
	output_layer = tf.keras.layers.Dense(7, activation='softmax', name='output')(intermediate_layer)
	

	return tf.keras.Model(inputs=input_layer, outputs=output_layer)

def create_data_iterator(directory):
	return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True).flow_from_directory(directory, classes=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'], color_mode="grayscale", target_size=(48, 48), batch_size=32)

def get_loss_object():
    return tf.keras.losses.CategoricalCrossentropy()

def get_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.001)

model = create_model()
model.summary()
model.compile(optimizer=get_optimizer(), loss=get_loss_object(), metrics=['accuracy'])

dataset_train = create_data_iterator("./fer_data/train_set")
dataset_test = create_data_iterator("./fer_data/test_set")

weight_for_0_angry = (1/3995) * (28709/7.0)
weight_for_1_disgust = (1/436) * (28709/7.0)
weight_for_2_fear = (1/4097) * (28709/7.0)
weight_for_3_happy = (1/7215) * (28709/7.0)
weight_for_4_neutral = (1/4965) * (28709/7.0)
weight_for_5_sad = (1/4830) * (28709/7.0)
weight_for_6_surprise = (1/3171) * (28709/7.0)

class_weight = {0: weight_for_0_angry, 1: weight_for_1_disgust, 2: weight_for_2_fear, 3: weight_for_3_happy, 4: weight_for_4_neutral, 5: weight_for_5_sad, 6: weight_for_6_surprise}

model.fit(dataset_train, validation_data=dataset_test, epochs=100, class_weight=class_weight)
