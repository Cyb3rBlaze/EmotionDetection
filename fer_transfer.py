import tensorflow as tf
from tqdm import tqdm

def create_model():
	model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=[197, 197, 3], pooling='avg', classes=7)
	for i in range(170):
		model.layers[i].trainable = False
	
	head = tf.keras.layers.Flatten()(model.output)
	head = tf.keras.layers.Dropout(0.5)(head)
	head = tf.keras.layers.Dense(4096, activation='relu')(head)
	head = tf.keras.layers.Dropout(0.5)(head)
	head = tf.keras.layers.Dense(1024, activation='relu')(head)
	head = tf.keras.layers.Dropout(0.5)(head)

	output_layer = tf.keras.layers.Dense(7, activation="softmax")(head)
	return tf.keras.Model(inputs=model.input, outputs=output_layer)

def create_data_iterator(directory):
        return tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0, horizontal_flip=True).flow_from_directory(directory, classes=['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise'], color_mode="rgb", target_size=(197, 197), batch_size=32)

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

