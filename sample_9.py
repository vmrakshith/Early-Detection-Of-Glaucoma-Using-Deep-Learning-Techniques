import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import numpy as np
import os
import pathlib

# Define constants
img_height = 224  # Changed to 224 for MobileNetV2 compatibility
img_width = 224   # Changed to 224 for MobileNetV2 compatibility
batch_size = 32

# Define the data augmentation function
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Define the dataset creation function
def create_dataset(image_dir, batch_size):
    dataset = tf.data.Dataset.list_files(str(image_dir/'*/*'))
    class_names = np.array(sorted([item.name for item in image_dir.glob('*') if item.name != "LICENSE.txt"]))
    label_to_index = dict((name, index) for index, name in enumerate(class_names))

    # Create a lookup table
    keys = tf.constant(list(label_to_index.keys()))
    values = tf.constant(list(label_to_index.values()))
    lookup_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1)

    dataset = dataset.map(lambda x: (x, tf.strings.split(x, os.path.sep)[-2]))
    dataset = dataset.map(lambda x, y: (x, lookup_table.lookup(y)))  # Use lookup table for labels
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset, class_names  # Return class_names as well

# Create the datasets
train_dir = pathlib.Path(r'C:\Users\KIRAN KUMAR\Glaucoma-Detection-using-CNN\split\images\train')
val_dir = pathlib.Path(r'C:\Users\KIRAN KUMAR\Glaucoma-Detection-using-CNN\split\images\val')
test_dir = pathlib.Path(r'C:\Users\KIRAN KUMAR\Glaucoma-Detection-using-CNN\split\images\test')

train_ds, class_names = create_dataset(train_dir, batch_size)
print("Class names:", class_names)
val_ds, _ = create_dataset(val_dir, batch_size)
test_ds, _ = create_dataset(test_dir, batch_size)

# Define the data augmentation function for training
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

# Apply data augmentation to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), 
                         num_parallel_calls=tf.data.AUTOTUNE)

# Define the model architecture
base_model = MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Unfreeze the last 20 layers
    layer.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('model_best.keras', save_best_only=True)  # Changed to .keras
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=150,
    callbacks=callbacks
)

# Save the model
model_save_path = 'C:/Users/KIRAN KUMAR/Glaucoma-Detection-using-CNN/split/saved_model/my_model4.h5'
model.save(model_save_path)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_ds)
print(f'Test accuracy: {test_acc:.2f}')

# Load the saved model (optional)
loaded_model = tf.keras.models.load_model(model_save_path)

# Evaluate the loaded model (optional)
loaded_test_loss, loaded_test_acc = loaded_model.evaluate(test_ds)
print(f'Loaded model test accuracy: {loaded_test_acc:.2f}')
