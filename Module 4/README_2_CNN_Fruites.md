**Project:** 2_CNN_Fruites.ipynb

- **Description:** A simple CNN classifier using a small fruit image dataset (fruits-small). The notebook shows how to preprocess images with tf.keras.preprocessing.image.ImageDataGenerator, build a convolutional neural network, compile and train the model, and run predictions for a single image.

**Contents:**
- **Notebook:** Module 4/2_CNN_Fruites.ipynb
- **Dataset:** fruits-small.zip (extracted into fruits-small/)

**Requirements**
- Python 3.8+ (3.11/3.12/3.13 tested in the workspace)
- pip packages:
  - tensorflow (>= 2.10)
  - numpy
  - pandas
  - pillow
  - scikit-learn (optional for preprocessing during experiments)
  - matplotlib (optional)

Install required packages:

```bash
python -m pip install --upgrade pip
pip install tensorflow numpy pandas pillow matplotlib scikit-learn
```

**Quick Start**
- Open `Module 4/2_CNN_Fruites.ipynb` in Jupyter or VS Code and run the cells in order.
- Ensure `fruits-small.zip` is in the same folder as the notebook and the extraction step runs successfully.

**Important Notes & Common Fixes**
- Input (Image) Size must be consistent:
  - The notebook uses `target_size = (224, 224)` in `ImageDataGenerator.flow_from_directory()` for train/test data. If you predict using a different target size (e.g., 256x256), you will get a Dense input shape mismatch error ("expected axis -1 of input shape to have value 100352, but received input with shape (1, 131072)").
  - Fix it by loading your prediction image with the same target size:
```python
image = tf.keras.preprocessing.image.load_img('some_image.jpg', target_size=(224,224))
image_array = tf.keras.preprocessing.image.img_to_array(image)/255.0  # use rescale 1.0/255.0 if using generators
image_np_array = np.expand_dims(image_array, axis=0)
prediction_probabilities = model.predict(image_np_array)
```
- Or, make the model input-size-agnostic by using a pooling layer instead of Flatten:
```python
# Replace Flatten with a global pooling layer to accept variable input image sizes
model.add(tf.keras.layers.GlobalAveragePooling2D())
# then add Dense layers for the head
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
```

- Callback comparison and ambiguous-array truth-value:
  - The default MyCLRuleMonitor compares log values (`logs['accuracy']` and `logs['val_accuracy']`) directly. If you use a metric that returns an array (e.g., some F1 implementations), Python raises the "truth value of an array is ambiguous" error.
  - Convert logs values to scalars before comparison or implement a safe reduction. Example:
```python
class MyCLRuleMonitor(tf.keras.callbacks.Callback):
    def __init__(self, CL, metric_name='accuracy', reduce='mean', verbose=False):
        super().__init__()
        self.CL = float(CL)
        self.metric_name = metric_name
        self.reduce = reduce
        self.verbose = verbose

    def _to_scalar(self, value):
        # Convert Tensor/numpy arrays to scalar safely
        if value is None:
            return None
        try:
            if isinstance(value, tf.Tensor):
                value = value.numpy()
            arr = np.asarray(value)
            if arr.size == 1:
                return float(arr.item())
            if self.reduce == 'mean':
                return float(arr.mean())
            else:
                return float(arr.mean())
        except Exception:
            return None

    def on_epoch_end(self, epoch, logs=None):
        train_val = self._to_scalar(logs.get(self.metric_name))
        val_val = self._to_scalar(logs.get('val_' + self.metric_name))
        if train_val is None or val_val is None:
            return
        if (val_val > train_val) and (val_val >= self.CL):
            self.model.stop_training = True
```
  - If you use a custom F1 metric, prefer `tf.keras.metrics` or TensorFlow Addons `tfa.metrics.F1Score` that returns a scalar, or ensure you reduce per-batch/multi-class F1 into a scalar for the callback.

**Recommended Changes**
- Use the `GlobalAveragePooling2D` head to allow variable image sizes and make model inference robust.
- Always apply the same rescaling (divide by 255.0) for prediction that you used in the training generators to ensure identical pixel normalization.
- If you want to monitor F1 during training, prefer a scalar F1 metric (e.g. `tfa.metrics.F1Score`) or implement a wrapper that returns a scalar.

**How to use**
1. Extract the dataset:
```python
import shutil
shutil.unpack_archive('fruits-small.zip', 'fruits-small')
```
2. Start Jupyter or run the notebook in VS Code.
3. Train the model with `model.fit` using: `train_image_generator` -> `flow_from_directory()` for train and validation.
4. For prediction, load the image with the same `target_size`, normalize, and pass to `model.predict()`.

**Troubleshooting**
- "ValueError: Input 0 of layer 'dense_x' is incompatible..." — fix by the steps above (match target_size OR use GlobalAveragePooling2D).
- "truth value of an array is ambiguous" — convert logs values into scalar before checking in the callback.
- If using `f1_score` as a keras metric, ensure it returns a scalar for the callback to use (or use `reduce='mean'` in callback).

**Credits**
- Created for the Simplylearn - AIML course Module 4 as part of the CNN examples.

**License**
- No license provided; add one as appropriate for your use.
