from tensorflow.keras.applications import ResNet50


base = ResNet50(include_top=False)
for layer in base.layers:
    print(layer.name)