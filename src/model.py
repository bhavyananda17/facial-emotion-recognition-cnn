"""The CNN from the paper, as one function.

Architecture (identical in all three official notebooks):
    4 blocks with 64, 128, 256, 512 filters. Each block is
        [conv 3x3 'same' -> BatchNorm -> ReLU] x 2 -> MaxPool 2x2 -> Dropout 0.25
    then Flatten -> Dense 512 ReLU -> BatchNorm -> Dropout 0.5 -> Dense 7 softmax.
Only two things change between datasets: the conv layer type (Conv2D for
CK+, SeparableConv2D for FER2013 and RAF-DB) and the input size.
"""

import keras
from keras import layers


def _conv(conv_type, filters, kernel_size, init):
    """One convolution layer. 'he_normal' is used for every kernel, as in the notebooks."""
    if conv_type == "conv2d":
        return layers.Conv2D(filters, kernel_size, padding="same", kernel_initializer=init)
    if conv_type == "separable":
        # A depthwise 3x3 conv followed by a 1x1 pointwise conv: fewer parameters.
        return layers.SeparableConv2D(
            filters, kernel_size, padding="same", depthwise_initializer=init, pointwise_initializer=init
        )
    raise ValueError(f"conv_type must be 'conv2d' or 'separable', got {conv_type!r}")


def build_model(conv_type, input_size, num_classes=7, filters=(64, 128, 256, 512), kernel_size=3,
                kernel_initializer="he_normal", block_dropout=0.25, dense_units=512, dense_dropout=0.5,
                learning_rate=0.001):
    """Build and compile the model.

    Args:
        conv_type:  'conv2d' or 'separable'.
        input_size: (height, width) of the grayscale input image.
    The defaults are the notebooks' values; train.py passes them from the YAML config.
    """
    model = keras.Sequential(name=f"fer_{conv_type}_{input_size[0]}x{input_size[1]}")
    model.add(keras.Input(shape=(input_size[0], input_size[1], 1)))  # 1 channel = grayscale

    for f in filters:
        for _ in range(2):  # two conv layers per block
            model.add(_conv(conv_type, f, kernel_size, kernel_initializer))
            model.add(layers.BatchNormalization())
            model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(block_dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu", kernel_initializer=kernel_initializer))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(dense_dropout))
    model.add(layers.Dense(num_classes, activation="softmax"))

    # The notebooks pass Adam(learning_rate=0.001) to compile(). Their
    # ExponentialDecay schedule is defined but never used, so it is not used here either.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_from_config(cfg):
    """Build the model for one dataset from its YAML config."""
    m = cfg["model"]
    return build_model(
        conv_type=m["conv_type"],
        input_size=tuple(cfg["image"]["size"]),
        num_classes=cfg["num_classes"],
        filters=tuple(m["filters"]),
        kernel_size=m["kernel_size"],
        kernel_initializer=m["kernel_initializer"],
        block_dropout=m["block_dropout"],
        dense_units=m["dense_units"],
        dense_dropout=m["dense_dropout"],
        learning_rate=cfg["training"]["learning_rate"],
    )


if __name__ == "__main__":
    # Architecture check: parameter counts must equal the notebooks' model.summary().
    from src.utils import load_config

    for name in ("ck", "fer2013", "rafdb"):
        cfg = load_config(f"configs/{name}.yaml")
        total = build_from_config(cfg).count_params()
        expected = cfg["paper"]["total_params"]
        print(f"{name:8s} params={total:>10,}  notebook={expected:>10,}  {'OK' if total == expected else 'MISMATCH'}")
