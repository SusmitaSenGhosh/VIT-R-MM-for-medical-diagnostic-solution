import cv2
import numpy as np
import tensorflow as tf
import layers
import matplotlib.cm as cm

def attention_map(model, image,alpha,gridSize):

    # size = model.input_shape[1]
    grid_size = gridSize#int(np.sqrt(model.layers[-13].output_shape[0][-2] - 1))
    X = image#vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]  # type: ignore
    
    outputs = [
        l.output[1] for l in model.layers if (isinstance(l, layers.TransformerBlockReverseMlpMixerHybrid) or isinstance(l, layers.TransformerBlockReverseMlpMixerVit) or isinstance(l, layers.TransformerBlock) or isinstance(l, layers.TransformerBlockReverseMlpMixer_aux))
    ]

    weights = np.array(
        tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    )
    num_layers = weights.shape[0]
    num_heads = weights.shape[2]
    reshaped = weights.reshape(num_layers, num_heads, grid_size ** 2 + 1, grid_size ** 2 + 1)


    reshaped = reshaped.mean(axis=1)

    reshaped = reshaped + np.eye(reshaped.shape[1])
    reshaped = reshaped / reshaped.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v = reshaped[-1]
    for n in range(1, len(reshaped)):
        v = np.matmul(v, reshaped[-1 - n])

    # Attention from the output token to the input space.
    mask = v[0, 1:].reshape(grid_size, grid_size)
    mask = cv2.resize(mask / mask.max(), (image.shape[1], image.shape[2]))
    heatmap = np.uint8(255 * mask)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    image = image
    superimposed_img = jet_heatmap* alpha + np.squeeze(image)
    superimposed_img = superimposed_img/superimposed_img.max()
    return jet_heatmap,superimposed_img
