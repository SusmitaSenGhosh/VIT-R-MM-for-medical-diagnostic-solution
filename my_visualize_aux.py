import cv2
import numpy as np
import tensorflow as tf
import layers
import matplotlib.cm as cm

def attention_map(model, image,alpha,gridSize):

    # size = model.input_shape[1]
    grid_size1 = 7 #int(np.sqrt(model.layers[-13].output_shape[0][-2] - 1))
    grid_size2 = 14
    # Prepare the input
    X = image#vit.preprocess_inputs(cv2.resize(image, (size, size)))[np.newaxis, :]  # type: ignore
    
    # Get the attention weights from each transformer.
    outputs = [
        l.output[1] for l in model.layers if (isinstance(l, layers.TransformerBlockReverseMlpMixerHybrid) or isinstance(l, layers.TransformerBlockReverseMlpMixerVit) or isinstance(l, layers.TransformerBlock) or isinstance(l, layers.TransformerBlockReverseMlpMixer_aux))
    ]

    weights = tf.keras.models.Model(inputs=model.inputs, outputs=outputs).predict(X)
    weights1 = np.array([weights[0],weights[3]])
    weights2 = np.array([weights[1],weights[2]])

    num_layers = weights1.shape[0]
    num_heads = weights1.shape[2]
    reshaped1 = weights1.reshape(num_layers, num_heads, grid_size1 ** 2 + 1, grid_size1 ** 2 + 1)
    
    num_layers = weights2.shape[0]
    num_heads = weights2.shape[2]
    reshaped2 = weights2.reshape(num_layers, num_heads, grid_size2 ** 2 + 1, grid_size2 ** 2 + 1)
    # )

    # # # From Appendix D.6 in the paper ...
    # # # Average the attention weights across all heads.
    reshaped1 = reshaped1.mean(axis=1)
    reshaped2 = reshaped2.mean(axis=1)

    # # # From Section 3 in https://arxiv.org/pdf/2005.00928.pdf ...
    # # # To account for residual connections, we add an identity matrix to the
    # # # attention matrix and re-normalize the weights.
    reshaped1 = reshaped1 + np.eye(reshaped1.shape[1])
    reshaped1 = reshaped1 / reshaped1.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    reshaped2 = reshaped2 + np.eye(reshaped2.shape[1])
    reshaped2 = reshaped2 / reshaped2.sum(axis=(1, 2))[:, np.newaxis, np.newaxis]

    # Recursively multiply the weight matrices
    v1 = reshaped1[-1]
    for n in range(1, len(reshaped1)):
        v1 = np.matmul(v1, reshaped1[-1 - n])
    v2 = reshaped2[-1]
    for n in range(1, len(reshaped2)):
        v2 = np.matmul(v2, reshaped2[-1 - n])

    # Attention from the output token to the input space.
    mask1 = v1[0, 1:].reshape(grid_size1, grid_size1)
    mask2 = v2[0, 1:].reshape(grid_size2, grid_size2)

    mask1 = cv2.resize(mask1 / mask1.max(), (image.shape[1], image.shape[2]))
    mask2 = cv2.resize(mask2 / mask2.max(), (image.shape[1], image.shape[2]))
    mask = (mask1+mask2)/2
    mask = mask / mask.max()
    
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    
    heatmap1 = np.uint8(255 * mask1)
    jet_heatmap1 = jet_colors[heatmap1]
    
    heatmap2 = np.uint8(255 * mask2)
    jet_heatmap2 = jet_colors[heatmap2]
    
    heatmap = np.uint8(255 * mask)
    jet_heatmap = jet_colors[heatmap]
    
    #jet_heatmap = jet_heatmap/jet_heatmap.max()

    # Superimpose the heatmap on original image
    # print(jet_heatmap.max(),jet_heatmap.min())
    superimposed_img1 = jet_heatmap1* alpha + np.squeeze(image)
    superimposed_img1 = superimposed_img1/superimposed_img1.max()
    
    superimposed_img2 = jet_heatmap2* alpha + np.squeeze(image)
    superimposed_img2 = superimposed_img2/superimposed_img2.max()
    
    superimposed_img = jet_heatmap* alpha + np.squeeze(image)
    superimposed_img = superimposed_img/superimposed_img.max()
    return jet_heatmap1,jet_heatmap2,jet_heatmap,superimposed_img1,superimposed_img2,superimposed_img
