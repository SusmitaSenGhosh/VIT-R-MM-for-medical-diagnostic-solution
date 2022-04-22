import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from functools import partial
import tensorflow_addons as tfa
import ViT,ViTR,ViTMM, ViTRMM, ResNetViT,ResNetViTRMM, AuxResNetViTRMM

def load_model(model_name,n,weights):
	if model_name == 'ViT':
        
		base_model = ViT.vit_b16()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())
		
	elif model_name == 'ViT-R':
        
		base_model = ViTR.vit_b16()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())

	elif model_name == 'ViT-MM':
        
		base_model = ViTMM.vit_b16()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())
				
	elif model_name == 'ViT-R-MM':
        
		base_model = ViTRMM.vit_b16()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())
         
	elif model_name == 'ResNet50':
		base_model = tf.keras.applications.ResNet50(include_top=True, input_tensor=None, input_shape=(224,224,3), pooling='avg')
		base_model.trainable = True

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
			],
			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())
		
	elif model_name == 'ResNet-ViT':
		base_model = ResNetViT.my_model()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())
        
						
										
	elif model_name == 'ResNet-ViT-R-MM':
		base_model = ResNetViTRMM.my_model()

		model = tf.keras.Sequential([
				base_model,
				tf.keras.layers.Flatten(),
				tf.keras.layers.BatchNormalization(),        
				tf.keras.layers.Dense(128, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(32, activation = tfa.activations.gelu),
				tf.keras.layers.BatchNormalization(),
				tf.keras.layers.Dense(n, 'softmax')
 			],
 			name = model_name)

		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam())

		
        	
	elif model_name == 'Aux-ResNet-ViT-R-MM':
        
		def weighted_mse(y_true, y_pred, weights):
 			y_pred = tf.convert_to_tensor(y_pred)
 			y_true = tf.cast(y_true, y_pred.dtype)
 			return tf.reduce_mean(weights*(tf.square(y_pred - y_true)))


		loss0 = partial(weighted_mse, weights=weights)
		loss1 = partial(weighted_mse, weights=weights)
		loss2 = partial(weighted_mse, weights=weights)

		loss0.__name__ = 'loss0'
		loss1.__name__ = 'loss1'
		loss2.__name__ = 'loss2'

		model = AuxResNetViTRMM.my_model(classes = n)

		model.summary()

		model.compile(loss={'output0': loss0,'output1': loss1, 'output2': loss2}, loss_weights=[.3, .3, .4],optimizer=Adam())

	return model



