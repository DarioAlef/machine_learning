import tensorflow as tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Carregar dados
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Separação Treino/Validação e Achatamento (Flatten) para DNN
# CIFAR são imagens 32x32x3. Para DNN, viram vetores de 3072.
X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]

# PADRONIZAÇÃO (Crucial para SELU, bom para todas as outras)
mean = X_train.mean(axis=0, keepdims=True)
std = X_train.std(axis=0, keepdims=True)

X_train_scaled = (X_train - mean) / std
X_valid_scaled = (X_valid - mean) / std
X_test_scaled = (X_test - mean) / std

# Achatar as imagens para entrada na Dense Layer
X_train_flat = X_train_scaled.reshape(-1, 32*32*3)
X_valid_flat = X_valid_scaled.reshape(-1, 32*32*3)
X_test_flat = X_test_scaled.reshape(-1, 32*32*3)



def build_model(n_hidden=20, n_neurons=100, learning_rate=3e-3, input_shape=[3072]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal"))
        
    model.add(keras.layers.Dense(10, activation="softmax")) # 10 classes
    
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# Callbacks
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Treinamento (Exemplo com LR fixo, mas você deve testar vários como 1e-5, 3e-5, 1e-4...)
model_a = build_model(learning_rate=5e-5) 
history_a = model_a.fit(X_train_flat, y_train, epochs=100, 
                        validation_data=(X_valid_flat, y_valid),
                        callbacks=[early_stopping])



def build_model_bn(learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[3072]))
    
    for layer in range(20):
        # A ordem canônica teórica: Dense (sem bias) -> BN -> Activation
        model.add(keras.layers.Dense(100, use_bias=False, kernel_initializer="he_normal"))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("elu"))
        
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# Com BN, podemos tentar um LR maior, ex: 1e-4 ou 1e-3
model_bn = build_model_bn(learning_rate=1e-3)
history_bn = model_bn.fit(X_train_flat, y_train, epochs=100, 
                          validation_data=(X_valid_flat, y_valid),
                          callbacks=[early_stopping])



def build_model_selu(learning_rate=3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[3072]))
    
    for layer in range(20):
        # O par mágico: activation="selu" e kernel_initializer="lecun_normal"
        model.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
        
    model.add(keras.layers.Dense(10, activation="softmax"))
    
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

model_selu = build_model_selu(learning_rate=1e-4) # SELU geralmente prefere LRs menores que BN
history_selu = model_selu.fit(X_train_flat, y_train, epochs=100, 
                              validation_data=(X_valid_flat, y_valid),
                              callbacks=[early_stopping])



# 1. Adicionando AlphaDropout
model_drop = keras.models.Sequential()
model_drop.add(keras.layers.InputLayer(input_shape=[3072]))
for layer in range(20):
    model_drop.add(keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal"))
    # AlphaDropout logo após a camada SELU
    if layer % 5 == 0: # Exemplo: Adicionar em algumas camadas para não matar a rede
         model_drop.add(keras.layers.AlphaDropout(rate=0.1))

model_drop.add(keras.layers.Dense(10, activation="softmax"))
# ... compile e treine ...

# 2. MC Dropout (Sem re-treinar!)
# Basicamente, forçamos training=True na inferência
class MCDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

# Ou, de forma mais simples e "hacky" para predição:
# Fazemos 100 predições com dropout ligado
y_probas = np.stack([model_drop(X_test_flat, training=True) for _ in range(100)])
y_proba = y_probas.mean(axis=0) # Média das probabilidades
y_pred = np.argmax(y_proba, axis=1) # Classe final



class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
        
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
        
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        keras.backend.set_value(self.model.optimizer.lr, rate)

# Uso:
# batch_size = 128
# n_epochs = 15
# onecycle = OneCycleScheduler(len(X_train) // batch_size * n_epochs, max_rate=0.05)
# history = model.fit(..., callbacks=[onecycle])