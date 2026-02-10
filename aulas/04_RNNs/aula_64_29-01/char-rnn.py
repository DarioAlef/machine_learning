import os
# Desabilitar GPU temporariamente devido a erro de configuração CUDA (libdevice missing)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import time
from datasets import load_dataset

# 1. Carregar o dataset
print("Carregando o dataset...")
dataset = load_dataset("tiagoblima/bible-ptbr-gun-gub-aligned")

# Vamos usar a tradução 'Scripture_nvi' (Nova Versão Internacional - Português)
# Se quiser outra, mude para 'Scripture_gub' (Guajajara) ou 'Scripture_gun' (Mbyá Guaraní)
TEXT_COLUMN = 'Scripture_nvi'

# Extrair todo o texto e juntar
print(f"Extraindo texto da coluna '{TEXT_COLUMN}'...")
text_list = [t for t in dataset['train'][TEXT_COLUMN] if t is not None]
text = " ".join(text_list)

print(f"Tamanho do texto: {len(text)} caracteres")
print(f"Exemplo de texto: {text[:200]}")

# 2. Processar o vocabulário
vocab = sorted(set(text))
print(f"{len(vocab)} caracteres únicos")

# Mapeamento de char para index e vice-versa
example_texts = ['abcdefg', 'xyz']
chars = tf.strings.unicode_split(example_texts, input_encoding='UTF-8')
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
chars_from_ids = tf.keras.layers.StringLookup(
    vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)

# 3. Criar dataset de treinamento
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

dataset_tf = sequences.map(split_input_target)

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset_tf = (
    dataset_tf
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# 4. Construir o Modelo
vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = 256
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__()
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = self.embedding(inputs, training=training)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    return x

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# 5. Treinamento
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss)

# Configurar checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 1 # Reduzido para demonstração rápida. Aumente para 10 ou mais para melhores resultados.
print(f"Iniciando treinamento por {EPOCHS} épocas...")
history = model.fit(dataset_tf, epochs=EPOCHS, callbacks=[checkpoint_callback])

# 6. Geração de Texto
class OneStep(tf.keras.Model):
  def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.chars_from_ids = chars_from_ids
    self.ids_from_chars = ids_from_chars

    # Criar máscara para impedir "[UNK]" de ser gerado.
    skip_ids = self.ids_from_chars(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        dense_shape=[len(ids_from_chars.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Converter strings para IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.ids_from_chars(input_chars).to_tensor()

    # Rodar o modelo.
    # predicted_logits.shape: [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Usar apenas a última predição.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Aplicar a máscara de predição.
    predicted_logits = predicted_logits + self.prediction_mask

    # Amostrar os logits de saída para gerar IDs de token.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Converter de IDs para caracteres.
    predicted_chars = self.chars_from_ids(predicted_ids)

    # Retornar os caracteres e o estado do modelo.
    return predicted_chars, states

one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['No princípio'])
result = [next_char]

print("Gerando texto...")
for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
