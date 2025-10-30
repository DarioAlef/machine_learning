import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter



# ==================================
# 1. Configuração e Hiperparâmetros
# ==================================



# Definir o device(GPU se disponível, senão CPU)
if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
# else:
#     device = "cpu"
print(f"Usando dispositivo: {device}")

# Hiperparâmetros
LEARNING_RATE = 0.001
BATCH_SIZE = 64
N_EPOCHS = 20  # 20 épocas são geralmente suficientes para >98%
INPUT_SIZE = 28 * 28  # Imagens MNIST são 28x28
HIDDEN_SIZE_1 = 300   # Neurônios na primeira camada oculta
HIDDEN_SIZE_2 = 100   # Neurônios na segunda camada oculta
OUTPUT_SIZE = 10      # 10 classes (dígitos 0-9)

# Para Checkpoints e TensorBoard
CHECKPOINT_PATH = "./atividades/mlp_mnist/best_model_checkpoint.pth"
TENSORBOARD_LOG_DIR = "./atividades/mlp_mnist/runs/mnist_mlp"



# =========================================
# --- 2. Preparação dos Dados (MNIST) ---
# =========================================



# Transformações:
# 1. Converter a imagem para Tensor
# 2. Normaliza os dados (média 0.1307, std 0.3081 - valores padrão para MNIST)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

# Baixar o dataset de treino
full_train_dataset = torchvision.datasets.MNIST(
    root='./atividades/mlp_mnist/data', 
    train=True, 
    download=True, 
    transform=transform
)

# Baixar o dataset de teste
test_dataset = torchvision.datasets.MNIST(
    root='./atividades/mlp_mnist/data', 
    train=False, 
    download=True, 
    transform=transform
)

# Dividir o treino em treino + validação (50000 para treino, 10000 para validação)
train_size = 50000
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# DataLoaders: criam os mini-batches
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)
test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)



# ==============================================
# --- 3. Definição do Modelo (MLP Profunda) ---
# ==============================================




class MLP(nn.Module):
    def __init__(self, input_size, hidden_1, hidden_2, output_size):
        super(MLP, self).__init__()
        # nn.Sequential é um contêiner que passa os dados por todas as camadas em ordem.
        self.layers = nn.Sequential(
            nn.Flatten(),  # Transforma a imagem 28x28 em um vetor de 784
            nn.Linear(input_size, hidden_1),
            nn.ReLU(),     # Função de ativação
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_size)
            # 'nn.CrossEntropyLoss' aplicada internamente.
        )

    def forward(self, x):
        return self.layers(x)

# Instanciar o modelo e movê-lo para o dispositivo (GPU)
model = MLP(INPUT_SIZE, HIDDEN_SIZE_1, HIDDEN_SIZE_2, OUTPUT_SIZE).to(device)
print(model)



# ===========================================
# --- 4. Loss, Otimizador e TensorBoard ---
# ===========================================



# Função de Perda: Cross-Entropy para classificação multiclasse
criterion = nn.CrossEntropyLoss()

# Otimizador: Adam é robusto e mais rápido que o SGD puro
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Setup do TensorBoard:
# Os logs serão salvos na pasta TENSORBOARD_LOG_DIR
writer = SummaryWriter(TENSORBOARD_LOG_DIR)



# =================================
# --- 5. Funções de Checkpoint ---
# =================================


# Salva o estado do modelo e do otimizador
def save_checkpoint(epoch, model, optimizer, accuracy, is_best):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': accuracy,
    }
    # Salva o checkpoint mais recente
    torch.save(state, "latest_checkpoint.pth")
    if is_best:
        # Salva o melhor checkpoint
        torch.save(state, CHECKPOINT_PATH)
        print(f"*** Novo melhor checkpoint salvo com acurácia: {accuracy:.4f} ***")

# Carrega o estado do modelo e do otimizador a partir de um checkpoint
def load_checkpoint(model, optimizer, checkpoint_file):
    start_epoch = 0
    best_accuracy = 0.0
    
    if os.path.isfile(checkpoint_file):
        print(f"=> Carregando checkpoint '{checkpoint_file}'")
        checkpoint = torch.load(checkpoint_file)
        
        start_epoch = checkpoint['epoch'] + 1 # Começa da *próxima* época
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_accuracy = checkpoint['best_accuracy']
        
        print(f"=> Checkpoint carregado! Treinamento continuará da época {start_epoch}")
    else:
        print(f"=> Nenhum checkpoint encontrado em '{checkpoint_file}', começando do zero.")
        
    return start_epoch, best_accuracy



# ================================
# --- 6. Função de Avaliação ---
# ================================


# Função para avaliar o modelo em um DataLoader (validação ou teste)
def evaluate_model(data_loader, model, criterion, device):
    model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad(): # Desativa o Autograd (como visto no seu notebook)
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Calcular perda
            loss = criterion(outputs, targets)
            total_loss += loss.item() * data.size(0)
            
            # Calcular acurácia
            _, predictions = torch.max(outputs, 1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
    avg_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return avg_loss, accuracy



# ================================
# --- 7. Loop de Treinamento ---
# ================================



print("Iniciando o treinamento...")

# Tenta carregar o último checkpoint, se existir
start_epoch, best_accuracy = load_checkpoint(model, optimizer, "latest_checkpoint.pth")

# Se o 'latest' não existir, tenta carregar o 'best' (para avaliação)
# if start_epoch == 0:
#     start_epoch, best_accuracy = load_checkpoint(model, optimizer, CHECKPOINT_PATH)


global_step = 0 # Para o TensorBoard

for epoch in range(start_epoch, N_EPOCHS):
    
    # --- Treino ---
    model.train() # Coloca o modelo em modo de treino
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Mover dados para o dispositivo
        data, targets = data.to(device), targets.to(device)
        
        # 1. Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 2. Backward pass (Autograd) e Otimização
        optimizer.zero_grad() # Zera os gradientes (como no seu notebook)
        loss.backward()       # Calcula os gradientes
        optimizer.step()      # Atualiza os pesos (como no seu notebook)
        
        running_loss += loss.item()
        
        # Logar perda do treino no TensorBoard a cada 100 batches
        if (batch_idx + 1) % 100 == 0:
            writer.add_scalar('Loss/train', running_loss / 100, global_step)
            running_loss = 0.0
        
        global_step += 1

    # --- Validação (após cada época) ---
    val_loss, val_accuracy = evaluate_model(val_loader, model, criterion, device)
    
    print(f"Época [{epoch+1}/{N_EPOCHS}] - "
          f"Loss Validação: {val_loss:.4f}, "
          f"Acurácia Validação: {val_accuracy * 100:.2f}%")
    
    # Logar perda e acurácia de validação no TensorBoard
    writer.add_scalar('Loss/validation', val_loss, epoch)
    writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

    # --- Checkpoint ---
    is_best = val_accuracy > best_accuracy
    if is_best:
        best_accuracy = val_accuracy
    
    # Salva o checkpoint da época atual (para restauração)
    # e também salva o melhor modelo se a acurácia melhorou
    save_checkpoint(epoch, model, optimizer, best_accuracy, is_best)


print("Treinamento concluído.")
writer.close()



# ============================================
# --- 8. Avaliação Final no Set de Teste ---
# ============================================



print("Carregando o melhor modelo para avaliação final no set de teste...")

# Carrega o *melhor* checkpoint salvo
load_checkpoint(model, optimizer, CHECKPOINT_PATH) 

# Avalia no set de teste
test_loss, test_accuracy = evaluate_model(test_loader, model, criterion, device)

print(f"=================================================")
print(f"Acurácia final no set de teste: {test_accuracy * 100:.2f}%")
print(f"=================================================")

if test_accuracy > 0.98:
    print("Sucesso! Acurácia acima de 98% alcançada.")
else:
    print("Acurácia abaixo de 98%. Tente ajustar hiperparâmetros ou treinar por mais épocas.")