
import cv2

# Inicia a captura da webcam (0 = câmera padrão)
camera = cv2.VideoCapture(0)

# Verifica se a câmera foi aberta corretamente
if not camera.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

# Loop principal para leitura dos frames da câmera
while True:
    # Captura frame por frame
    ret, frame = camera.read()

    # Verifica se a captura foi bem-sucedida
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Converte para tons de cinza (recomendado para detecção)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # Detecta rostos no frameq
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Desenha retângulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o frame com as detecções
    cv2.imshow("Detecção de Faces (pressione 'q' para sair)", frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos
camera.release()
cv2.destroyAllWindows()

