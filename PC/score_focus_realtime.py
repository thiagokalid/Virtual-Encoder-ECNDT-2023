import time

import cv2
import matplotlib.pyplot as plt

from visual_encoder.score_focus import score_teng

camera_id = 1  # Altere o id da câmera aqui
total_rec_time = 60  # seconds
max_fps = 30  # Define o FPS máximo desejado

# Inicialize as listasc para armazenar os scores
score_history = []

# Define a função para atualizar e exibir o gráfico em tempo real
def update_graph(score):
    global score_history  # Declarar como global
    score_history.append(score)
    score_history = score_history[-100:]  # Mantenha apenas os últimos 20 valores

    plt.clf()  # Limpe o gráfico anterior
    plt.plot(score_history)
    plt.xlabel('Frame')
    plt.ylabel('Score')
    plt.title('Score em tempo real ' + "{:.3f}".format(score))  # Exibe apenas 3 dígitos após a vírgula
    plt.pause(0.01)  # Pause para atualizar o gráfico

# define a video capture object
print('Requesting access to camera. This may take a while...')
vid = cv2.VideoCapture(camera_id)
print('Got access to camera!')

frame_num = 0  # para guardar o número de frames.
start_time = time.time()

while time.time() <= start_time + total_rec_time:
    # Calcular o tempo de execução de cada iteração
    start_iteration_time = time.time()

    ret, frame = vid.read(1)
    if ret:
        cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        focus_score = score_teng(cv2_img) #pode ser alterado por qualquer função de score -> disponível em score_focus.py
        update_graph(focus_score)
    else:
        print("Erro ao ler o quadro.")

    # Calcular o tempo de espera necessário para manter o FPS máximo
    elapsed_time = time.time() - start_iteration_time
    wait_time = max(1.0 / max_fps - elapsed_time, 0)
    time.sleep(wait_time)

vid.release()
plt.show()  # Exibe o gráfico final após a execução do loop
