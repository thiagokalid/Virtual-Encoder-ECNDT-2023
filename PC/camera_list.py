import cv2

def list_ports():
    """
    Testa as portas e retorna uma tupla com as portas disponíveis e as que estão funcionando.
    """
    non_working_ports = []
    working_ports = []
    available_ports = []

    # Limitar o intervalo de portas de câmera a serem verificadas
    start_port = 0
    end_port = 10  # Escolha um valor razoável de acordo com o número de portas disponíveis em seu sistema

    for dev_port in range(start_port, end_port):
        try:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
                print("Porta %s não está funcionando." % dev_port)
                continue

            is_reading, img = camera.read()
            if is_reading:
                w = camera.get(3)
                h = camera.get(4)
                print("Porta %s está funcionando e lê imagens (%s x %s)." % (dev_port, h, w))
                working_ports.append(dev_port)
            else:
                w = camera.get(3)
                h = camera.get(4)
                print("Porta %s para câmera (%s x %s) está presente mas não lê." % (dev_port, h, w))
                available_ports.append(dev_port)

            # Libera a câmera
            camera.release()
        except Exception as e:
            print(f"Erro ao verificar porta {dev_port}: {e}")

    return available_ports, working_ports, non_working_ports

available_ports, working_ports, non_working_ports = list_ports()
print("Portas disponíveis:", available_ports)
print("Portas funcionando:", working_ports)
print("Portas não funcionando:", non_working_ports)