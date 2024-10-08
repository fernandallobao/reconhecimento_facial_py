# importando biblioteca
import cv2 #h
import numpy as np   #faz o mapeamento do face de forma numerica
import os

def captura(largura, altura):
    #classificadores
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')

    #abre camera
    camera = cv2.VideoCapture(0) #0 camera padrão do notebook

    #amostrar da imegem do usuario
    amostra = 1
    n_amostras = 25 #25 imagens diferentes do usuario para que ele consiga diferenciar um usuario do outro

    #recebe o ID do usuario
    id = input('Digite o ID do usuário: ')
    
    #msg indicando a captura das imagens
    print('Capturando as imagens...')

    #loop
    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY) #TRANSFORMA AS IMAGENS EM CINZA
        #define a escala de cinza
        print(np.average(imagem_cinza)) 
        faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150,150)) #minSize escala da imagem

        #identifica a geometria das faces
        for (x, y, l, a) in faces_detectadas:
            cv2.rectangle(imagem, (x,y), (x+l, y+a), (0,0,255), 2)
            regiao = imagem[y:y + a, x:x + l]
            regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho) #retira a parte do olho

            #identifica a geometria dos olhos
            for (ox, oy, ol, oa) in olhos_detectados:
                cv2.rectangle(regiao, (ox, oy), (ox+ol, oy+oa), (0, 255,0), 2)

            
            if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
                cv2.imwrite(f'fotos/pesso.{str(id)}.{str(amostra)}.jpg', imagem_face)
                print(f'[foto] {str(amostra)} capturada com sucesso')
                amostra += 1
            
        cv2.imshow('Detectar faces', imagem)    
        cv2.waitKey(1)

        if (amostra >= n_amostras + 1):
            print('Faces capturadas com sucesso.')
            break
        elif cv2.waitKey(1) == ord('q'):
            print('Camera encerrada.')
            break
    #encerra a captura
    camera.release()
    cv2.destroyAllWindows()
    #fim da função
            
def get_imagem_com_id(): #ler as fotos e captura os dados em uma lista
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')] #for list compreenching 
    faces = []
    ids = []

    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY) #pra pegar a cor cinza das imagens
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1]) #pega valor str e separa em uma lista
        ids.append(id)
        faces.append(imagem_face)

    return np.array(ids), faces

def treinamento():
    #criando os elementos de reconhecimento necessarios 
    eigenface = cv2.face.EigenFaceRecognizer_create() #pega as imagens em cinza para o treinamento
    fisherface = cv2.face.FisherFaceRecognizer_create() #mapa do rosto
    lbph = cv2.face.LBPHFaceRecognizer_create()

    ids, faces = get_imagem_com_id()

    #treinanda o algoritimo do programa
    print('Treinando...')
    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')
    fisherface.train(faces, ids)
    fisherface.write('classificadorFisher.yml')
    lbph.train(faces, ids)
    lbph.write('classificadorLBPH.yml')

    #finaliza treinamento
    print('Treinamento fiznalizado com sucesso!')

#NOTE - PROGRAMA PRINCIPAL
if __name__ == '__main__':
    #define o tamanho da camera
    largura = 220
    altura = 220

    while True:
        #menu
        print('0 - Sair do programa!')
        print('1 - Capturar imagem do usuário.')
        print('2 - Treinar sistema.')
        
        op = input('Opção desejada: ')

        match op:
            case '0':
                print('Programa encerrado!')
                break
            case '1':
                captura(largura, altura)
                continue
            case '2':
                treinamento()
                continue
            case _:
                print('Opção inválida!')
                continue
            