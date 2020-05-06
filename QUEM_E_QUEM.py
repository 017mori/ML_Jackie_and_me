#Classificação Vatorial de Suporte Linear
#Classificação por Regressão Vetorial de Suporte


import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.svm import SVR

#### Importando as imagens do Jackie chan e as minhas fotos####
#Fotos Jackie Chan
jack1_i = cv2.imread(r'C:\Users\jack (1).jpg') 
jack2_i = cv2.imread(r'C:\Users\jack (2).jpg') 
jack3_i = cv2.imread(r'C:\Users\jack (3).jpg') 
jack4_i = cv2.imread(r'C:\Users\jack (4).jpg') 
jack5_i = cv2.imread(r'C:\Users\jack (5).jpg') 
jack6_i = cv2.imread(r'C:\Users\jack (6).jpg') 
jack7_i = cv2.imread(r'C:\Users\jack (7).jpg') 
jack8_i = cv2.imread(r'C:\Users\jack (8).jpg') 

#Minhas fotos
mat1_i = cv2.imread(r'C:\Users\mat (1).jpg') 
mat2_i = cv2.imread(r'C:\Users\mat (2).jpg') 
mat3_i = cv2.imread(r'C:\Users\mat (3).jpg') 
mat4_i = cv2.imread(r'C:\Users\mat (4).jpg') 
mat5_i = cv2.imread(r'C:\Users\mat (5).jpg') 
mat6_i = cv2.imread(r'C:\Users\mat (6).jpg') 
mat7_i = cv2.imread(r'C:\Users\mat (7).jpg') 
mat8_i = cv2.imread(r'C:\Users\mat (8).jpg') 

#print(jack1_i.shape) #Tamanho das imagens

#### Redimencionando as imagens ####
jack1 = cv2.resize(jack1_i,(10,10)) #(10,10,3)
jack2 = cv2.resize(jack2_i,(10,10))
jack3 = cv2.resize(jack3_i,(10,10)) #(10,10,3)
jack4 = cv2.resize(jack4_i,(10,10))
jack5 = cv2.resize(jack5_i,(10,10)) #(10,10,3)
jack6 = cv2.resize(jack6_i,(10,10))
jack7 = cv2.resize(jack7_i,(10,10)) #(10,10,3)
jack8 = cv2.resize(jack8_i,(10,10))

mat1  = cv2.resize(mat1_i,(10,10))
mat2  = cv2.resize(mat2_i,(10,10))
mat3  = cv2.resize(mat3_i,(10,10))
mat4  = cv2.resize(mat4_i,(10,10))
mat5  = cv2.resize(mat5_i,(10,10))
mat6  = cv2.resize(mat6_i,(10,10))
mat7  = cv2.resize(mat7_i,(10,10))
mat8  = cv2.resize(mat8_i,(10,10))

#print(jack1.shape) 

#### Concatenando as imagens ####
X = np.concatenate((jack1, jack2, jack3, jack4, jack5, jack6, jack7, jack8,
                    mat1, mat2, mat3, mat4, mat5, mat6, mat7, mat8), axis = 0)
#print(X.shape)

#### Matriz das imagens####
y = list(range(1,17)) #Indices para cada imagem 
y = np.array(y) #Convertendo em Matriz
Y = y.reshape(-1)
X = X.reshape(len(y), -1)
#print(X)

#### Treinamento ####
# Criando objetos para o treinamento 
clf_lin = SVC(kernel='linear') #linear, poly, rbf
svr_lin = SVR(kernel='poly')   #linear, poly, rbf

#### Importando a imagem de teste ####
test1_i = cv2.imread(r'C:\Users\test (2).jpg') 
test1 = cv2.resize(test1_i,(10,10))

#### MODELO SVC ####
print('----------------------------------------------------------------')
print('Inicio do treinamento do Modelo SVC')

clf_lin.fit(X,Y)

print('Termino do treinamento do Modelo SVC')

#### TESTE 1 do modelo criado ####
predicao = clf_lin.predict(test1.reshape(1,-1))
score = clf_lin.score(X,Y)

print(predicao)
print(score)

#Condições de resultados 
if predicao <= 8:
    print("O asiatico escolhido é o JACKIE CHAN")
if predicao > 8:
    print("O asiatico escolhido é o MATHEUS")

#### MODELO SVR ####
print('----------------------------------------------------------------')
print('Inicio do treinamento do Modelo SVR')

svr_lin.fit(X,Y)

print('Termino do treinamento do Modelo SVR')

#### TESTE 2 do modelo criado ####
predicao = svr_lin.predict(test1.reshape(1,-1))
score = svr_lin.score(X,Y)

print(predicao)
print(score)

#Condições de resultados 
if predicao <= 8:
    print("O asiatico escolhido é o JACKIE CHAN")
if predicao > 8:
    print("O asiatico escolhido é o MATHEUS")



