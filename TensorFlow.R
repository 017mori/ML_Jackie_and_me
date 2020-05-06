####Importando bibliotecas ####
library(EBImage)
library(keras)
library(caret)
library(pbapply)
library(tensorflow)

## Definindo os diretorios
jack_dir <- "JACK/"
mat_dir <- "MAT/"
test_dir <- "TESTE/"

##Altura e Largura das imagens
width <-30
heigth <- 30

####FUNÇÃO: Extração de caracteristicas ####
extrair_caracteristicas <- function(dir_path, width, heigth){
  img_size <- width*heigth
  image_name <- list.files(dir_path)
  print(paste("Iniciando Processo", length(image_name), "imagens"))

  lista_parametros <- pblapply(image_name, function(imgname){
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- resize(img, w = width, h = heigth)
    img_matrix <- as.matrix(img_resized@.Data)
    img_vector <- as.vector(t(img_matrix))
    
    return(img_vector)
  })
  
  feature_matrix <- do.call(rbind, lista_parametros)
  feature_matrix <- as.data.frame(feature_matrix)
  
  names(feature_matrix) <- paste0("pixel", c(1:img_size))
  return(feature_matrix)
}

#### Aplicando as nossas imagens ####
jack_data <- extrair_caracteristicas(dir_path = jack_dir, 
                                     width = width, heigth = heigth)
mat_data <- extrair_caracteristicas(dir_path = mat_dir,
                                    width = width, heigth = heigth)

# Nomeando os bancos de dados
jack_data$label <- 0 
mat_data$label <- 1

#Unindo nosso bancos de dados
allData <- rbind(jack_data, mat_data)

#Pequena amostra para o treinamento 
indices <- createDataPartition( allData$label, 
                                p = 0.90, list = FALSE )
train <- allData[indices, ]
test <- allData[-indices, ]

#Tornando as respostas em Categoricas
trainLabels <- to_categorical((train$label))
testLabels <- to_categorical((test$label))

#Data frame ou conjunto de dados
x_train <- data.matrix(train[, -ncol(train)])
y_train <- data.matrix(train[, ncol(train)])

x_test <- data.matrix(test[, -ncol(test)])
y_test <- data.matrix(test[, ncol(test)])

#Criando o modelo sequencial
model <- keras_model_sequential()
model %>%
  layer_dense(units = 256, activation = 'relu', input_shape = c(2700)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 2, activation = 'softmax')

summary(model)

#Copilando o modelo
model %>%
  compile(loss = "binary_crossentropy", optimizer_adam(), 
          metrics = c('accuracy'))

#Aplicando os nossos dados ao modelo
history <-model %>%
  fit(x_train, trainLabels, epochs = 10,
      batch_size = 32, validation_split = 0.2)

plot(history)

##### Avaliando o modelo (ERRO)
model %>% evaluate(x_test, testLabels, verbose = 1)

# Predição do modelo
pred <- model %>% predict_classes(x_test)
table(Predicted = pred, Reais = y_test)

#TESTE FINAL 
test <- extrair_caracteristicas(test_dir, width = width, heigth = heigth)
pred_test <- model %>% predict_classes(as.matrix(test))
pred_test
A = pred_test

#Resultado dos testes
for (i in 1:length(A)){
  if (A[i] == 0){
    A[i] = "JACK CHAN"
  }else{
    A[i] = "MATHEUS"
  }  
}
print(matrix(A))
