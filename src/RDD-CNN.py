import numpy as np
import cPickle as pkl
import keras, json, gzip
from keras.models import Sequential
from keras.layers import Conv1D, GlobalMaxPooling1D, Merge
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers.embeddings import Embedding
from keras.utils import plot_model, np_utils
from sklearn.model_selection import StratifiedKFold


configuration_pr = json.load(open("configurations/configuration_preprocess_article.json"))[0]
configuration_exp = json.load(open("configurations/configuration_CNN_article.json"))[0]
np.random.seed(configuration_exp["seed"])

f = gzip.open(configuration_pr["outputFilePath"][3:], 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)
f.close()

f = gzip.open(configuration_pr["embeddingsPklPath"][3:], 'rb')
embeddings = pkl.load(f)
f.close()

max_position = (max(np.max(positionTrain1), np.max(positionTrain2))+1)
n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)


print("sentenceTrain: ", sentenceTrain.shape)
print("positionTrain1: ", positionTrain1.shape)
print("yTrain: ", yTrain.shape)
print("Embeddings: ", embeddings.shape)

def create_model():
  distanceModel1 = Sequential()
  distanceModel1.add(Embedding(max_position, configuration_exp["position_dims"], input_length=positionTrain1.shape[1]))

  distanceModel2 = Sequential()
  distanceModel2.add(Embedding(max_position, configuration_exp["position_dims"], input_length=positionTrain2.shape[1]))

  print("VISUALIZING DATA")
  print("embeddings.shape[0] = ", embeddings.shape[0])
  print("embeddings.shape[1] = ", embeddings.shape[1])
  print("input_length = "       ,sentenceTrain.shape[1])

  wordModel = Sequential()
  wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))
  model = Sequential()
  model.add(Merge([wordModel, distanceModel1, distanceModel2], mode='concat'))
  model.add(Conv1D(nb_filter=configuration_exp["nb_filter"], filter_length=configuration_exp["filter_length"], border_mode='same', activation='relu', subsample_length=1))
  model.add(GlobalMaxPooling1D())
  model.add(Reshape((10,10), input_shape=(1,100)))
  model.add(Conv1D(nb_filter=configuration_exp["nb_filter"], filter_length=configuration_exp["filter_length"], activation='relu'))
  model.add(GlobalMaxPooling1D())
  model.add(Dropout(0.25))
  model.add(Dense(n_out, activation='softmax'))
  #plot_model(model, show_shapes = True, to_file='RDD-CNN-Cross.png')
  model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
  model.summary()
  model.save("relation_extraction_model.h5")
  return model


def getPrecision(pred_test, yTest, targetLabel):
    #Precision for non-vague
    targetLabelCount = 0
    correctTargetLabelCount = 0
    print("getPrecision: targetLabel = ", targetLabel)
    for idx in xrange(len(pred_test)):
        if pred_test[idx] == targetLabel:
            targetLabelCount += 1
            if pred_test[idx] == yTest[idx]:
                correctTargetLabelCount += 1
    return 0 if correctTargetLabelCount == 0  else float(correctTargetLabelCount) / targetLabelCount



max_prec, max_rec, max_acc, max_f1 = 0,0,0,0


n_folds = 10
skf = StratifiedKFold(n_splits=n_folds)
cvscores = []  
fold = 0
cvprec = []
cvrec = []
cvf1 = []


fpred=open("predictions-CNN.txt","w")
fpred2=open("real-CNN.txt","w")
for train_index, test_index in skf.split(sentenceTrain, yTrain):
 X_train_sent, X_train_pos1, X_train_pos2 = sentenceTrain[train_index], positionTrain1[train_index], positionTrain2[train_index]
 X_test_sent, X_test_pos1, X_test_pos2    = sentenceTrain[test_index], positionTrain1[test_index], positionTrain2[test_index]
 y_train, y_test = yTrain[train_index], yTrain[test_index]
 model = create_model()
 train_y_cat = np_utils.to_categorical(y_train, n_out)
 print("train_y_cat = ", train_y_cat)
 model.fit([X_train_sent, X_train_pos1, X_train_pos2], train_y_cat, batch_size=configuration_exp["batch_size"], verbose=True,nb_epoch=configuration_exp["nb_epoch"])
 model.save("CNN.h5")
 test_y_cat = np_utils.to_categorical(y_test, n_out)
 scores = model.evaluate([X_test_sent, X_test_pos1, X_test_pos2], test_y_cat)
 print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 cvscores.append(scores[1] * 100)
 print("y_test    = ", y_test)
 pred_test = model.predict_classes([sentenceTrain[test_index], positionTrain1[test_index], positionTrain2[test_index]], verbose=False)
 print("pred_test = ", pred_test)
 precSum = 0
 recSum  = 0
 f1Sum   = 0
 Count   = 0
 for idxsen in range(len(pred_test)):
   fpred.write("%s " % pred_test[idxsen])
   fpred2.write("%s " % y_test[idxsen])
   fpred.write("\n")
   fpred2.write("\n")
 for targetLabel in xrange(0, max(y_test)+1):
      print("targetLabel = ", targetLabel)
      prec = getPrecision(y_test, pred_test, targetLabel)
      print("prec = ", prec)
      rec  = getPrecision(pred_test, y_test, targetLabel)
      print("rec = ", rec)
      f1   = 0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)
      print("f1 = ", f1)
      precSum += prec
      recSum  += rec
      f1Sum   += f1
      Count   += 1
 precfold = precSum / Count
 recfold  = recSum / Count
 f1fold   = f1Sum / Count
 cvprec.append(precfold * 100)
 cvrec.append(recfold * 100)
 cvf1.append(f1fold * 100)
 fold = fold + 1
 print("PREC fold %d: %.2f%% " % (fold, precfold))
 print("REC fold %d: %.2f%% " % (fold, recfold))
 print("F1 fold %d: %.2f%% " % (fold, f1fold))
print("PREC FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvprec), np.std(cvprec)))
print("PEC FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvrec), np.std(cvrec)))
print("F1 FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvf1), np.std(cvf1)))
fpred.close()
fpred2.close()
