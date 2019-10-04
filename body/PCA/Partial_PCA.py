from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
import random
import copy

############# AIXO SON MODELS PER PARTS ##############
#part = 'belly'
#part = "height"
#part = "breast"
#part = "torso"
#part = "armslegs"
#part = "hip"
#part = "gender"
#part = "weight"
#part = "muscle"
#part = "length"
part = "strength"
######################################################
print(part)
path_input = '/home/aniol/IRI/DadesMarta/models/parts/' + part +"/humans"
path_output = '/home/aniol/IRI/DadesMarta/codi/PCA/Eigenbodies/parts/' + part
n_models = 20 #3000 molaria
X = []
mean = []
n_vertex = 13380 # M ha sortit aixo #13652 # 13652 is without eyes, 14444 is with eyes :)
n_coords = n_vertex*3
mean = mean + n_coords * [0]
information = []


print("Loading models")
for m in range(0,n_models):
    c = 0
    model_name = "Model%04d_naked" % (m)
    model = "%s/%s.obj" % (path_input, model_name)
    file = open(model,'r')
    vector = []
    for i in range(0,4):
        file.readline()
    for line in file:
        line = line.split()
        if line[0] == 'v':
            c+=1
            vector.append(float(line[1]))
            vector.append(float(line[2]))
            vector.append(float(line[3]))
    X.append(vector)

    mean = [x + y for x, y in zip(mean, vector)]
print("Number of vertex... n_vertex/c -> " +str(n_vertex)+"/"+str(c))
print("Models loaded")
mean = [x / n_models for x in mean]


#X = map(list, zip(*X)) # this is a transpose (np.transpose(X))
# PER VISUALITZAR CORRECTAMENT EL TEMA MILLOR ESCRIURE EN UN FITXER LA X A VEURE QUE COI ESCRIU JODERT
X = np.array(X)
X = np.transpose(X)

print(len(X))
print(len(X[0]))
newX = []

#newX = np.array(newX)

for i in range(0,len(X)):
    new_coords= []
    for element in X[i]:
        list = np.array(element)
        new_coords.append(list-mean[i])
    newX.append(new_coords)



#X = preprocessing.scale(X)
newX = np.array(newX)
#newX = np.array(X)
X_scaled = []
for coordinate in newX:
    list = np.array(coordinate)
    information.append([list.mean(),list.std()])
    list = preprocessing.scale(list)
    X_scaled.append(list)


print("Computing PCA")
# Compute a PCA
pca = PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
  svd_solver='full', tol=0.0, whiten=False)
pca.fit(X_scaled)
print(pca.explained_variance_ratio_)

print(pca.components_)
#components = pca.components_

C = pca.get_covariance()

data = pca.transform(X_scaled)
components = np.transpose(data)

first = components[0]
first = np.array(first)
second = components[1]
second = np.array(second)

information = np.transpose(information)
mean = np.array(mean)

eigenbody0 = []
eigenbody0 = first *information[1]  + information[0] + mean

eigenbody1 = []
eigenbody1 = second *information[1] + information[0] + mean

eigenbody0 = np.array(eigenbody0)
eigenbody1 = np.array(eigenbody1)


print("Writing eigenbodies txt")
model_name = "eigenbody%d" % (0)
model2 = "%s/%s.txt" % (path_output, model_name)
PrincipalComponents = open(model2,'w')

for coord in eigenbody0:
    PrincipalComponents.write(str(coord) + "\n")

PrincipalComponents.close()

model_name = "eigenbody%d" % (1)
model2 = "%s/%s.txt" % (path_output, model_name)
PrincipalComponents1 = open(model2,'w')

for coord in eigenbody1:
    PrincipalComponents1.write(str(coord) + "\n")

PrincipalComponents1.close()

model_name = "StandardModel"
model2 = "%s/%s.txt" % (path_output, model_name)
StandardModel = open(model2,'w')
mean = np.array(mean)
mean = np.transpose(mean)

for coord in mean:
    StandardModel.write(str(coord)+"\n")

#print(X_pca)
StandardModel.close()
