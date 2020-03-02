# coding: utf-8
# pip install numpy scikit-learn
import time

import numpy as np
import re
import os
from collections import defaultdict
from sklearn import svm
from sklearn.model_selection import KFold


PRIMELE_N_CUVINTE = 6000


def accuracy(y, p):
    return 100 * (y == p).astype('int').mean()


def files_in_folder(mypath):
    fisiere = []
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)):
            fisiere.append(os.path.join(mypath, f))
    return sorted(fisiere)


def extrage_fisier_fara_extensie(cale_catre_fisier):
    nume_fisier = os.path.basename(cale_catre_fisier)
    nume_fisier_fara_extensie = nume_fisier.replace('.txt', '')
    return nume_fisier_fara_extensie


def citeste_texte_din_director(cale):
    date_text = []
    iduri_text = []
    for fis in files_in_folder(cale):
        id_fis = extrage_fisier_fara_extensie(fis)
        iduri_text.append(id_fis)
        with open(fis, 'r', encoding='utf-8') as fin:
            text = fin.read()
            #text_fara_punct = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
        date_text.append(text.split())
    return (iduri_text, date_text)

### citim datele
dir_path = './trainData/'
labels = np.loadtxt(os.path.join(dir_path, 'labels_train.txt'))
train_data_path = os.path.join(dir_path, 'trainExamples')
iduri_train, data = citeste_texte_din_director(train_data_path)
print(data[0][:10])

### numaram cuvintele din toate documentele
contor_cuvinte = defaultdict(int)
for doc in data:
    for word in doc:
        contor_cuvinte[word] += 1

# transformam dictionarul in lista de tupluri ['cuvant1': frecventa1, 'cuvant2': frecventa2]
perechi_cuvinte_frecventa = list(contor_cuvinte.items())

# sortam descrescator lista de tupluri dupa frecventa
perechi_cuvinte_frecventa = sorted(perechi_cuvinte_frecventa, key=lambda kv: kv[1], reverse=True)

# extragem primele N cele mai frecvente cuvinte din toate textele
perechi_cuvinte_frecventa = perechi_cuvinte_frecventa[0:PRIMELE_N_CUVINTE]

print ("Primele 10 cele mai frecvente cuvinte ", perechi_cuvinte_frecventa[0:10])


list_of_selected_words = []
for cuvant, frecventa in perechi_cuvinte_frecventa:
    list_of_selected_words.append(cuvant)
### numaram cuvintele din toate documentele


def get_bow(text, lista_de_cuvinte):
    contor = dict()
    cuvinte = set(lista_de_cuvinte)
    for cuvant in cuvinte:
        contor[cuvant] = 0
    for cuvant in text:
        if cuvant in cuvinte:
            contor[cuvant] += 1
    return contor

def get_bow_pe_corpus(corpus, lista):
    bow = np.zeros((len(corpus), len(lista)))
    for idx, doc in enumerate(corpus):
        bow_dict = get_bow(doc, lista)
        v = np.array(list(bow_dict.values()))
        v = (v - np.mean(v)) / np.std(v)
        bow[idx] = v
    return bow


data_bow = get_bow_pe_corpus(data, list_of_selected_words)
print("Data bow are shape: ", data_bow.shape)

nr_exemple_train = 2000
nr_exemple_valid = 500
nr_exemple_test = len(data) - (nr_exemple_train + nr_exemple_valid)

indici_train = np.arange(0, nr_exemple_train)
indici_valid = np.arange(nr_exemple_train, nr_exemple_train + nr_exemple_valid)
indici_test = np.arange(nr_exemple_train + nr_exemple_valid, len(data))


for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    clasificator = svm.LinearSVC(C=C, dual=False)
    clasificator.fit(data_bow[indici_train, :], labels[indici_train])

    predictii = clasificator.predict(data_bow[indici_valid, :])
    print("Acuratete pe validare cu C =", C, ": ", accuracy(predictii, labels[indici_valid]))


indici_train_valid = np.concatenate([indici_train, indici_valid])
clasificator = svm.LinearSVC(C=10, dual=False)
start_time = time.time()
clasificator.fit(data_bow[indici_train_valid, :], labels[indici_train_valid])
print("Timpul de antrenare: ", time.time() - start_time, "s")
predictii = clasificator.predict(data_bow[indici_test])
print ("Acuratete pe test cu C = 10: ", accuracy(predictii, labels[indici_test]))

M = np.zeros((11, 11))
for (a, b) in zip(predictii, labels[indici_test]):
    M[int(a)][int(b)] += 1
print(M)

X = np.arange(2983)
np.random.shuffle(X)
acc = 0

kfold = KFold(10)
for train, test in kfold.split(X):
    clasificator = svm.LinearSVC(C=10, dual=False)
    clasificator.fit(data_bow[train, :], labels[train])
    predictii = clasificator.predict(data_bow[test])
    acc += accuracy(predictii, labels[test])
err = 100 - acc/10
print("Eroare:", err)


def scrie_fisier_submission(nume_fisier, predictii, iduri):
    with open(nume_fisier, 'w') as fout:
        fout.write("Id,Prediction\n")
        for id_text, pred in zip(iduri, predictii):
            fout.write(id_text + ',' + str(int(pred)) + '\n')


testId = np.arange(2984, 4480+1)
predictedLabels = 6*np.ones(1497)


cale_data_test = './testData-public'
indici_test, date_test = citeste_texte_din_director(cale_data_test)
print('Am citit: ', len(date_test))
data_bow_test = get_bow_pe_corpus(date_test, list_of_selected_words)


clf = svm.LinearSVC(C = 1, dual=False)
clf.fit(data_bow, labels)
predicte = clf.predict(data_bow_test)

scrie_fisier_submission('D:\PyCharm\proiect_32\Project_kaggle.csv', predicte, indici_test)