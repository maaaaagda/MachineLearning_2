# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

from __future__ import division
import numpy as np
import scipy.sparse as sp
import scipy.spatial.distance as p
from scipy.spatial.distance import cdist
from collections import Counter


def hamming_distance(X, X_train):
    """
    :param X: zbior porownwanych obiektow N1xD
    :param X_train: zbior obiektow do ktorych porownujemy N2xD
    Funkcja wyznacza odleglosci Hamminga obiektow ze zbioru X od
    obiektow X_train. ODleglosci obiektow z jednego i drugiego
    zbioru zwrocone zostana w postaci macierzy
    :return: macierz odleglosci pomiedzy obiektami z X i X_train N1xN2
    """
    X = sp.spmatrix.toarray(X)
    X_train = sp.spmatrix.toarray(X_train)
    return p.cdist(X, X_train, "hamming") * X.shape[1]


def sort_train_labels_knn(Dist, y):
    """
    Funkcja sortujaca etykiety klas danych treningowych y
    wzgledem prawdopodobienstw zawartych w macierzy Dist.
    Funkcja zwraca macierz o wymiarach N1xN2. W kazdym
    wierszu maja byc posortowane etykiety klas z y wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist
    :param Dist: macierz odleglosci pomiedzy obiektami z X
    i X_train N1xN2
    :param y: wektor etykiet o dlugosci N2
    :return: macierz etykiet klas posortowana wzgledem
    wartosci podobienstw odpowiadajacego wiersza macierzy
    Dist. Uzyc algorytmu mergesort.
    """
    n1 = len(Dist[:, 0])
    n2 = len(y)

    w = np.zeros(shape=(n1, n2))
    print(y)

    for i in range(0, n1):
        w[i, :] = y[Dist[i, :].argsort(kind='mergesort')]
    return w


def p_y_x_knn(y, k):
    """
    Funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla
    kazdej z klas dla obiektow ze  zbioru testowego wykorzystujac
    klasfikator KNN wyuczony na danych trenningowych
    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najblizszuch sasiadow dla KNN
    :return: macierz prawdopodobienstw dla obiektow z X
    """
    n1 = y.shape[0]
    n2 = len(np.unique(y))

    w = np.zeros(shape=(n1, n2))
    for i in range(n1):
        c = Counter(y[i, :k])
        for z in range(1, n2+1):
            c.update({z: 0})
        d = sorted(c.keys())
        j = 0
        for p in d:
            w[i, j] = c[p]
            j+=1

    return w/k



def classification_error(p_y_x, y_true):
    """
    Wyznacz blad klasyfikacji.
    :param p_y_x: macierz przewidywanych prawdopodobienstw
    :param y_true: zbior rzeczywistych etykiet klas 1xN.
    Kazdy wiersz macierzy reprezentuje rozklad p(y|x)
    :return: blad klasyfikacji
    """

    n1 = p_y_x.shape[0]
    p_y_x = np.fliplr(p_y_x)
    r = p_y_x.shape[1] - np.argmax(p_y_x, axis=1)   #ostatnie wystapienie
    q = np.count_nonzero(r-y_true)
    return q/n1


def model_selection_knn(Xval, Xtrain, yval, ytrain, k_values):
    """
    :param Xval: zbior danych walidacyjnych N1xD
    :param Xtrain: zbior danych treningowych N2xD
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartosci parametru k, ktore maja zostac sprawdzone
    :return: funkcja wykonuje selekcje modelu knn i zwraca krotke (best_error,best_k,errors), gdzie best_error to najnizszy
    osiagniety blad, best_k - k dla ktorego blad byl najnizszy, errors - lista wartosci bledow dla kolejnych k z k_values
    """
    y = sort_train_labels_knn(hamming_distance(Xval, Xtrain), ytrain)
    l = len(k_values)
    errors = []
    best_error = 10
    best_k = -1
    for k in range (len(k_values)):
        pxy = p_y_x_knn(y, k_values[k])
        e = classification_error(pxy, yval)
        if(e<best_error):
            best_error = e
            best_k = k_values[k]
        errors.append(e)

    return (best_error, best_k, errors)


def estimate_a_priori_nb(ytrain):
    """
    :param ytrain: etykiety dla dla danych treningowych 1xN
    :return: funkcja wyznacza rozklad a priori p(y) i zwraca p_y - wektor prawdopodobienstw a priori 1xM
    """
    c = Counter(ytrain)
    s = sorted(c.items(), key=lambda x: x[0])
    u = len(ytrain)
    return [x[1]/u for x in s]


def estimate_p_x_y_nb(Xtrain, ytrain, a, b):
    """
    :param Xtrain: dane treningowe NxD
    :param ytrain: etykiety klas dla danych treningowych 1xN
    :param a: parametr a rozkladu Beta
    :param b: parametr b rozkladu Beta
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(x|y) zakladajac, ze x przyjmuje wartosci binarne i ze elementy
    x sa niezalezne od siebie. Funkcja zwraca macierz p_x_y o wymiarach MxD.
    """"""

    n1 = Xtrain.shape[0]        #50
    n2 = Xtrain.shape[1]        #20
    q = np.unique(ytrain)       #klasy
    w = len(q)                  #ilosc klas
    k = np.zeros(shape=(n2, w))
    d = Counter(ytrain) # zliczam ilosci poszczegolnych etykietkow potem bede przez nie dzieliła
    t = sorted(d.items(), key=lambda x: x[0])
    h = np.array([tt[1] for tt in t])
    print(h)
    for i in range(n2):
        u = []
        for j in range(n1):
            x = Xtrain[j, i]*ytrain[j]
            if(x!=0):
                u.append(x)
        c = Counter(u)
        for z in range(1, w + 1):       #dodaje te klasy, ktore mogly nie zostac wylapane, moze nikonieczne
            c.update({z: 0})
            d.update({z: 0})
        s = sorted(c.items(), key=lambda x: x[0])         #sortuje po kluczach - klasach
        k[i]=[ss[1] for ss in s]
        k[i]=np.divide((k[i]+a-1),(h+a+b-2))
    return k.T"""
    Xtrain = Xtrain.toarray()
    NUMBER_OF_CLASSES = 4
    rows = []

    def summ(row, yequalk):
        return np.sum(np.bitwise_and(row, yequalk))

    for i in range(1, NUMBER_OF_CLASSES + 1):
        yk = np.equal(ytrain, i)
        yksum = np.sum(yk)          #ilosc wystapien danej klasy
        row = np.apply_along_axis(summ, axis=0, arr=Xtrain, yequalk=yk) #dla danej klasy zlicza ilosc jedyneczek
        rows.append(np.divide(np.add(row, a - 1), yksum + a + b - 2))

    return np.vstack(rows)


def p_y_x_nb(p_y, p_x_1_y, X):
    """
    :param p_y: wektor prawdopodobienstw a priori o wymiarach 1xM
    :param p_x_1_y: rozklad prawdopodobienstw p(x=1|y) - macierz MxD
    :param X: dane dla ktorych beda wyznaczone prawdopodobienstwa, macierz NxD
    :return: funkcja wyznacza rozklad prawdopodobienstwa p(y|x) dla kazdej z klas z wykorzystaniem klasyfikatora Naiwnego
    Bayesa. Funkcja zwraca macierz p_y_x o wymiarach NxM.
    """
    X = X.toarray()

    def normalise(row):
        Z = 1 / np.sum(row)
        return np.multiply(row, Z)

    def calc(row):
        out = X * row
        out += np.negative(X) - np.negative(X) * row
        out = np.apply_along_axis(np.prod, arr=out, axis=1)
        return out

    def calc2(row, x2):
        return np.multiply(row, x2)

    test = np.apply_along_axis(calc, axis=0, arr=p_x_1_y.transpose())
    test = np.apply_along_axis(calc2, axis=1, arr=test, x2=p_y)
    test = np.apply_along_axis(normalise, axis=1, arr=test)

    return test


def model_selection_nb(Xtrain, Xval, ytrain, yval, a_values, b_values):
    """
    :param Xtrain: zbior danych treningowych N2xD
    :param Xval: zbior danych walidacyjnych N1xD
    :param ytrain: etykiety klas dla danych treningowych 1xN2
    :param yval: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrow a do sprawdzenia
    :param b_values: lista parametrow b do sprawdzenia
    :return: funkcja wykonuje selekcje modelu Naive Bayes - wybiera najlepsze wartosci parametrow a i b. Funkcja zwraca
    krotke (error_best, best_a, best_b, errors) gdzie best_error to najnizszy
    osiagniety blad, best_a - a dla ktorego blad byl najnizszy, best_b - b dla ktorego blad byl najnizszy,
    errors - macierz wartosci bledow dla wszystkich par (a,b)
    """
    A = len(a_values)
    B = len(b_values)
    error_best = 2
    errors = np.zeros(shape=(A, B), dtype=np.float64)
    best_a = 0
    best_b = 0

    for a in range(A):
        for b in range(B):
            p_y = np.squeeze(np.asarray(estimate_a_priori_nb(ytrain)))
            p_x_1_y = estimate_p_x_y_nb(Xtrain, ytrain, a_values[a], b_values[b])
            p_y_x = p_y_x_nb(p_y, p_x_1_y, Xval)
            error = classification_error(p_y_x, yval)
            errors[a, b] = error
            if error < error_best:
                error_best = error
                best_a = a
                best_b = b

    return error_best, a_values[best_a], b_values[best_b], errors





















"""
a = np.array([[1,1]])
b = np.array([[0,1]])
aa = np.array([[1,1], [0, 1], [1,1]])
bb = np.array([[0,0],[0,1]])
def xor(a, b):
    return a != b
def w_xor(a, b):
    return np.sum(xor(a, b))
print((cdist(aa, bb, 'hamming')))
print((cdist(aa, bb, 'hamming'))*a.shape[1])
print(xor(aa, b))
print(np.sum(xor(bb, aa)))"""