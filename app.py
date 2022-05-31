#*********************************************************************************************************************
#*******************************************     PAGINA    ***********************************************************
#*********************************************************************************************************************
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "1234"

@app.route("/home")
def index():
    flash("table.to_html()")
    return render_template("index.html")

#*********************************************************************************************************************
#*******************************************     CODIGO    ***********************************************************
#*********************************************************************************************************************
import re, collections
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import pandas as pd
import math
from scipy import spatial
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch 
from scipy.cluster.hierarchy import linkage, fcluster
import statsmodels.api as sm

url = "https://raw.githubusercontent.com/rebecau/ML_PROYECTO1/main/ml_data.csv"
df = pd.read_csv(url)

title = df.iloc[:, 0]
keyword = df.iloc[:, 1]
abstract = df.iloc[:, 2]

def Normalizacion(Documentos):
  for i in range(len(Documentos)):#NORMALIZACION ELIMINACION DE CARACTERES ESPECIALES Y MAYUSCULA
    Documentos[i] = re.sub('[^A-Za-z0-9]+', ' ', Documentos[i])
    Documentos[i] = Documentos[i].lower()
  return Documentos

d1 = Normalizacion(title)
d2 = Normalizacion(keyword)
d3 = Normalizacion(abstract)

def TOKENIZACION(Documentos):
  for i in range(len(Documentos)):#TOKENIZACION
    Documentos[i] = Documentos[i].split()
  return Documentos

d1 = TOKENIZACION(d1)
d2 = TOKENIZACION(d2)
d3 = TOKENIZACION(d3)

def Diccionario(documentos):
  Dic = []
  for i in range(len(documentos)):
    wordset = documentos[i]
    for j in range(len(wordset)):
      Dic.append(wordset[j])
  Dic = list(set(Dic))
  return Dic

dic_tittle = Diccionario(d1)
dic_keyword = Diccionario(d2)
dic_abstract = Diccionario(d3)

#STOPWORDS
def STOPWORDS(Documentos):
  for i in range(len(Documentos)):#STOPWORDS
    n = stopwords.words("english")
    for word in Documentos[i]:
      if word in n:
        Documentos[i].remove(word)
  return Documentos

d1 = STOPWORDS(d1)
d2 = STOPWORDS(d2)
d3 = STOPWORDS(d3)

#STEMMING
def STEMMING(Documentos):
  for i in range(len(Documentos)):#stemming(reduce hacia la raiz/ORIGEN)
    stemmer = PorterStemmer()
    stem = Documentos[i]
    for j in range(len(Documentos[i])):
      d11 = stemmer.stem(stem[j])
      stem[j] = d11
    Documentos[i] = stem
  return Documentos

d1 = STEMMING(d1)
d2 = STEMMING(d2)
d3 = STEMMING(d3)

#COSENO
def Coseno(Dic_d3):
  Diccionario3 = []
  for i in range(len(Dic_d3)):
    wordset = Dic_d3[i]
    for j in range(len(wordset)):
      Diccionario3.append(wordset[j])
  Diccionario3 = list(set(Diccionario3))
  return Diccionario3

coseno_d1 = Coseno(d1)
coseno_d2 = Coseno(d2)
coseno_d3 = Coseno(d3)


#Bolsa de palabras (Bag Of Words) -> Modelo de incidencia binaria [Matriz]
def calculateBOW(wordset,l_doc):
  tf_diz = dict.fromkeys(wordset,0)
  bag_words = []
  for i in range(len(l_doc)):
    cont = []
    vec = l_doc[i]
    for word in wordset:#numero de palabras por resumen
      #print(word," ",l_doc[i])
      tf_diz[word]=l_doc[i].count(word)
      cont.append(vec.count(word))
    bag_words.append(cont)
  return bag_words

bag_words_d1 = calculateBOW(coseno_d1,d1)
bag_words_d2 = calculateBOW(coseno_d2,d2)
bag_words_d3 = calculateBOW(coseno_d3,d3)
bag_words_d1 = np.array(bag_words_d1).reshape(len(d1),len(coseno_d1))
bag_words_d2 = np.array(bag_words_d2).reshape(len(d2),len(coseno_d2))
bag_words_d3 = np.array(bag_words_d3).reshape(len(d3),len(coseno_d3))

uniqueWords_d1 = set(coseno_d1)
uniqueWords_d2 = set(coseno_d2)
uniqueWords_d3 = set(coseno_d3)

numOfWords_d1 = dict.fromkeys(uniqueWords_d1, 0)
numOfWords_d2 = dict.fromkeys(uniqueWords_d2, 0)
numOfWords_d3 = dict.fromkeys(uniqueWords_d3, 0)

def prev_wtf(documentos,numOfWords,uniqueWords):
  num = []
  for i in range(len(documentos)):
    cont = 1
    for word in documentos[i]:
      if cont == 1:
        numOfWords = dict.fromkeys(uniqueWords, 0)
      else:
        numOfWords[word] += 1
        #print(i," [",cont,"/",len(documentos[i]),"] ",word," ",numOfWords[word])
      cont += 1
    num.append(numOfWords)
  return num

numOfWords_d1 = prev_wtf(d1,numOfWords_d1,uniqueWords_d1)
numOfWords_d2 = prev_wtf(d2,numOfWords_d2,uniqueWords_d2)
numOfWords_d3 = prev_wtf(d3,numOfWords_d3,uniqueWords_d3)

def computeTF(wordDict, bagOfWords):#wtf
    tfDict = {}
    tf = []
    for i in range(len(bagOfWords)):
      bagOfWordsCount = len(bagOfWords[i])
      vec = []
      for word, count in wordDict[i].items():
          #print(count,"/",bagOfWordsCount,"=",count / float(bagOfWordsCount))########
          vec.append(count / float(bagOfWordsCount))
      #print()
      tf.append(vec)
    return tf

tf_d1 = computeTF(numOfWords_d1, d1)
tf_d2 = computeTF(numOfWords_d2, d2)
tf_d3 = computeTF(numOfWords_d3, d3)

def computeIDF(documents):
  N = len(documents)
  vec = []
  idfs = []
  for i in range(N):
    for word, count in documents[i].items():
      if count != 0:
        vec.append(math.log(N / count))
        #print(N,"/",count,"=",math.log(N / count))
      else:
        #print(N,"/",count,"= 0")
        vec.append(0)
    #print()
    idfs.append(vec)
  return idfs

idfs_d1 = computeIDF(numOfWords_d1)
idfs_d2 = computeIDF(numOfWords_d2)
idfs_d3 = computeIDF(numOfWords_d3)

for i in range(len(idfs_d1)):
  r = len(idfs_d1[i])
  a = len(idfs_d1)
  d = int(r/a)

idfs_d1 = np.array(idfs_d1[0]).reshape(a,d)

idfs_d1[0].flatten().tolist()

def computeTFIDF(tfBagOfWords, idfs):
  tf_idf = []
  for i in range(len(tfBagOfWords)):
    tf_idf.append([x*y for x,y in zip(tfBagOfWords[i],idfs[i])])
  return tf_idf

tf_idf_d1 = computeTFIDF(tf_d1, idfs_d1)
tf_idf_d2 = computeTFIDF(tf_d2, idfs_d2)
tf_idf_d3 = computeTFIDF(tf_d3, idfs_d3)


#JACCARD
def jaccard(doc_m, doc_n):
    a = set(doc_m)
    b = set(doc_n)
    union = a.union(b)
    inter = a.intersection(b)
    
    if len(union) == 0:
        if len(inter) == 0:
            return 1

    similitud = len(inter) / len(union)
    return similitud

def Jaccard(Documentos):
  doc = []
  mat_doc = np.zeros(shape=(len(Documentos),len(Documentos)))
  for m in range(len(Documentos)):
    for n in range(len(Documentos)):
      j = jaccard(Documentos[m],Documentos[n])#enviamos al metodo jac
      mat_doc[m,n] = j
      doc.append(round(mat_doc[m,n], 4))
  return doc

ja_d1 = Jaccard(d1)#abstracts
a = ja_d1
ja_d2 = Jaccard(d2)#keywords
b = ja_d2

ja_d1 = np.array(ja_d1).reshape(len(d1),len(d1))
ja_d2 = np.array(ja_d2).reshape(len(d2),len(d2))

#tf_d1 = [[4,3],[4,3]]
def Modulo(tf):
  mod = []
  for i in range(len(tf)):
    mod.append(np.linalg.norm(tf[i]))
  return mod

mod_d1 = Modulo(tf_d1)
mod_d2 = Modulo(tf_d2)
mod_d3 = Modulo(tf_d3)

def Vec_Unitario(tf, mod):
  V_Uni = []
  for i in range(len(tf)):
    cont = []
    for j in range(len(tf[i])):
      cont.append(tf[i][j] / mod[i])
    V_Uni.append(cont)
  return V_Uni

V_Uni_d3 = Vec_Unitario(tf_d3, mod_d3)

def Coseno_Vec(tf):
  cos = []
  for i in range(len(tf)):
    result = []  
    for j in range(len(tf)):
      result.append(1 - spatial.distance.cosine(tf[i], tf[j]))
    cos.append(result)
  return cos

cos_d3 = Coseno_Vec(tf_d3)
c = np.ravel(cos_d3)
#c = cos_d3.flatten()
    
def Mul_Vec(doc,m):
  for i in range(len(doc)):
    t = doc[i]
    for j in range(len(t)):
      t[j] = (t[j] * m)
    doc[i] = t
  return doc

m_d1 = Mul_Vec(ja_d1,0.20)#jaccard d1
m_d2 = Mul_Vec(ja_d2,0.30)#jaccard d2
m_d3 = Mul_Vec(cos_d3,0.50)#coseno vectorial d3
m_d3 = np.array(m_d3).reshape(len(d3),len(d3))

def Similitudes(d1,d2,d3):
  similitud = []
  for i in range(len(d1)):
    v_sum = d1[i]
    for j in range(len(d1)):
      v_sum[j] = (d1[i][j] + d2[i][j] + d3[i][j])
    similitud.append(v_sum)
  return similitud

sim = Similitudes(m_d1,m_d2,m_d3)

#for i in range(len(sim)):
  #print(sim[i])
    
'''
sns.heatmap(sim)
plt.title("Mapa de Calor")
plt.show()
print()
cluster = linkage(sim, "ward")
dendogram = sch.dendrogram(cluster)
plt.title("Dendograma")
plt.show()
print()
m_d1 = m_d1.flatten()
m_d2 = m_d2.flatten()
m_d3 = m_d3.flatten()

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.scatter(
    x = m_d1,
    y = m_d2, 
    c = 'blue',
    marker    = 'o',
    edgecolor = 'black', 
)
ax.set_title('MDS - Titulos y Palabras Clave');

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.scatter(
    x = m_d1,
    y = m_d3, 
    c = 'green',
    marker    = 'o',
    edgecolor = 'black', 
)
ax.set_title('MDS - Titulos y Resumenes');

fig, ax = plt.subplots(1, 1, figsize=(6, 3.84))
ax.scatter(
    x = m_d2,
    y = m_d3, 
    c = 'red',
    marker    = 'o',
    edgecolor = 'black', 
)
ax.set_title('MDS - Palabras Clave y Resumenes');

d = {'title': a, 'keyword': b, 'abstract':c}
data  = pd.DataFrame(data=d)
data
plt.scatter(data['title'],data['keyword'],data['abstract'])
plt.xlim(-0.15,1.2)
plt.ylim(-0.15,1.2)
plt.show()
x = data.iloc[:,0:3] # 1t for rows and second for columns
kmeans = KMeans(5)
kmeans.fit(x)

identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['title'],data_with_clusters['abstract'],c=data_with_clusters['Clusters'],cmap='rainbow')
plt.title("Cluster")
'''
