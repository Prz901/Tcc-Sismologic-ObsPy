#Inicializa um objeto client 
import numpy as np
import math
from obspy.clients.fdsn import Client
client = Client("IRIS")
from obspy.clients.iris import Client
from obspy import UTCDateTime
from sklearn import svm
from sklearn.metrics import confusion_matrix


dt = UTCDateTime("2014-04-18T09:28:00.000")
kt = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
tr1 = abs(kt[0].data)
terremoto=tr1

dt = UTCDateTime("2010-02-27T06:45:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto1 = abs(st[0].data)

dt = UTCDateTime("2018-02-16T17:39:00.000")
lt = client.get_waveforms("IU", "ANMO", "00", "LHZ", dt, dt + 60 * 60)
terremoto4 = abs(lt[0].data)

dt = UTCDateTime("2019-11-05T18:01:00.000")
ht = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto5 = abs(st[0].data)

dt = UTCDateTime("2017-09-07T23:49:00.000")
rt = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto6 = abs(rt[0].data)

dt = UTCDateTime("2010-04-04T15:40:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto7 = abs(rt[0].data)


dt = UTCDateTime("2012-04-02T12:44:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto8 = abs(rt[0].data)

dt = UTCDateTime("2014-04-18T14:27:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto9 = abs(rt[0].data)

dt = UTCDateTime("2017-09-19T18:14:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto10 = abs(rt[0].data)


dt = UTCDateTime("2018-02-16T17:39:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremoto11 = abs(rt[0].data)


# Terremoto para as bases
# 1
dt = UTCDateTime("2019-09-29T18:28:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremotoTeste = abs(rt[0].data)


# 2
dt = UTCDateTime("2019-09-26T16:36:00.000")
st = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
terremotoTeste1 = abs(rt[0].data)



t = UTCDateTime("2019-01-02T06:45:00.000")
at = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto = at[0].data

t = UTCDateTime("2019-01-03T06:45:00.000")
vt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto1 = vt[0].data


t = UTCDateTime("2019-01-04T06:45:00.000")
tt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto2 = tt[0].data


t = UTCDateTime("2019-01-06T06:45:00.000")
ot = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto3 = ot[0].data


t = UTCDateTime("2019-01-05T06:45:00.000")
tv = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto4 = tv[0].data


t = UTCDateTime("2019-01-07T06:45:00.000")
qt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto5 = st[0].data

t = UTCDateTime("2019-01-08T06:45:00.000")
wt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto6 = st[0].data

t = UTCDateTime("2019-01-09T06:45:00.000")
yt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto7 = st[0].data


t = UTCDateTime("2019-01-10T06:45:00.000")
ut = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto8 = ut[0].data

t = UTCDateTime("2019-01-11T06:45:00.000")
ut = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto9 = ut[0].data

dt = UTCDateTime("2017-08-02T03:15:00.000")
zt = Client().timeseries("IU", "ANMO", "00", "BHZ", dt, dt+10)
naoTerremoto10 = abs(zt[0].data)

# Nao terremotos para a base de dados 
t = UTCDateTime("2019-01-15T06:45:00.000")
tt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto13 = tt[0].data

t = UTCDateTime("2019-01-16T06:45:00.000")
tt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto14 = tt[0].data

t = UTCDateTime("2019-01-17T06:45:00.000")
tt = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
Naoterremoto15 = tt[0].data

svc = svm.SVC()

#ondas sísmicas
sismo1=terremoto[0:199]
sismo2=terremoto1[0:199]
sismo3=terremoto4[0:199]
sismo4=terremoto5[0:199]
sismo5=terremoto6[0:199]
sismo6=terremoto7[0:199]
sismo7=terremoto8[0:199]
sismo8=terremoto9[0:199]
sismo9=terremoto10[0:199]
sismo10=terremoto11[0:199]

naoSismo=Naoterremoto[0:199]
naoSismo2=Naoterremoto1[0:199]
naoSismo3=Naoterremoto2[0:199]
naoSismo4=Naoterremoto3[0:199]
naoSismo5=Naoterremoto4[0:199]
naoSismo6=Naoterremoto5[0:199]
naoSismo7=Naoterremoto6[0:199]
naoSismo8=Naoterremoto7[0:199]
naoSismo9=Naoterremoto8[0:199]
naoSismo10=Naoterremoto9[0:199]

#conjunto de treinamento
X=[sismo1,sismo2,sismo3,sismo4,sismo5,sismo6,sismo7,sismo8,sismo9,sismo10,naoSismo,naoSismo2,naoSismo3,naoSismo4,naoSismo5,naoSismo6,naoSismo7,naoSismo8,naoSismo9,naoSismo10]
Y=[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

#treinamento da máquina
svc.fit(X,Y)

# verificação do treinamento
print(svc.predict([sismo1]))   #classe 1 - terremoto
print(svc.predict([naoSismo2]))   #classe 0 - não é terremoto

#aplicação dos sismos de teste
arrayDaIa = []
sismoTesteTerremoto=terremotoTeste[0:199]
print(svc.predict([sismoTesteTerremoto]))
a = svc.predict([sismoTesteTerremoto])
arrayDaIa.append(a)

sismoTesteTerremoto1=terremotoTeste1[0:199]
print(svc.predict([sismoTesteTerremoto1]))
b = svc.predict([sismoTesteTerremoto1])
arrayDaIa.append(b)

sismoTesteNaoTerremoto=Naoterremoto13[0:199]
print(svc.predict([sismoTesteNaoTerremoto]))
c =svc.predict([sismoTesteNaoTerremoto])
arrayDaIa.append(c)

sismoTesteNaoTerremoto1=Naoterremoto14[0:199]
print(svc.predict([sismoTesteNaoTerremoto1]))
d =svc.predict([sismoTesteNaoTerremoto])
arrayDaIa.append(d)

sismoTesteNaoTerremoto2=Naoterremoto15[0:199]
print(svc.predict([sismoTesteNaoTerremoto2]))
e = svc.predict([sismoTesteNaoTerremoto2])
arrayDaIa.append(e)

y_true = [1,1,0,0,0]
y_pred =arrayDaIa

print(confusion_matrix(y_true, y_pred, labels=[1, 0]))

#Amostra do terremoto no mexico -------- 1)

# t = UTCDateTime("2010-02-27T06:45:00.000")
# st = client.get_waveforms("IU", "ANMO", "00", "LHZ", t, t + 60 * 60)
# st.plot()

#todas as amostras de nao terremotos