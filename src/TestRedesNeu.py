import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sknn.mlp import Regressor, Layer

# Test point
Estacion_01 = pd.read_excel('PorvenirCompl.xlsx',sheet_name='Estacion_01',index_col=0)
Estacion_01 = Estacion_01.rename(columns={'Agregado': 'Est1'})
Estacion_02 = pd.read_excel('sanantonio.xlsx',sheet_name='Estacion_02',index_col=0)
Estacion_02 = Estacion_02.rename(columns={'Agregado': 'Est2'})
Estacion_03 = pd.read_excel('cunumbuque.xlsx',sheet_name='Estacion_03',index_col=0)
Estacion_03 = Estacion_03.rename(columns={'Agregado': 'Est3'})


TodasEstaciones = Estacion_01.resample('24H', how='sum')
# TodasEstaciones = Estacion_01.resample('24H').sum()
TodasEstaciones['Est2']=Estacion_02['Est2'].resample('24H', how='sum')
TodasEstaciones['Est3']=Estacion_03['Est3'].resample('24H', how='sum')
TodasEstaciones.head()


capasinicio = TodasEstaciones.ix['1983-08-02':'2014-04-30'].as_matrix()[:,[0,2]]
capasalida = TodasEstaciones.ix['1983-08-02':'2014-04-30'].as_matrix()[:,1]
neurones =  1000
tasaaprendizaje = 0.00001
numiteraciones = 9000

redneural = Regressor(
    layers=[
        Layer("ExpLin", units=neurones),
        Layer("ExpLin", units=neurones), Layer("Linear")],
    learning_rate=tasaaprendizaje,
    n_iter=numiteraciones)
redneural.fit(capasinicio, capasalida)


capasinicio1 = TodasEstaciones.ix['2010-01-01':'2010-12-31'].as_matrix()[:,[0,2]]
valor1 = ([])
for i in range(capasinicio1.shape[0]):
    prediccion = redneural.predict(np.array([capasinicio1[i,:].tolist()]))
    valor1.append(prediccion[0][0])



TodasEstaciones['Est2_Completed']=TodasEstaciones['Est2']
TodasEstaciones['Est2_Completed'].ix['2010-01-01':'2010-12-31']=valor1


fig, axs = plt.subplots(4, 1, sharex=True)

fig.subplots_adjust(hspace=0)
axs[0].plot(TodasEstaciones['Est1'].ix['1983-08-02':'2014-04-30'],label='PorvenirCompl')
axs[0].legend(loc=2)
axs[1].plot(TodasEstaciones['Est2'].ix['1983-08-02':'2014-04-30'],label='sanantonio',color='g')
axs[1].legend(loc=2)
axs[2].plot(TodasEstaciones['Est3'].ix['1983-08-02':'2014-04-30'],label='cunumbuque',color='orange')
axs[2].legend(loc=2)
axs[3].plot(TodasEstaciones['Est2_Completed'].ix['1983-08-02':'2014-04-30'],label='sanantonioCompletado',color='firebrick')
axs[3].legend(loc=2)


plt.show()

writer = pd.ExcelWriter('sanantonioComplTest.xlsx')
TodasEstaciones['Est2_Completed'].to_excel(writer,'Sheet1')
writer.save()
