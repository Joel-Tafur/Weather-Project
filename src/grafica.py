import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sknn.mlp import Regressor, Layer


Estacion_01 = pd.read_excel('sanantonioCompl.xlsx',sheet_name='Estacion_03',index_col=0)
Estacion_01 = Estacion_01.rename(columns={'Agregado': 'Est1'})
Estacion_02 = pd.read_excel('sanantonioComplTest.xlsx',sheet_name='Estacion_02',index_col=0)
Estacion_02 = Estacion_02.rename(columns={'Agregado': 'Est2'})


fig, axs = plt.subplots(1, 1, sharex=True)

plt.plot(Estacion_01.ix['2010-01-01':'2010-12-31'],label='observado',color='yellowgreen')
plt.plot(Estacion_02.ix['2010-01-01':'2010-12-31']-1.344202,label='pronosticado',color='peru')
plt.legend(loc=2)

plt.show()
