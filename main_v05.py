from classes.neural_v05 import NeuralNetwork
import matplotlib.pyplot as plt
import numpy as np

start = -2
end = 3
step = 0.05
x = np.arange(start,end+step,step) # Vector from 0 - 1 
N = x.shape[0]
a, b, c = 1, 2, 3
noise = np.random.normal(0.2,0.4,(N,))
y = a*np.power(x,2) + b*x + c + 0*noise
ne = 1
nm = 3
num_out = 1
layers = 1 
bias = True
act_fun = "sigmoid_1"

nn = NeuralNetwork(ne, x, num_out, y, layers, nm, bias, act_fun)

learn_rate = 0.1
max_iter = 10000

yd, J, Iter =nn.execute(learn_rate,max_iter)

plt.figure(1)
plt.plot(x,y,'*',x,yd,'-r')
plt.figure(2)
plt.plot(np.arange(Iter),J)
plt.show()