import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

mov_num = "01"
x = range(1, 64, 3)
y = range(2, 64, 3)
z = range(3, 64, 3)

#FINGERTAPPING
filename = ('./mov_csv/fingertap_' + mov_num + '.csv')
fingertap = genfromtxt(filename, delimiter=',')
permutation = [0, *x, *y, *z]
fingertap = fingertap[:, permutation]
fingertap[:, 0] = fingertap[:, 0]/1000

plt.figure(1)
plt.plot(fingertap[:, 0], fingertap[:, 26])
plt.plot(fingertap[:, 0], fingertap[:, 30])
plt.show()

#PRONOSUP
filename = ('./mov_csv/pronosup_' + mov_num + '.csv')
pronosup = genfromtxt(filename, delimiter=',')
permutation = [0, *x, *y, *z]
pronosup = pronosup[:, permutation]
pronosup[:, 0] = pronosup[:, 0]/1000

plt.figure(2)
plt.plot(pronosup[:, 0], pronosup[:, 5])
plt.plot(pronosup[:, 0], pronosup[:, 21])
plt.show()

#FIST OPEN CLOSE
filename = ('./mov_csv/fist_' + mov_num + '.csv')
fist = genfromtxt(filename, delimiter=',')
permutation = [0, *x, *y, *z]
fist = fist[:, permutation]
fist[:, 0] = fist[:, 0]/1000

plt.figure(3)
plt.plot(fist[:, 0], fist[:, 26])
plt.plot(fist[:, 0], fist[:, 30])
plt.plot(fist[:, 0], fist[:, 34])
plt.plot(fist[:, 0], fist[:, 38])
plt.plot(fist[:, 0], fist[:, 42])
plt.show()



