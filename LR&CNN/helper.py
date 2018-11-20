import numpy as np
import matplotlib
import matplotlib.pyplot as plt

x1 =[0]
y1 =[0]
x2 =[-1,1]
y2 =[i * i for i in x2]

x22 =[-1,1]
y22 =[i * 0 for i in x2]

x3 = np.linspace(-5,5,0.001)
y3 = x3 - x3

x4 = [-np.sqrt(2)/2,np.sqrt(2)/2]
y4 = [0.5,0.5]

x44 = [-np.sqrt(2)/2,np.sqrt(2)/2]
y44 = [0,0]
plt.close()
plt.figure(figsize=(8,4))
plt.scatter(x1,y1,color="red",linewidth=2 )
plt.scatter(x2,y2,color="blue",linewidth=2 )
plt.scatter(x4,y4,color="orange",linewidth=2 )
plt.scatter(x44,y44,color="orange",linewidth=2 )
plt.scatter(x22,y22,color="blue",linewidth=2 )
plt.plot([-3,3],[0.5,0.5])
plt.xlabel("x")
plt.ylabel("x^2")
plt.show()