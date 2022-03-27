import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel, ExpSineSquared, Matern, RationalQuadratic, DotProduct
import numpy as np

def func(x,y):
    return -np.exp(-(x-3)**2/5-(y-6)**2/5)-1.5*np.exp(-(x-6)**2/3-(y-8)**2/3)-2*np.exp(-(x-8)**2/4-(y-3)**2/4)-np.exp(-(x-4)**2/6-(y-2)**2/6)
X = []
Y = []
Z = []
xm,ym,zm = 0,0,0
for i in range(101):
    for j in range(101):
        x,y = i/10,j/10
        z = func(x, y)
        X.append(x)
        Y.append(y)
        Z.append(z)
        if  zm > z:
            xm = x
            ym = y
            zm = z

fig = plt.figure(figsize=(10,5))
ax1 = fig.add_subplot(1,2,1,projection='3d')
ax1.scatter(X,Y,Z,s=10,c='red')
ax2 = fig.add_subplot(1,2,2)
mappable = ax2.scatter(X,Y,c=Z,cmap='jet')
fig.colorbar(mappable,ax=ax2)
ax2.scatter(xm,ym,c='black',marker='x')
ax2.text(xm,ym,'min='+str(zm))
ax2.set_xlim(0,10)
ax2.set_ylim(0,10)
fig.savefig("imgs/img_func.png")
plt.show()

def evalf(pred,std,n):
    return -pred+std*n**(1/2)

xp = []
yp = []
for i in range(101):
    for j in range(101):
        x0,y0 = i/10,j/10
        xp.append(x0)
        yp.append(y0)
Gxy = np.array([xp,yp]).T
Xr = []
Yr = []
Zr = []
p = (1,1)
for i in range(40):
    x,y = p
    z = func(x,y)
    Xr.append(x)
    Yr.append(y)
    Zr.append(z)
    GP = GaussianProcessRegressor(kernel=ConstantKernel() * RBF() + WhiteKernel() + DotProduct(),alpha=0)
    GP.fit(np.array([Xr,Yr]).T,np.array(Zr))
    pp = []

    predz,std = GP.predict(Gxy,return_std=True)
    pp = evalf(predz,std,len(Xr))
    ind = np.argmax(pp)
    p = (Gxy[ind][0],Gxy[ind][1])
    fig = plt.figure(figsize=(11,5))
    ax1 = fig.add_subplot(1,2,1)
    mappable = ax1.scatter(Xr,Yr,c=Zr,cmap='jet')
    ax1.scatter(x,y,c='black',marker='x')
    ax1.set_xlim(0,10)
    ax1.set_ylim(0,10)
    fig.colorbar(mappable,ax=ax1)
    ax2 = fig.add_subplot(1,2,2)
    mappable = ax2.scatter(Gxy[:,0],Gxy[:,1],c=pp,cmap='jet')
    ax2.scatter(Gxy[ind][0],Gxy[ind][1],c='black',marker='x')
    fig.colorbar(mappable,ax=ax2)
    ax2.set_xlim(0,10)
    ax2.set_ylim(0,10)
    fig.savefig("imgs/img_calu{}.png".format(i+1))
    plt.show()

predz,std = GP.predict(Gxy,return_std=True)
ind = np.argmin(predz)
fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,1,1)
mappable = ax1.scatter(Gxy[:,0],Gxy[:,1],c=predz,cmap='jet')
ax1.scatter(Gxy[ind][0],Gxy[ind][1],c='black',marker='x')
ax1.text(Gxy[ind][0],Gxy[ind][1],'min='+str(predz[ind]))
ax1.set_xlim(0,10)
ax1.set_ylim(0,10)
fig.colorbar(mappable,ax=ax1)
fig.savefig("imgs/img_end.png")
plt.show()