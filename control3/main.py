import numpy as np
from numpy import cos, sin, pi
import matplotlib.pyplot as plt

#a. Generar señal de ejmplo f(t):
def signal(t):
    return cos(2*pi*3*t)+sin(2*pi*2*t)

# Graficando función f(t)
t = np.arange(0, 1, 1/500)
f = signal(t)
plt.plot(t, f)
plt.grid(True)
plt.title("f(t) = cos(2*pi*3*t)+sin(2*pi*2*t)")
plt.xlabel("time (s)")
plt.show()




#b. Frecuencia de muestreo correcta y señal muestreada fs
# Función de muestreo
def sinc(t, compress=1):
    return sin(compress*pi*t)/(compress*pi*t)

# Graficando función de muestreo
Fs = 6     #Frecuencia de muestreo = 6 (para error < 10%, Fs = 7)
tn = np.arange(0, 1.1, 1/Fs)
fs = signal(tn)
plt.figure()
plt.plot(t, f, '--')
plt.stem(tn, fs, 'r', markerfmt='C3o', use_line_collection=True)
plt.grid(True)
plt.title("fs(n) = cos(2*pi*3*n)+sin(2*pi*2*n)")
plt.xlabel("time (s)")
plt.show()

#c. Reconstruyendo f(t) a partir de fs
tf = np.arange(-5, 5, 1/500)
sinc_n = -1*sinc((tf-2))

plt.figure()
for i,s in zip(tn,fs):
  sinc_s = s*sinc((tf-i),Fs)
  

plt.plot(tf,sinc_n)
plt.stem(tn,fs, 'r', markerfmt='C3o', use_line_collection=True)
plt.xlim([-1, 2])
plt.grid(True)
plt.title("fi*sinc(t-ti)")
plt.xlabel("time (s)")
plt.show()


# d. Comparando señal original con señal reconstruida
sincs = []
tf = np.arange(1e-10,1,1/500)
plt.figure()

for s,n in zip(fs,tn):
  sinc_n = s*sinc(Fs*(tf-n))
  sincs.append(sinc_n)
sincs = np.array(sincs)
sincs = np.sum(sincs, axis=0)
plt.plot(t, f)
plt.plot(tf, sincs, 'r--')
plt.xlim([0, 1])
plt.grid(True)
plt.title("SUM")
plt.xlabel("time (s)")
plt.show()

# Obteniendo porcentaje de error
rms = 0
n = len(f)
for i in range(n):
    rms = rms + (sincs[i] - f[i])**2
rms = (rms/n)**(1/2)
print(rms)

#Con frecuencia de muestreo 7, el error es < 10%.



