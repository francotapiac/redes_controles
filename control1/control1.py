import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

#Datos:
N = 100
t = np.linspace(-5,5,200)
A = 1
pi = 3.14159265359
a0 = A/2

 #Funciones:
f1 = 2*A*(1 + t)
f2 = A
f3 = 2*A*(1-t)

#Gráfico serie de fourier
fs = [((6*A/(n**2 * pi**2))*((np.cos(n*pi/3))-(np.cos(2*n*pi/3)))) * np.cos(2*n*pi*t/3) for n in range(1, N+1, 2)]
f = np.sum(fs, axis=0) + a0

#Gráfico espectro
F = np.array([((6*A/(n**2 * pi**2))*((np.cos(n*pi/3))-(np.cos(2*n*pi/3))))/2 for n in range(-N-1, N+1) if n%2 != 0 ])
fn = [n for n in range(-N-1, N+1) if n%2 != 0]

#Obtención error cuadrático medio

plt.figure()
plt.stem(fn, abs(F), use_line_collection=True)
plt.title("Espectro de la señal")
plt.xlabel("n$\omega_0$ rad/s ($\omega_0=\pi$)")
plt.ylabel("|$F_n$|")

plt.figure()
plt.plot(t, f)
plt.title("Serie trigonometrica de Fourier")
plt.xlabel("t (s)")
plt.ylabel("f(t)")

plt.grid()
plt.show()