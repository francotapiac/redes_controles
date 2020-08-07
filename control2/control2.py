import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fourier


#Pregunta 1

#Parte B) Gráfica Transformada de Fourier

#Variables para graficar
A = 5
w = np.linspace(-140, 140,70)
Fw = A*((5 - 4*np.exp(w*1j) -  np.exp(-4*w*1j))/(4*w**2))
t = np.linspace(-1, 4, 100)

#Gráfico de Transformada
plt.plot(w,abs(Fw))
plt.title("Transformada de f(t) con A = " + str(A))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.grid()
plt.savefig("transformadaFourier.png")
plt.show()

#Calcular energia para comprobar Parseval
energia_t = 5*(A**2)/3
print("Energía respecto a t es " + str(energia_t))

energia_w = np.trapz(abs(Fw**2),w)
print("Energía respecto a w es " + str(energia_w))


#Pregunta 2

#Parta a) Calcular la convolución

#Funciones para graficar f(t) y g(t)

#Función g(t)
def tri(t):
    if 0 <= t <=3:
        return t/3
    else:
        return 0

#Función(t)
def squa(t):
    if -1 <= t <= 1:
        return 1
    else:
        return 0

#Calculo de la convolución
tau = np.linspace(-5, 5, 100)
c = []
for t in tau:
  g = np.array(list(map(tri, t-tau)))
  f = np.array(list(map(squa, tau)))
  c.append(np.trapz(f*g,tau))

plt.plot(tau,c)
plt.title("Convolución entre f(t) y g(t)")
plt.xlabel('tiempo (s)')
plt.ylabel('f(t)*g(t)')
plt.grid()
plt.savefig("convolucion.png")
plt.show()

#Parte b)

#Se obtienen funciones f(t) y g(t)
t = np.linspace(-5,5,100)
g = np.array(list(map(tri, t)))
f = np.array(list(map(squa, t)))

#Gráfico de ambas funciones respecto al tiempo
plt.plot(t,f,color="red",label="f(t)")
plt.plot(t,g,color="blue",label="g(t)")
plt.title("Función f(t) y g(t)")
plt.xlabel('tiempo (s)')
plt.ylabel('f(t) y g(t)')
plt.grid()
plt.legend()
plt.savefig("funciones.png")

plt.show()

#Calculo de la transformada de fourier de ambas funciones
Fw = fourier.fft(f,norm="ortho")
Fw = fourier.fftshift(Fw)


Gw = fourier.fft(g,norm="ortho")
Gw = fourier.fftshift(Gw)

#Calculo de las frencuencias de ambas funciones
freq_F = fourier.fftfreq(len(f),2)
freq_F = fourier.fftshift(freq_F)

freq_G = fourier.fftfreq(len(g),3)
freq_G = fourier.fftshift(freq_G)

#Gráfico de ambas transformadas
plt.plot(freq_F,abs(Fw))
plt.title("Transformada de f(t)")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.savefig("transformada_f.png")
plt.grid()
plt.show()

plt.plot(freq_G,abs(Gw))
plt.title("Transformada de g(t)")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.savefig("transformada_g.png")
plt.grid()
plt.show()

#Multpiplicación de transformadas Fw y Gw
multiplicion = Fw*Gw
freq_mult = fourier.fftfreq(multiplicion.size)
freq_mult = fourier.fftshift(freq_mult)
#Gráfico de de la multplicación
plt.plot(freq_mult,abs(multiplicion))
plt.title("Multiplicación de transformada F(w) y G(w)")
plt.xlabel('Frecuencia(Hertz)')
plt.ylabel('Amplitud')
plt.savefig("multiplicacion_transformada.png")
plt.grid()
plt.show()

#Obtención de la inversa de la multiplicación y gráfico
inversa = fourier.ifft(multiplicion,norm="ortho")
inversa = fourier.ifftshift(inversa)
plt.plot(t,abs(inversa))
plt.title("Transformada Inversa de multiplicación entre F(w) y G(w)")
plt.xlabel('Tiempo(s)')
plt.ylabel('Inversa(F(w)·G(w))')
plt.grid()
plt.savefig("inversa.png")
plt.show()