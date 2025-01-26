import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#------------------Parte 0: Definición de constantes, variables y más-------------------------
# Valores de las constantes
rho_0 = 0.01
P_0 = 1
gamma = 5/3
xi = 0.
xf = 2*np.pi
phi = 0
k = 1
N = 400
L = 2*np.pi
dx = L/N
A = 1e-4
c_s0 = np.sqrt(gamma*P_0/rho_0)
v_0 = 0
G=1

# Variables
h=dx
x = np.arange(xi - dx/2, L + 3*dx/2, dx)

rho = rho_0 + A*np.cos(k*x)
P = P_0 + gamma*A*np.cos(k*x)  #Modo positivo y negativo
#P = P_0 #Modo nulo
c_s = np.sqrt(gamma*P/rho)
v = v_0 + c_s*A*np.cos(k*x)  #Modo positivo
#v = v_0 - c_s*A*np.cos(k*x) #Modo negativo
#v = v_0 #Modo nulo
u = P/(rho*(gamma-1))
e = u + 0.5*v**2

# Energías internas
u_m = rho 
u_v = rho * v
u_e = rho * e
# Flujos
f1= rho * v
f2= rho * v*v + P
f3= (rho*e + P)*v

# Paso de tiempo
dt = 0.1 * dx / (max(np.max(c_s+v),np.max(c_s-v)))
dt_show = 0.02
t = 0
next_show = dt_show + t

#------------------Parte 1: Potencial Gravitatorio-------------------------

def fourier(rho): 
    rho_k = np.fft.fft(rho)
    k = 2*np.pi*np.fft.fftfreq(len(rho), dx)
    phi_k = np.zeros(len(rho_k))
    phi_k[1:] = -4*np.pi*G*rho_k[1:]/(k[1:])**2
    phi = np.real(np.fft.ifft(phi_k))
    return phi 

def analitica(x):
    return 4*np.pi*G*(-A*np.cos(x))


plt.figure()
plt.plot(x, fourier(rho), label='Solución numérica', color='blue')
plt.plot(x, analitica(x), label='Solución analítica', color='red', linestyle='--')
plt.xlabel('x')
plt.ylabel(r'$\phi$')
plt.legend()
plt.grid(True)
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

#------------------Parte 2: Esquemas LF y LWR-------------------------

def Df_Forward(x, y, h):
    y_prim = (y[1:] - y[:-1]) / h
    return y_prim

def Df_Centered(x, y, h):
    y_prim = (y[2:] - y[:-2]) / (2 * h)
    return y_prim

def Df_Backward(x, y, h):
    y_prim = (y[:-1] - y[1:]) / (-h)
    return y_prim

def lf(i):
    global t, next_show, dt, u_m, u_v, u_e, rho, v, e, P
    while t < next_show:
        c_s = np.sqrt(gamma*P/rho)
        dt = 0.9 * dx / (max(np.max(c_s+v),np.max(c_s-v)))
        u_m = rho 
        u_v = rho * v
        u_e = rho * e
        
        u = e - 0.5*v**2
        
        f1= rho * v
        f2= rho * v*v + P
        f3= (rho*e + P)*v
        #Gravedad
        rho2= rho[1:-1]*Df_Centered(x, fourier(rho), h)*dt
        rho3= rho[1:-1]*v[1:-1]*Df_Centered(x, fourier(rho), h)*dt
        #Esquema LF
        u_m[1:-1] = 0.5 * (u_m[2:] + u_m[:-2]) - dt/(2*dx) * (f1[2:]-f1[:-2])
        u_v[1:-1] = 0.5 * (u_v[2:] + u_v[:-2]) - dt/(2*dx) * (f2[2:]-f2[:-2])
        u_e[1:-1] = 0.5 * (u_e[2:] + u_e[:-2]) - dt/(2*dx) * (f3[2:]-f3[:-2])
        #Euler
        u_v[1:-1] = u_v[1:-1] + rho2
        u_e[1:-1] = u_e[1:-1] + rho3
        # Condiciones de contorno periódicas
        u_m[-1] = u_m[1]
        u_m[0] = u_m[-2]
        u_v[-1] = u_v[1]
        u_v[0] = u_v[-2]
        u_e[-1] = u_e[1]
        u_e[0] = u_e[-2]
        
        rho = u_m
        v = u_v/rho
        e = u_e/rho
        u = e - 0.5*v**2
        P = u*rho*(gamma-1)
        
       
        t += dt

    next_show = t + dt_show

    ax1.set_title(r"$t=%.3f$" % t)
    line1.set_data(x, rho)
    line2.set_data(x, v)  
    line3.set_data(x, P)  
    
def lwr(i):
    global t, next_show, dt, u_m, u_v, u_e, rho, v, e, P
    while t < next_show:
        c_s = np.sqrt(gamma*P/rho)
        dt = 0.9 * dx / (max(np.max(c_s+v),np.max(c_s-v)))
        u_m = rho 
        u_v = rho * v
        u_e = rho * e
        
        u = e - 0.5*v**2
        
        f1= rho * v
        f2= rho * v*v + P
        f3= (rho*e + P)*v
        #Gravedad
        rho2= rho[1:-1]*Df_Centered(x, fourier(rho), h)*dt
        rho3= rho[1:-1]*v[1:-1]*Df_Centered(x, fourier(rho), h)*dt
        #Esquema LWR Paso 1:
        #Terminos i+1/2    
        u_m2 = 0.5 * (u_m[1:-1] + u_m[2:]) - dt/(2*dx) * (f1[2:]-f1[1:-1])
        u_v2 = 0.5 * (u_v[1:-1] + u_v[2:]) - dt/(2*dx) * (f2[2:]-f2[1:-1])
        u_e2 = 0.5 * (u_e[1:-1] + u_e[2:]) - dt/(2*dx) * (f3[2:]-f3[1:-1])
        #Terminos i-1/2 
        u_m22 = 0.5 * (u_m[:-2] + u_m[1:-1]) - dt/(2*dx) * (f1[1:-1]-f1[:-2])
        u_v22 = 0.5 * (u_v[:-2] + u_v[1:-1]) - dt/(2*dx) * (f2[1:-1]-f2[:-2])
        u_e22 = 0.5 * (u_e[:-2] + u_e[1:-1]) - dt/(2*dx) * (f3[1:-1]-f3[:-2])
        
        #Calcular los flujos
        #Terminos i+1/2 
        f11 = u_v2
        f21 = u_v2 * u_v2/u_m2 + (u_e2/u_m2 - 0.5*(u_v2/u_m2)**2)*u_m2*(gamma-1)
        f31 = (u_m2*u_e2/u_m2 + (u_e2/u_m2 - 0.5*(u_v2/u_m2)**2)*u_m2*(gamma-1))*u_v2/u_m2
        #Terminos i-1/2 
        f12 = u_v22
        f22 = u_v22 * u_v22/u_m22 + (u_e22/u_m22 - 0.5*(u_v22/u_m22)**2)*u_m22*(gamma-1)
        f32 = (u_m22*u_e22/u_m22 + (u_e22/u_m22 - 0.5*(u_v22/u_m22)**2)*u_m22*(gamma-1))*u_v22/u_m22
        
        #Esquema LWR Paso 2:
        u_m[1:-1] = u_m[1:-1] - dt/(dx) * (f11-f12)
        u_v[1:-1] = u_v[1:-1] - dt/(dx) * (f21-f22)
        u_e[1:-1] = u_e[1:-1] - dt/(dx) * (f31-f32)
        #Euler
        u_v[1:-1] = u_v[1:-1] + rho2
        u_e[1:-1] = u_e[1:-1] + rho3
        # Condiciones de contorno periódicas
        u_m[-1] = u_m[1]
        u_m[0] = u_m[-2]
        u_v[-1] = u_v[1]
        u_v[0] = u_v[-2]
        u_e[-1] = u_e[1]
        u_e[0] = u_e[-2]
        
        rho = u_m
        v = u_v/rho
        e = u_e/rho
        u = e - 0.5*v**2
        P = u*rho*(gamma-1)
        
       
        t += dt

    next_show = t + dt_show

    ax1.set_title(r"$t=%.3f$" % t)
    line1.set_data(x, rho)
    line2.set_data(x, v)  
    line3.set_data(x, P)  
    

ani = animation.FuncAnimation(fig, lf, frames=None, interval=100, blit=False, cache_frame_data=False)


line1, = ax1.plot([], [], label='rho')
line2, = ax2.plot([], [], label='v')
line3, = ax3.plot([], [], label='P')

ax1.set_xlim(xi, xf)
ax1.set_ylim(0, 1.)
ax1.set_xlabel(r'$x$')
ax1.set_ylabel(r'$\rho$')

ax2.set_xlim(xi, xf)
ax2.set_xlabel(r'$x$')
ax2.set_ylabel(r'$v$')

ax3.set_xlim(xi, xf)
ax3.set_xlabel(r'$x$')
ax3.set_ylabel(r'$P$')
ax1.set_ylim(1.9998, 2.0002)
ax2.set_ylim(-0.001, 0.001)
ax3.set_ylim(0.9995, 1.0005)
plt.show()


plt.figure()

#------------------Parte 3: Plot rhomax-------------------------


#Valores de P y rho
P_0_values = [1, 10]
rho_0 = 1

for P_0 in P_0_values:
    #Configuracion inicial
    P = P_0 + gamma*A*np.cos(k*x)  
    rho = rho_0 + A*np.cos(k*x)
    c_s = np.sqrt(gamma*P/rho)
    v = v_0 + c_s*A*np.cos(k*x)
    e = P/(rho*(gamma-1)) + 0.5*v**2
    
    #Paso de tiempo
    dt = 0.1 * dx / (max(np.max(c_s+v),np.max(c_s-v)))
    t = 0
    rhomax = []
    time = []

    #Bucle con metodo LF
    while t < 10:
        c_s = np.sqrt(gamma*P/rho)
        dt = 0.9 * dx / (max(np.max(c_s+v),np.max(c_s-v)))

        # Actualizar variables
        u_m = rho 
        u_v = rho * v
        u_e = rho * e
        f1= rho * v
        f2= rho * v*v + P
        f3= (rho*e + P)*v

        # Gravedad
        rho2= rho[1:-1]*Df_Centered(x, fourier(rho), h)*dt
        rho3= rho[1:-1]*v[1:-1]*Df_Centered(x, fourier(rho), h)*dt

        #Esquema LF
        u_m[1:-1] = 0.5 * (u_m[2:] + u_m[:-2]) - dt/(2*dx) * (f1[2:]-f1[:-2])
        u_v[1:-1] = 0.5 * (u_v[2:] + u_v[:-2]) - dt/(2*dx) * (f2[2:]-f2[:-2])
        u_e[1:-1] = 0.5 * (u_e[2:] + u_e[:-2]) - dt/(2*dx) * (f3[2:]-f3[:-2])

        # Euler
        u_v[1:-1] = u_v[1:-1] + rho2
        u_e[1:-1] = u_e[1:-1] + rho3

        # Condiciones de contorno periódicas
        u_m[-1] = u_m[1]
        u_m[0] = u_m[-2]
        u_v[-1] = u_v[1]
        u_v[0] = u_v[-2]
        u_e[-1] = u_e[1]
        u_e[0] = u_e[-2]

        rho = u_m
        v = u_v/rho
        e = u_e/rho
        u = e - 0.5*v**2
        P = u*rho*(gamma-1)

        rhomax.append(np.max(rho))
        time.append(t)
        t += dt

    plt.plot(time, rhomax, label=f'P_0={P_0}')

plt.xlabel('Tiempo')
plt.ylabel('Valor máximo de rho')
plt.legend()
plt.show()
