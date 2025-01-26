# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:32:50 2023

@author: javie
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Ecuación de Lane-Emden
def lane_emden(xi, u, n):
    theta, dtheta_dxi = u
    dudxi = [dtheta_dxi, -2 / xi * dtheta_dxi - theta**n]
    return dudxi

def end(xi,u,n):
    return u[0]

# Valores de n (no está puesto el n=0 para que no salte error al calcular Bn)
n_values = [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]

# Intervalo de integración
xi_span = (1e-6, 33) #Empiezo en 1e-6 para evitar dividir entre 0

# Condiciones iniciales
theta0 = 1
u0 = 0
end.terminal = True

plt.figure()
color_map = cm.get_cmap('spring', len(n_values))

for i, n in enumerate(n_values):
    color = color_map(i)
    sol = solve_ivp(lane_emden, xi_span, [theta0, u0], args=(n,), t_eval=np.linspace(xi_span[0], xi_span[1], 1000), events=(end))
    xi = sol.t
    theta, dtheta_dxi = sol.y
    
    plt.plot(xi, theta, color=color, label=f'n = {n}')
    
    Rn= xi[-1]
    Mn = -(xi[-1]**2)*dtheta_dxi[-1]
    Dn = -((3/xi[-1])*dtheta_dxi[-1])**-1
    a= (3*Dn)**((3-n)/(3*n))
    b= (n+1)*Mn**((n-1)/n)*Rn**((3-n)/n)
    Bn = a/b
    print(f'n = {n}: Dn = {Dn}, Mn = {Mn}, Rn = {Rn}, Bn = {Bn}')
   
plt.xlabel(r'$\xi$')
plt.ylabel(r'$\theta(\xi)$')
plt.title('Soluciones de la Ecuación de Lane-Emden')
plt.grid()
plt.legend()
plt.show()
