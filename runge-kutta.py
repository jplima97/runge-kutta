import numpy as np
import matplotlib.pyplot as plt

T = 31400
g = 9.81 
vector_state_0 = np.array([0., 0.]) # (space,velocity)

def df(x, t): 
    w = 13500 - 360*t
    return np.array([x[1], (g/w)*(T - w - 0.036*g*(x[1]**2))])

t0 = 0
tf = 3
dt = 0.05

# function that creates steps for t
t = np.arange(t0, tf, dt) 

# Array that receives the state vector with the initial conditions for each time
x = np.zeros((vector_state_0.size, t.size))
for k in range(t.size-1):
    K1 = df(x[:,k], t[k])
    K2 = df(x[:,k] + dt*K1/2, t[k] + dt/2)
    K3 = df(x[:,k] + dt*K2/2, t[k] + dt/2)    
    K4 = df(x[:,k] + dt*K3, t[k] + dt)
    dx = (K1 + 2*K2 + 2*K3 + K4)*dt/6
    x[:,k+1] = x[:,k] + dx

a = np.zeros(t.size)
for i in range(t.size):
    a[i] = df(x[:,i], t[i])[1]

plt.plot(t,x[0],color='black',label='y(t)')
plt.grid()
plt.title('Space slope y(t) (t=3s)')    
plt.xlabel('Time [s]')
plt.ylabel('space [m]')
plt.legend()
plt.savefig('space_t3s.jpeg')
plt.show()

plt.plot(t,x[1],color='black',label='v(t)')
plt.grid()
plt.title('Velocity slope v(t) (t=3s)')    
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.savefig('velocity_t3s.jpeg')
plt.show()

plt.plot(t,a,color='black',label='a(t)')
plt.grid()
plt.title('Aceleration slope a(t) (t=3s)')    
plt.xlabel('Time [s]')
plt.ylabel('Aceleration [m/(s.s)]')
plt.legend()
plt.savefig('aceleration_t3s.jpeg')
plt.show()    