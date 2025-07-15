import numpy as np
import matplotlib.pyplot as plt
from odeSolvers import df_RK4
from mpl_toolkits import mplot3d
from numpy import linspace
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


G = 1 # Newton's G in AU**3 Msun**(-1) yr**(-2)

# 0-Mercury, 1-Venus, 2-Earth, 3-Mars, 4-Jupiter, 5-Saturn, 6-Uranus, 7-Neptune
Planets = np.array(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Comet'])
Msun = 1
#astropy table??
M = np.array([1.65*10**(-7), 2.45*10**(-6), 3*10**(-6), 3.21*10**(-7), 9.54*10**(-4), 2.86*10**(-4), 4.36*10**(-5), 5.15*10**(-5), 1.1*10**(-16)]) #Msun
e = np.array([0.206, 0.007, 0.017, 0.093, 0.048, 0.056, 0.0469, 0.010, 0.96658])
a = np.array([0.3871, 0.7233, 1, 1.5273, 5.2028, 9.5388, 19.1914, 30.0611, 17.737]) #AU
T = np.array([0.2408, 0.6152, 1, 1.8809, 11.862, 29.458, 84.01, 164.79, 74.7]) #yr
x0 = np.array([0.31, 0.72, 0.98, 1.38, 4.95, 9.01, 18.28, 29.80, 0.59278]) #perihelion in AU
y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
z0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
vx0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
vy0 = (a/T)*(((1+e)/(1-e))**0.5) #AU/yr 
vz0 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])


def derivs_gravity(state, t):
    '''Given an array of the instantaneous positions and velocities for
    two objects as input, returns the time derivatives of each input 
    quantity for objects orbiting each other.'''

    # create our x and v arrays
    nvar,nobj = state.shape # determine how many vars and objects we have
    ndim = int(nvar/2)  # count how many dimensions we have
    x = state[:ndim]    # first half of the variables are positions
    v = state[ndim:]    # second half of the variables are velocities

    # x (position) derivatives are equal to velocities
    dxdt = v

    # calculate forces --> find the distances between the planet and the sun, then calculate the forces

    rnorm = np.sqrt(x[0,:]**2 + x[1,:]**2 + x[2,:]**2)
    # Fg = -G*Msun*M/rnorm**3 * x
    ag = -G*Msun/rnorm**3 * x

    # v (velocity) derivatives are equal to accelerations
    # dvdt = np.c_[ Fg/M ]
    dvdt = np.c_[ ag ]

    return np.append(dxdt,dvdt,axis=0)

# Create the time axis
dt=0.01      # length of time step in years
tmax = 1500  # total simulation time in years
ts=np.arange(0, tmax+dt, dt)
Npts=len(ts)

# Create state array with initial conditions
# 1st dim is variable: x,y,vx,vy,vz
# 2nd dim is object id: A, B
# 3rd dim is time step: 0, 1, 2, ... Npts-1
state = np.zeros((6,9,Npts))

#properties of Mercury
state[0,0,0]=x0[0]
state[1,0,0]=y0[0]
state[2,0,0]=z0[0]
state[3,0,0]=vx0[0]
state[4,0,0]=vy0[0]
state[5,0,0]=vz0[0]

#properties of Venus
state[0,1,0]=x0[1]
state[1,1,0]=y0[1]
state[2,1,0]=z0[1]
state[3,1,0]=vx0[1]
state[4,1,0]=vy0[1]
state[5,1,0]=vz0[1]

#properties of Earth
state[0,2,0]=x0[2]
state[1,2,0]=y0[2]
state[2,2,0]=z0[2]
state[3,2,0]=vx0[2]
state[4,2,0]=vy0[2]
state[5,2,0]=vz0[2]

#properties of Mars
state[0,3,0]=x0[3]
state[1,3,0]=y0[3]
state[2,3,0]=z0[3]
state[3,3,0]=vx0[3]
state[4,3,0]=vy0[3]
state[5,3,0]=vz0[3]

#properties of Jupiter
state[0,4,0]=x0[4]
state[1,4,0]=y0[4]
state[2,4,0]=z0[4]
state[3,4,0]=vx0[4]
state[4,4,0]=vy0[4]
state[5,4,0]=vz0[4]

#properties of Saturn
state[0,5,0]=x0[5]
state[1,5,0]=y0[5]
state[2,5,0]=z0[5]
state[3,5,0]=vx0[5]
state[4,5,0]=vy0[5]
state[5,5,0]=vz0[5]

#properties of Uranus
state[0,6,0]=x0[6]
state[1,6,0]=y0[6]
state[2,6,0]=z0[6]
state[3,6,0]=vx0[6]
state[4,6,0]=vy0[6]
state[5,6,0]=vz0[6]

#properties of Neptune
state[0,7,0]=x0[7]
state[1,7,0]=y0[7]
state[2,7,0]=z0[7]
state[3,7,0]=vx0[7]
state[4,7,0]=vy0[7]
state[5,7,0]=vz0[7]

#properties of Comet
state[0,8,0]=x0[8]
state[1,8,0]=y0[8]
state[2,8,0]=z0[8]
state[3,8,0]=vx0[8]
state[4,8,0]=vy0[8]
state[5,8,0]=vz0[8]

for i in range(1, Npts):
    delta_i = df_RK4(state[:,:,i-1],ts[i-1],dt,derivs_gravity)
    state[:,:,i] = state[:,:,i-1] + delta_i

fontSize = 12
lineSize = 1
lineColor = 'k'
tickSize = 10

# 3D plot of position

fig = plt.figure(figsize = (8,8))
ax = plt.axes(projection = '3d')
for i in range(0, 9):
    ax.plot3D(state[0,i], state[1,i], state[2,i], linewidth=lineSize, ls='-', label=Planets[i])
ax.set_title('Solar System, tmax={:5.3f}'.format(tmax))
ax.set_xlabel('X Position (AU)', fontsize=fontSize)
ax.set_ylabel('Y Position (AU)', fontsize=fontSize)
ax.set_zlabel('Z Position (AU)', fontsize=fontSize)
ax.tick_params(axis="x", labelsize=tickSize)
ax.tick_params(axis="y", labelsize=tickSize)
ax.legend(loc='upper right', fontsize=fontSize)
fig.tight_layout()
plt.show()

#2D plot of position

fig,ax = plt.subplots()
ax.plot([0], [0], color='k', marker='+')
for i in range(0, 9):
    ax.plot(state[0,i], state[1,i], linewidth=lineSize, ls='-', label=Planets[i])
ax.set_title('Solar System, tmax={:5.3f}'.format(tmax))
ax.set_xlabel('X position (AU)', fontsize=fontSize)
ax.set_ylabel('Y position (AU)', fontsize=fontSize)
ax.tick_params(axis="x", labelsize=tickSize)
ax.tick_params(axis="y", labelsize=tickSize)
ax.axis('equal')
ax.legend(loc='upper right',fontsize=fontSize)
fig.tight_layout()
plt.show()

#Velocity
# 0-Mercury, 1-Venus, 2-Earth, 3-Mars, 4-Jupiter, 5-Saturn, 6-Uranus, 7-Neptune, 8-Comet

fig,axs = plt.subplots(2,1)
axs[0].set_title('Velocity, tmax={:5.3f}'.format(tmax))
axs[0].plot(ts, state[3,6], color='r', linewidth=lineSize, ls='-', label="Uranus vx")
axs[0].plot(ts, state[4,6], color='m', linewidth=lineSize, ls='-', label="Uranus vy")
axs[1].plot(ts, state[3,7], color='b', linewidth=lineSize, ls='-', label="Neptune vx")
axs[1].plot(ts, state[4,7], color='c', linewidth=lineSize, ls='-', label="Neptune vy")
for ax in axs:
    ax.axhline(0,color='k',ls='--',lw=0.5,zorder=-1)
    ax.set_xlabel('time (yr)', fontsize=fontSize)
    ax.set_ylabel('v (AU/yr)', fontsize=fontSize)
    ax.tick_params(axis="x", labelsize=tickSize)
    ax.tick_params(axis="y", labelsize=tickSize)
    ax.legend(loc='lower left',fontsize=fontSize)
fig.tight_layout()
plt.show()


# 2D animation of motion

fig, ax = plt.subplots()

xdata0, ydata0 = [], []
xdata1, ydata1 = [], []
xdata2, ydata2 = [], []
xdata3, ydata3 = [], []
xdata4, ydata4 = [], []
xdata5, ydata5 = [], []
xdata6, ydata6 = [], []
xdata7, ydata7 = [], []
xdata8, ydata8 = [], []
ln0 = plt.plot(state[0,0,0], state[1,0,0], linewidth=0.5)[0]
ln1 = plt.plot(state[0,1,0], state[1,1,0], linewidth=0.5)[0]
ln2 = plt.plot(state[0,2,0], state[1,2,0], linewidth=0.5)[0]
ln3 = plt.plot(state[0,3,0], state[1,3,0], linewidth=0.5)[0]
ln4 = plt.plot(state[0,4,0], state[1,4,0], linewidth=1)[0]
ln5 = plt.plot(state[0,5,0], state[1,5,0], linewidth=2)[0]
ln6 = plt.plot(state[0,6,0], state[1,6,0], linewidth=3)[0]
ln7 = plt.plot(state[0,7,0], state[1,7,0], linewidth=4)[0]
ln8 = plt.plot(state[0,8,0], state[1,8,0], linewidth=0.5)[0]

def update(frame):
    xdata0.append(state[0,0,frame])
    ydata0.append(state[1,0,frame])
    ln0.set_data(xdata0, ydata0)

    xdata1.append(state[0,1,frame])
    ydata1.append(state[1,1,frame])
    ln1.set_data(xdata1, ydata1)

    xdata2.append(state[0,2,frame])
    ydata2.append(state[1,2,frame])
    ln2.set_data(xdata2, ydata2)

    xdata3.append(state[0,3,frame])
    ydata3.append(state[1,3,frame])
    ln3.set_data(xdata3, ydata3)

    xdata4.append(state[0,4,frame])
    ydata4.append(state[1,4,frame])
    ln4.set_data(xdata4, ydata4)

    xdata5.append(state[0,5,frame])
    ydata5.append(state[1,5,frame])
    ln5.set_data(xdata5, ydata5)

    xdata6.append(state[0,6,frame])
    ydata6.append(state[1,6,frame])
    ln6.set_data(xdata6, ydata6)

    xdata7.append(state[0,7,frame])
    ydata7.append(state[1,7,frame])
    ln7.set_data(xdata7, ydata7)

    xdata8.append(state[0,8,frame])
    ydata8.append(state[1,8,frame])
    ln8.set_data(xdata8, ydata8)

    plt.axis('equal')
    plt.xlim([-30,30])

    return ln0, ln1, ln2, ln3, ln4, ln5, ln6, ln7
#for i in range(0,8):
 #   plt.plot(state[0,i], state[1,i], ls='--', label=Planets[i])
ax.plot([0], [0], color='k', marker='+', alpha=0.3)
plt.title("Solar System")
plt.xlabel("X Position")
plt.ylabel("Y Position")
ani = animation.FuncAnimation(fig, update, frames=5000, interval=1, cache_frame_data=False)
f = f"c://Users/sophi/Downloads/animation.gif"
writergif = animation.PillowWriter(fps=10000000)
ani.save(f, writer=writergif)
plt.close()