import time
import math
import numpy as np
import pylab as py
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib import rc

class Plotter():
  def __init__(self):
    pass

  def animate_pendulum(phi1, phi2, L1, L2, dt):
  # animStep: delay between each frame of the animation in milliseconds
    animStep = dt*1000

    rc('animation', html='jshtml')

    def coordinateComp(phi1, phi2, L1, L2):
      x1 = L1 * np.sin(phi1)
      y1 = - L1 * np.cos(phi1)

      x2 = x1 + L2 * np.sin(np.add(phi1,phi2))
      y2 = y1 - L2 * np.cos(np.add(phi1,phi2))  

      x2P = x1 + L2/2 * np.sin(np.add(phi1,phi2)) # point mass rod 2
      y2P = y1 - L2/2 * np.cos(np.add(phi1,phi2)) # point mass rod 2

      limit = 1.1*(L1 + L2)

      return x1, y1, x2, y2, x2P, y2P, limit


    # compute x and y coordinates of the masses 
    # phi1, phi2: numpy arrays with the displacement of the rods at each time step
    # L1, L2: length of the rods
    x1, y1, x2, y2, x2P, y2P, limit = coordinateComp(phi1, phi2, L1, L2)
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.axes(xlim=(-limit, limit), ylim=(-limit, limit))
    line1, = ax.plot([], [], 'o-',color = '#d2eeff',markersize = 16, markerfacecolor = '#0077BE',lw=3, markevery=10000, markeredgecolor = 'k')
    line2, = ax.plot([], [], 'o-',color = '#ffebd8',markersize = 16, markerfacecolor = '#f66338',lw=3, markevery=10000, markeredgecolor = 'k')
    line3, = ax.plot([], [], color='k', linestyle='-', linewidth=3)
    line4, = ax.plot([], [], color='k', linestyle='-', linewidth=3)
    line5, = ax.plot([], [], 'o', color='k', markersize = 7)
    time_template = 'Time = %.1f s'
    time_string = ax.text(0.05, 0.9, '', transform=ax.transAxes)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


    # initialization function: plot the background of each frame
    def init():
      line1.set_data([], [])
      line2.set_data([], [])
      line3.set_data([], [])
      line4.set_data([], [])
      line5.set_data([], [])
      time_string.set_text('')
      
      return  line3, line4, line5, line1, line2, time_string
    
    # animation function.  This is called sequentially
    def animate(i):
      # Motion trail sizes. Defined in terms of indices. Length will vary with the time step, dt. E.g. 5 indices will span a lower distance if the time step is reduced.
      trail1 = 6              # length of motion trail of weight 1 
      trail2 = 8              # length of motion trail of weight 2
      
      if i == 0 or i == 1:
        line1.set_data(x1[i]/2, y1[i]/2)   # marker + line of first weight
        line2.set_data(x2P[i], y2P[i])   # marker + line of the second weight
      else:
        line1.set_data(x1[i:max(1,i-trail1):-1]/2, y1[i:max(1,i-trail1):-1]/2)   # marker + line of first weight
        line2.set_data(x2P[i:max(1,i-trail2):-1], y2P[i:max(1,i-trail2):-1])   # marker + line of the second weight
      
      line3.set_data([x1[i], x2[i]], [y1[i], y2[i]])       # line connecting weight 2 to weight 1
      line4.set_data([x1[i], 0], [y1[i],0])                # line connecting origin to weight 1
      
      line5.set_data([0, 0], [0, 0])
      time_string.set_text(time_template % (i*dt))
      
      return  line3, line4, line5, line1, line2, time_string


    anim = animation.FuncAnimation(fig, animate, np.arange(0, len(x1)), interval=animStep, blit=True, init_func=init)
    plt.close() 
    
    return anim