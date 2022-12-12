from scipy.integrate import solve_ivp
import numpy as np
# from model_parameters import ModelParameters


class Simulator():
  def __init__(self, dt):
    self.dt = dt

  def set_model_params(self, model_parameters):
    self.m1 = model_parameters.m1
    self.m2 = model_parameters.m2
    self.L1 = model_parameters.L1
    self.L2 = model_parameters.L2
    self.l1 = model_parameters.l1
    self.l2 = model_parameters.l2
    self.I1 = model_parameters.I1
    self.I2 = model_parameters.I2
    self.f1 = model_parameters.f1
    self.f2 = model_parameters.f2
		
    self.g = model_parameters.g

  def simulate_step(self, x, u=0):
    self.u = u

    t_eval = np.arange(0, self.dt, self.dt/10)
    
    solution = solve_ivp(self._func, [0, self.dt], x, method='RK45', t_eval=t_eval)

    return solution.y[:, -1]

  def _func(self, t, x):
    M_inv = 1/(- self.L1**2*self.l2**2*self.m2**2*np.cos(x[1])**2 + self.I2*self.L1**2*self.m2 + self.I1*self.I2)*np.array([[self.I2, -(self.I2 + self.L1*self.l2*self.m2*np.cos(x[1]))],
                [-(self.I2 + self.L1*self.l2*self.m2*np.cos(x[1])), (self.m2*self.L1**2 + 2*self.l2*self.m2*np.cos(x[1])*self.L1 + self.I1 + self.I2)]])

    C = np.array([[-2*self.m2*self.L1*self.l2*np.sin(x[1])*x[3], -self.m2*self.L1*self.l2*np.sin(x[1])*x[3]],
                  [self.m2*self.L1*self.l2*np.sin(x[1])*x[2], 0]])

    tau = np.array([[-self.m1*self.g*self.l1*np.sin(x[0]) - self.m2*self.g*(self.L1*np.sin(x[0])+self.l2*np.sin(x[0]+x[1]))],
                    [-self.m2*self.g*self.l2*np.sin(x[0]+x[1])]])

    F = np.array([[self.f1, 0],[0, self.f2]])

    B = np.array([[1], [0]])

    f = np.zeros_like(x)
    f[0] = x[2]
    f[1] = x[3]
    f[2:4] = (M_inv @ (-(C+F) @ x[2:4].reshape(2,)
              + tau.reshape(2,) + B.reshape(2,) * self.u)).T
        
    return f
