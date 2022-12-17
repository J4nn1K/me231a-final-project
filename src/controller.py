import pyomo.environ as pyo
import numpy as np
import logging

logging.getLogger('pyomo.core').setLevel(logging.ERROR)
import scipy

class MPC():
  def __init__(self):
    pass

  def set_model_params(self, model_parameters):
    self.m1 = model_parameters.m1
    self.m2 = model_parameters.m2
    self.L1 = model_parameters.L1
    self.L2 = model_parameters.L2
    self.l1 = model_parameters.l1
    self.l2 = model_parameters.l2
    self.I1 = model_parameters.I1
    self.I2 = model_parameters.I2
    self.g = model_parameters.g
    self.f1 = model_parameters.f1
    self.f2 = model_parameters.f2

  def set_controller_params(self, controller_parameters):
    self.N = controller_parameters.N
    self.dt = controller_parameters.dt
    self.u_lim = controller_parameters.u_lim
    self.max_iter = controller_parameters.max_iter
    self.x_fac = 1

  def solve_cftoc(self, x0):    
    model = pyo.ConcreteModel()
    model.N = self.N
    model.nx = 4
    model.nu = 1

    # length of finite optimization problem:
    model.tIDX_x = pyo.Set( initialize= range(self.x_fac*model.N+1), ordered=True )
    model.tIDX = pyo.Set( initialize= range(model.N+1), ordered=True )
    model.xIDX = pyo.Set( initialize= range(model.nx), ordered=True )
    model.uIDX = pyo.Set( initialize= range(model.nu), ordered=True ) 

    # Create state and input variables trajectory:
    model.x = pyo.Var(model.xIDX, model.tIDX_x)
    model.u = pyo.Var(model.uIDX, model.tIDX, bounds=(-self.u_lim, self.u_lim))

    # model params:
    model.m1 = self.m1
    model.m2 = self.m2
    model.L1 = self.L1
    model.L2 = self.L2
    model.l1 = self.l1
    model.l2 = self.l2
    model.I1 = self.I1
    model.I2 = self.I2
    model.f1 = self.f1
    model.f2 = self.f2

    model.g = self.g
    model.last_t = None
    
    #Objective:
    model.cost = pyo.Objective(rule = self.objective_rule, sense=pyo.minimize)
    
    # Constraints:
    model.equality_constraints = pyo.Constraint(model.tIDX_x, model.xIDX, rule=self.equality_const_rule)
    model.init_const1 = pyo.Constraint(expr = model.x[0, 0] == x0[0])
    model.init_const2 = pyo.Constraint(expr = model.x[1, 0] == x0[1])
    model.init_const3 = pyo.Constraint(expr = model.x[2, 0] == x0[2])
    model.init_const4 = pyo.Constraint(expr = model.x[3, 0] == x0[3])
           
    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = self.max_iter
    results = solver.solve(model)#,tee=True)
    
    if str(results.solver.termination_condition) == "optimal" or str(results.solver.termination_condition) == "maxIterations":
        feas = True
    else:
        feas = False
            
    xOpt = np.asarray([[model.x[i,t]() for i in model.xIDX] for t in model.tIDX_x if t%self.x_fac == 0]).T # if t%self.x_fac == 0
    uOpt = np.asarray([model.u[:,t]() for t in model.tIDX]).T
    
    JOpt = model.cost()

    return [model, feas, xOpt, uOpt, JOpt]

  def objective_rule(self,model):     
    E_kin, E_pot = 0, 0
    for t in model.tIDX:
      t *= self.x_fac
      E_kin_p1 = 1 / 2 * model.I1 * model.x[2, t]**2
      E_kin_p2 = 1 / 2 * (model.m2*model.L1**2 + model.I2 + 2*model.m2*model.L1*model.l2*pyo.cos(model.x[1, t]))*model.x[2, t]**2 + 1/2*model.I2*model.x[3, t]**2 + (model.I2 + model.m2*model.L1*model.l2*pyo.cos(model.x[1, t]))*model.x[2, t]*model.x[3, t]
      E_kin += E_kin_p1 + E_kin_p2
      E_pot += model.m1*model.l1*model.g + model.m2*(model.L1+model.l2)*model.g -(model.m1 * model.g * model.l1 * pyo.cos(model.x[0, t]) + model.m2 * model.g * (model.L1 * pyo.cos(model.x[0, t]) + model.l2 * pyo.cos(model.x[0, t]+model.x[1, t])))
    return E_kin - E_pot

  def equality_const_rule(self,model, t, i):
    if i < 2:
      return model.x[i, t+1] == model.x[i, t] + self.dt/self.x_fac*model.x[i+2, t] if t < model.N else pyo.Constraint.Skip
    else:
      if t != model.last_t:
        model.last_t = t
        model.M_inv = (np.array([[model.I2/(- model.L1**2*model.l2**2*model.m2**2*pyo.cos(model.x[1, t])**2 + model.I2*model.L1**2*model.m2 + model.I1*model.I2), -1*(model.I2 + model.L1*model.l2*model.m2*pyo.cos(model.x[1, t]))/(- model.L1**2*model.l2**2*model.m2**2*pyo.cos(model.x[1, t])**2 + model.I2*model.L1**2*model.m2 + model.I1*model.I2)],
                                 [-(model.I2 + model.L1*model.l2*model.m2*pyo.cos(model.x[1, t]))/(- model.L1**2*model.l2**2*model.m2**2*pyo.cos(model.x[1, t])**2 + model.I2*model.L1**2*model.m2 + model.I1*model.I2), (model.m2*model.L1**2 + 2*model.l2*model.m2*pyo.cos(model.x[1, t])*model.L1 + model.I1 + model.I2)/(- model.L1**2*model.l2**2*model.m2**2*pyo.cos(model.x[1, t])**2 + model.I2*model.L1**2*model.m2 + model.I1*model.I2)]]))
        model.C = np.array([[-2*model.m2*model.L1*model.l2*pyo.sin(model.x[1, t])*model.x[3, t], -model.m2*model.L1*model.l2*pyo.sin(model.x[1, t])*model.x[3, t]],
                            [model.m2*model.L1*model.l2*pyo.sin(model.x[1, t])*model.x[2, t], 0]])
        model.tau = np.array([[-model.m1*model.g*model.l1*pyo.sin(model.x[0, t]) - model.m2*model.g*(model.L1*pyo.sin(model.x[0, t])+model.l2*pyo.sin(model.x[0, t]+model.x[1, t]))],
                        [-model.m2*model.g*model.l2*pyo.sin(model.x[0, t]+model.x[1, t])]])
        model.B = np.array([[1.0],[0.0]])
        model.MC = np.zeros_like(model.M_inv)
        model.Mtau = np.zeros_like(model.tau)
        model.MB = np.zeros_like(model.tau)
        for i2 in range(2):
          for k in range(2):
            model.MC[i2,k] = sum(model.M_inv[i2,j] * model.C[j,k] for j in range(2))
            
          model.Mtau[i2,0] = sum(model.M_inv[i2,j] * model.tau[j,0] for j in range(2))
          model.MB[i2,0] = sum(model.M_inv[i2,j] * model.B[j,0] for j in range(2))
      return (model.x[i, t+1] == model.x[i, t] + self.dt*(-sum(-model.MC[i-2, j] * model.x[j+2, t] for j in range(2)) + model.Mtau[i-2, 0]
                             +  model.MB[i-2, 0] * model.u[0, t//self.x_fac] )) if t < model.N*self.x_fac else pyo.Constraint.Skip
   
  def update(self, x0):
    model, feas, xOpt, uOpt, JOpt = self.solve_cftoc(x0)
    u = uOpt[0]
    return u

class LinearReferenceMPC(MPC):
  def __init__(self):
        super().__init__()  #necessary??

  def objective_rule(self,model):
    return sum((model.x[0,t]-np.pi)**2 + (model.x[1,t]+model.x[0,t]-np.pi)**2 + 0.01*model.u[0,t]**2  for t in model.tIDX if t < model.N) 

  def equality_const_rule(self,model, t, i):
    if i < 2:
      return model.x[i, t+1] == model.x[i, t] + self.dt/self.x_fac*model.x[i+2, t] if t < model.N else pyo.Constraint.Skip
    else:
      if t != model.last_t:
        #model.M_inv = np.array([[-self.I2/(-self.I1*self.I2 - self.I2*self.L1**2*self.m2 + 2.0*self.L1**2*self.l2**2*self.m2**2), (2.0*self.I2 - 2.0*self.L1*self.l2*self.m2)/(-2.0*self.I1*self.I2 - 2.0*self.I2*self.L1**2*self.m2 + 2.0*self.L1**2*self.l2**2*self.m2**2)], 
        #                    [(1.0*self.I2 - 1.0*self.L1*self.l2*self.m2)/(-1.0*self.I1*self.I2 - 1.0*self.I2*self.L1**2*self.m2 + 1.0*self.L1**2*self.l2**2*self.m2**2), (-1.0*self.I1 - 1.0*self.I2 - 1.0*self.L1**2*self.m2 + 2.0*self.L1*self.l2*self.m2)/(-1.0*self.I1*self.I2 - 1.0*self.I2*self.L1**2*self.m2 + 1.0*self.L1**2*self.l2**2*self.m2**2)]])
        model.M_inv = np.array([[-self.I2/(-self.I1*self.I2 - self.I2*self.L1**2*self.m2 + self.L1**2*self.l2**2*self.m2**2), (self.I2 + self.L1*self.l2*self.m2)/(-self.I1*self.I2 - self.I2*self.L1**2*self.m2 + self.L1**2*self.l2**2*self.m2**2)], 
                                [(self.I2 + self.L1*self.l2*self.m2)/(-self.I1*self.I2 - self.I2*self.L1**2*self.m2 + self.L1**2*self.l2**2*self.m2**2), (-self.I1 - self.I2 - self.L1**2*self.m2 - 2*self.L1*self.l2*self.m2)/(-self.I1*self.I2 - self.I2*self.L1**2*self.m2 + self.L1**2*self.l2**2*self.m2**2)]])

        
        model.Q = -np.array([[1.0*self.g*self.l1*self.m1 - self.g*self.m2*(-1.0*self.L1 - 1.0*self.l2), 1.0*self.g*self.l2*self.m2],
                      [1.0*self.g*self.l2*self.m2, 1.0*self.g*self.l2*self.m2]])
        model.F = np.array([[self.f1,0],[0,self.f2]])
        model.B = np.array([[1],[0]])
        
        model.MQ = np.zeros_like(model.Q)
        model.MF = np.zeros_like(model.F)
        model.MB = np.zeros_like(model.B)
        for i2 in range(2):
          for k in range(2):
            model.MQ[i2,k] = sum(model.M_inv[i2,j] * model.Q[j,k] for j in range(2))
            model.MF[i2,k] = sum(model.M_inv[i2,j] * model.F[j,k] for j in range(2))
          model.MB[i2,0] = sum(model.M_inv[i2,j] * model.B[j,0] for j in range(2))
          coordinate_diff = np.array([np.pi,0])
      
      return (model.x[i, t+1] == model.x[i, t] + self.dt*(sum(-model.MQ[i-2, j] * (model.x[j, t]-coordinate_diff[j]) - model.MF[i-2, j] * model.x[j+2, t] for j in range(2))
                                 +  model.MB[i-2, 0] * model.u[0, t//self.x_fac] )) if t < model.N*self.x_fac else pyo.Constraint.Skip

class PID():
  def __init__(self, Kp, Ki, Kd):
    self.Kp = Kp
    self.Ki = Ki
    self.Kd = Kd

  def update(self, e, de):
    u = self.Kp * e + self.Kd * de
    
    return u

class LQR():
  def __init__(self):
    pass

  def set_model_params(self, model_parameters):
    self.m1 = model_parameters.m1
    self.m2 = model_parameters.m2
    self.L1 = model_parameters.L1
    self.L2 = model_parameters.L2
    self.l1 = model_parameters.l1
    self.l2 = model_parameters.l2
    self.I1 = model_parameters.I1
    self.I2 = model_parameters.I2
    self.g = model_parameters.g
    self.f1 = model_parameters.f1
    self.f2 = model_parameters.f2
    
  def set_controller_params(self, Q, R):
    self.Q = Q
    self.R = R

  def get_K(self):
    M_inv = np.linalg.inv(np.array([[self.I1 + self.I2 + self.L1**2*self.m2 + 2*self.L1*self.l2*self.m2, self.I2 + self.L1*self.l2*self.m2],
                                    [self.I2 + self.L1*self.l2*self.m2, self.I2]]))
    Q = -np.array([[1.0*self.g*self.l1*self.m1 - self.g*self.m2*(-1.0*self.L1 - 1.0*self.l2), 1.0*self.g*self.l2*self.m2],
                      [1.0*self.g*self.l2*self.m2, 1.0*self.g*self.l2*self.m2]])
    F = np.array([[self.f1,0],[0,self.f2]])
            
    A = np.block([[np.zeros((2,2)), np.eye(2)],
                  [-M_inv @ Q, -M_inv @ F]])#.reshape(4,4)
    #print(A)
    B = np.array([[1],[0]])

    B = np.block([[np.zeros((2,1))],
                  [-M_inv@B]])

    # solve Discrete Algebraic Riccatti equation  
    P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)

    # compute the LQR gain
    #K = scipy.linalg.inv(B.T @ P @ B + self.R) @ (B.T @ P @ A)
    K = (B.T @ P)
    return K
  