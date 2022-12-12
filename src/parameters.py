from dataclasses import dataclass

@dataclass
class ModelParameters:
  m1: float
  m2: float
  L1: float
  L2: float
  l1: float
  l2: float
  I1: float
  I2: float
  f1: float
  f2: float

  g: float = 9.81


@dataclass
class MPCParameters:
  N: int
  dt: float
  u_lim: float
  max_iter: int
