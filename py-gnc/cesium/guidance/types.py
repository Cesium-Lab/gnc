from dataclasses import dataclass

import numpy as np


@dataclass
class FlybyStichScalarInput:
    v_arrival_PCI: float
    v_departure_PCI: float
    delta_des: float
    mu: float
    
@dataclass
class FlybyStichScalarOutput:
    dv : float
    v_p1 : float
    v_p2 : float
    r_p : float
    e1 : float
    e2 : float
    delta1 : float
    delta2 : float
    b1 : float
    b2 : float
    
@dataclass
class FlybyStichVectorInput:
    v_arrival_PCI: np.ndarray
    v_departure_PCI: np.ndarray
    mu: float
    
@dataclass
class FlybyStichVectorOutput:
    dv : np.ndarray
    v_p1 : np.ndarray
    v_p2 : np.ndarray
    r_p : np.ndarray
    e1 : float
    e2 : float
    delta1 : float
    delta2 : float
    b1 : np.ndarray
    b2 : np.ndarray