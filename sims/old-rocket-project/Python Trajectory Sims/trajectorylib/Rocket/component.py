import numpy as np
import matplotlib.pyplot as plt
class Component:

    def __init__(self,
                 mass,
                 top_pos,
                 CG_loc,
                 I_matrix,
                 name = "Component",
                 type = "Generic Component",
                 ):

        self.name = name
        self.type = type
        self.mass = mass
        self.top_pos = top_pos
        self.CG_loc = CG_loc
        self.I_matrix = I_matrix

    def set_plot_elements(self,
                          fmt = None,
                          color = None,
                          name = None
                          ):
        if fmt:
            self.plot_fmt = fmt
        if color:
            self.plot_color = color
        if name:
            self.plot_name = name
        
    def plot_CG(self, color = 'k'):
        # print(self.CG_loc)
        plt.plot(self.CG_loc[2], self.CG_loc[0], 'x', color = color)

