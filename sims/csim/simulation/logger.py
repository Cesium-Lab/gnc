# """Logs everything. Pass in a logged component to have it logged"""
# import numpy as np

# class BaseLog:
#     def __init__(self): # TODO: do I need this?
#         self.vars = []
#         self.logged_values = []

#     def log(self):
#         for var in self.vars:
#             self.logged_values.append(var)


# class SC(BaseLog):
#     def __init__(self):
#         super().__init__()
#         self.pos = np.array([2])

#         self.vars.append(self.pos)


# spacecraft = SC()

# spacecraft.log()

# spacecraft.pos[0] += 2
# spacecraft.log()

# print(spacecraft.logged_values)

