import numpy as np
from image import generate_diagram
# call generate_diagram to get ship
grid = generate_diagram()

#flatten the 2d array to a 1d array
input = grid.flatten()
print(input)

