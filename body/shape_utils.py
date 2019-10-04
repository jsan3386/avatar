#
#
import numpy as np

def read_eigenbody (filename):
    eigenbody = []
    f_eigen = open(filename,'r')

    for line in f_eigen:
        eigenbody.append(float(line))

    return np.array(eigenbody)

def compose_vertices_eigenmat (eigenmat):
    eigenvertices = []
    for i in range(0,len(eigenmat),3):
        eigenvertices.append([eigenmat[i],-eigenmat[i+2],eigenmat[i+1]])

