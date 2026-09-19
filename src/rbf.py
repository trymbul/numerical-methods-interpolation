import autograd.numpy as np #Had to use autograd.numpy for np.linalg.solve to work when using autograd in task e)

def fi(r_sq, eps):
    return np.e**(-r_sq * eps**2)   #using r^2 as the variable instead of r to avoid getting NaN values

def M(nodes, eps):
    #Constructing a distance matrix where all distances are squared
    r_matrix = (nodes[:, None] - nodes[None, :]) ** 2
    return fi(r_matrix, eps)    #Calculating M

def RBF(nodes, vals, x, eps):
    nodes = np.array(nodes) #Making sure nodes is an array
    M_matrix = M(nodes, eps)

    w = np.linalg.solve(M_matrix, vals) #Constructing the vector w consisting of weights w_i
    summands = w * fi((x[:, None]-nodes)**2, eps)
    return np.sum(summands, axis=1)

def M_condition_number(nodes, eps):
    return np.linalg.cond(M(nodes, eps))