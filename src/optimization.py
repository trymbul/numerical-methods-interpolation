from autograd import grad

def gradient_descent(num_nodes, start, stop, N, f, f_approx, initial_L=1, tol=1e-6):
    #Initial guesses
    rho_up = 2.0
    rho_down = 0.5
    initial_nodes = np.linspace(start, stop, num_nodes)
    initial_eps = 1.5
    real_line = np.linspace(start, stop, N +1)

    L = initial_L
    params = np.append(initial_nodes, initial_eps)

    def C(params):
        nodes = params[:-1]
        #Making sure the nodes stay sorted and witithin the domain
        nodes = np.sort(np.clip(nodes, start, stop))
        eps = params[-1]
        coeff = (stop - start)/N
        summand = (f(real_line)- f_approx(nodes, f(nodes), real_line, eps))**2
        return  coeff * np.sum(summand)

    gradient_func = grad(C) #Telling autograd C is the function its finding the gradiant of
    history = [C(params)]   #Creating a list to save the costs for plotting later

    for k in range(10000):
        grad_vec = gradient_func(params)    #Computing the gradient
        grad_vec[-1] = 0.1*grad_vec[-1]    #Making sure the epsilon gradient doesnt dominate the nodes
        phi = C(params) #Finding the current cost

        #Checking if the gradient contains NaN
        if np.isnan(grad_vec).any():
            raise ValueError('Gradienten contains NaN!.')

        #Checking if the gradient is 0 i.e. the cost is at a minimum
        if np.linalg.norm(grad_vec) < tol:
            break
        while True:
            params_tilde = params - (1.0 / L) * grad_vec    #Computing new parameters
            phi_tilde = C(params_tilde) #Finding the cost of the new parameters

            #Checking if the new parameters should be accepted
            #The cost should be reduced sufficiently relative to the gradient, to make sure we havent overshot the minimum
            dot_product = np.dot(grad_vec, params_tilde - params)
            norm_squared = np.linalg.norm(params_tilde - params) ** 2
            if phi_tilde <= phi + dot_product + L * norm_squared / 2:
                #Accepting the new parameters and phi
                params = params_tilde
                phi = phi_tilde

                #Saving the new cost and increasing the length of the new step
                history.append(phi)
                L = rho_down * L
                break
            
            #If the step is not accepted reduce the step length
            else:
                L = rho_up * L
    #Signaling if the approximation did not converge
    if k == 10000-1:
        print('Function didnt converge')

    #Extracting the parameters
    optimized_nodes = params[:-1]
    optimized_eps = params[-1]
    return optimized_nodes, optimized_eps, history