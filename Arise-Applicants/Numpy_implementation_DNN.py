# activation function

def sigmoid(Z):
    return 1/(1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0,Z)

def sigmoid_backward(DA, Z):
    sig = sigmoid(Z)
    return DA * sig * (1 - sig)

def relu_backward(DA, Z):
    dZ = np.array(DA, copy = True)
    DZ[Z <= 0] = 0
    return DZ



def single_layer_forward_propagation(A_prev, w_curr,b_curr, activation = 'relu'):
    print(w_curr.shape, A_prev.shape, b_curr.shape)
    z_curr = np.dot(w_curr, A_prev) + b_curr
    
    if activation is 'relu':
        activation_func = relu
    elif activation is 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
        
    return activation_func(z_curr), z_curr



def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        
        activ_function_curr = layer['activation']
        W_curr = params_values['W' + str(layer_idx)]
        b_curr = params_values['b' + str(layer_idx)]
        A_curr, z_curr = single_layer_forward_propagation(A_prev, W_curr, b_curr, activ_function_curr)
        
        memory['A' + str(idx)] = A_prev
        memory['Z' + str(layer_idx)] = z_curr
        
    return A_curr, memory


# loss function

def get_cost_value(y_hat, Y):
    m = y_hat.shape[1]
    cost = -1 /m * (np.dot(Y, np.log(y_hat).T) + np.dot(1- Y, np.log(1 - y_hat).T))
    return np.squeeze(cost)

def get_accuracy_value(y_hat, Y):
    y_hat_ = convert_prob_into_class(y_hat)
    return (y_hat_ == Y).all(axis = 0).mean()



def single_layer_backward_propagation(da_curr, w_curr, b_curr, z_curr, A_prev, activation = 'relu'):
    m = A_prev.shape[1]
    
    if activation is 'relu':
        backward_activation_func = relu_backward
    elif activation is 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')
        
    dz_curr = backward_activation_func(da_curr, z_curr)
    dw_curr = np.dot(dz_curr, A_prev.T)/m
    db_curr = np.sum(dz_curr, axis = 1, keepdims = True) /m
    da_curr = np.dot(w_curr.T, dz_curr)
    
    return da_prev, dw_curr, db_curr



def full_backward_propagation(y_hat, Y, memory, params_values, nn_architecture):
    grads_value = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)
    
    
    da_prev = -(np.divide(Y, Y_hat) - np.divide(1-Y, 1 - Y_hat))
    
    for layer_idx_prev, layer in reversed(list(enumeration(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer['activation']
        
        da_curr = da_prev
        a_prev = memory['A' + str(layer_idx_prev)]
        z_curr = memory['Z' + str(layer_idx_curr)]
        w_curr = params_values['W' + str(layer_idx_curr)]
        b_curr = params_values['b' + str(layer_idx_curr)]
        
        
        da_prev, dw_curr, db_curr = single_layer_backward_propagation(
        da_curr, w_curr, b_curr, z_curr, a_prev, activ_function_curr)
        
        grads_values['dw' + str(layer_idx_curr)] = dw_curr
        grads_values['db' + str(layer_idx_curr)] = db_curr
        
    return grads_values


def update(params_values, grad_values, nn_architecture, learning_rate):
    for layer_idx, layer in enumerate(nn_architecture):
        params_values['W' + str(layer_idx)] -= learning_rate * grads_values['dw' + str(layer_idx)]
        params_values['b' + str(layer_idx)] -= learning_rate * grads_values['db' + str(layer_idx)]
        
    return params_values


def train(X, y, nn_architecture, epochs, learning_rate):
    params_values = init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []
    
    for i in range(epochs):
        Y_hat, cashe = full_forward_propagation(X, params_values, nn_architecture)
        cost = get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(Y_hat, Y)
        accuracy_history.append(accuracy)
        
        grads_values = full_backward_propagation(Y_hat, Y, cashe, params_values, nn_archtecture)
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)
    return params_values, cost_history, accuracy_history


params_value, cost_history, acc = train(x_train, y_train, nn_architecture, epochs = 20, learning_rate = 0.001)
    
