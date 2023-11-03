from interface import *

# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            return  parameter - self.lr*parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter

            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam

                :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = updater.inertia*self.momentum + self.lr*parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        inputs = inputs.copy()
        inputs[inputs < 0] = 0
        return inputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        relu_deriv = self.forward_inputs.copy()
        relu_deriv[relu_deriv >= 0] = 1
        relu_deriv[relu_deriv < 0] = 0
        return grad_outputs*relu_deriv 
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, d)), output values

                n - batch size
                d - number of units
        """
        # your code here \/
        inputs = inputs.copy()
        inputs -= np.max(inputs, axis = 1)[:, None]
        exps = np.exp(inputs)
        norm_coefs = np.sum(exps, axis=1)[:,None]
        return np.divide(exps, norm_coefs)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of units
        """
        # your code here \/
        # dL/dz = SM * (dL/dOutputs - (dl/dOutputs).T * SM)
        exps = np.exp(self.forward_inputs - np.max(self.forward_inputs, axis = 1)[:,None])
        sm = exps / np.sum(exps, axis = 1, keepdims=True)
        dL_mult_sm = np.sum(grad_outputs * sm, axis = 1)[:,None]
        return sm * (grad_outputs - dL_mult_sm)


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d)), input values

            :return: np.array((n, c)), output values

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        return np.dot(inputs, self.weights) + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs

            :return: np.array((n, d)), dLoss/dInputs

                n - batch size
                d - number of input units
                c - number of output units
        """
        # your code here \/
        self.weights_grad = np.dot(self.forward_inputs.T, grad_outputs)
        self.biases_grad = np.sum(grad_outputs, axis=0)
        return np.dot(grad_outputs, self.weights.T)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((1,)), mean Loss scalar for batch

                n - batch size
                d - number of units
        """
        # your code here \/
        target_indices = np.argmax(y_gt, axis=1)
        n = target_indices.shape[0]
        probs = y_pred[(np.arange(n), target_indices)]
        # print(probs)
        loss = -np.mean((np.log(probs + eps)))[None]
        return loss
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values

            :return: np.array((n, d)), dLoss/dY_pred

                n - batch size
                d - number of units
        """
        # your code here \/
        target_indices = np.argmax(y_gt, axis=1)
        dprediction = y_gt.copy()
        n = y_pred.shape[0]
        dprediction /= -(y_pred + eps)
        dprediction = dprediction/n
        return dprediction
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    lr = 1e-3
    momentum = 0.1
    model = Model(CategoricalCrossentropy(), SGDMomentum(lr, momentum))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    
    model.add(Dense(512, (784,)))
    model.add(ReLU(512))
    model.add(Dense(512))
    model.add(ReLU(512))
    # ...
    model.add(Dense(1024))
    model.add(ReLU(1024))
    model.add(Dense(10))
    model.add(Softmax(10))

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, batch_size=16,epochs=5, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get('USE_FAST_CONVOLVE', False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
        :param inputs: np.array((n, d, ih, iw)), input values
        :param kernels: np.array((c, d, kh, kw)), convolution kernels
        :param padding: int >= 0, the size of padding, 0 means 'valid'

        :return: np.array((n, c, oh, ow)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
    """
    # your code here \/

    kernels = kernels[...,::-1,::-1]
    
    di = kernels.shape[-2]
    dj = kernels.shape[-1]

    if padding != 0:
        inputs = np.pad(inputs, [[0,0],[0,0],[padding, padding],[padding, padding]])
    # if padding == 1:
    #     inputs = np.pad(inputs, [[0,0],[0,0],[(di-1)//2,(di-1)//2],[(dj-1)//2,(dj-1)//2]])
    # elif padding == 2:
    #     inputs = np.pad(inputs, [[0,0],[0,0],[di-1,di-1],[dj-1,dj-1]])

    ext_img = inputs[:,None,...]
    ext_kernel = kernels[None,...]
    res = np.zeros((inputs.shape[0],kernels.shape[0], inputs.shape[-2]- kernels.shape[-2] + 1,inputs.shape[-1] - kernels.shape[-1] + 1))

    for i in range(res.shape[-2]):
        for j in range(res.shape[-1]):
            i_input = i# + di//2
            j_input = j# + dj//2
            res[..., i,j] = ext_img[...,i_input:i_input+di,j_input:j_input+dj].reshape(inputs.shape[0], -1) @ ext_kernel.reshape(kernels.shape[0],-1).T
    return res
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name='kernels',
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_channels,),
            initializer=np.zeros
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, c, h, w)), output values

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        p = (self.kernels.shape[-1] - 1)//2
        # print('--------', self.biases[...,None,None].shape, '-----------')
        return convolve(inputs, self.kernels, p) + self.biases[...,None,None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of input channels
                c - number of output channels
                (h, w) - image shape
        """
        # your code here \/
        X_T = np.swapaxes(self.forward_inputs[...,::-1,::-1], 0, 1)
        dl_dy_T = np.swapaxes(grad_outputs, 0, 1)
        p = (self.kernels.shape[-1] - 1)//2
        self.kernels_grad = np.swapaxes(convolve(X_T, dl_dy_T, p), 0, 1)

        self.biases_grad = np.sum(grad_outputs, axis=0) 
        self.biases_grad = np.sum(self.biases_grad.reshape(self.biases_grad.shape[0], -1), axis = 1)
        
        K_T = np.swapaxes(self.kernels[...,::-1,::-1], 0, 1)
        p = (K_T.shape[-1] - 1)//2

        inputs_grad = convolve(grad_outputs, K_T, p)


        return inputs_grad
        # your code here /\


# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode='max', *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {'avg', 'max'}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, ih, iw)), input values

            :return: np.array((n, d, oh, ow)), output values

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        n,d,ih,iw = inputs.shape
        p = self.pool_size
        blocks = inputs.reshape((n, d, -1, p, iw//p, p))
        blocks = blocks.transpose((0, 1, 2,4,3,5))
        
        if self.pool_mode == 'max':
            maxes = np.max(np.max(blocks, axis=-1, keepdims=True), axis = -2, keepdims=True)
            pooled = maxes.transpose((0, 1, 2,4,3,5)).reshape(n, d, ih//p, iw//p)
            self.max_indices = np.argmax(blocks.reshape(-1, p*p), axis=-1)
        
        else:
            block_means = np.sum(blocks, axis=(-1,-2)) / p**2
            pooled = block_means.reshape(n, d, ih//p, iw//p)

        return pooled
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

            :return: np.array((n, d, ih, iw)), dLoss/dInputs

                n - batch size
                d - number of channels
                (ih, iw) - input image shape
                (oh, ow) - output image shape
        """
        # your code here \/
        p = self.pool_size
        n,d,ih,iw = self.forward_inputs.shape
        grads_flatten = np.repeat(grad_outputs.reshape(-1, 1), p*p, axis=-1)
        

        if self.pool_mode == 'max':
            result_flatten = np.zeros_like(grads_flatten)
            result_flatten[(np.arange(grads_flatten.shape[0]), self.max_indices)] = grads_flatten[\
                (np.arange(grads_flatten.shape[0]), self.max_indices)]
        else:
            result_flatten = grads_flatten/p**2
        
        result = result_flatten.reshape(n,d, -1, iw//p, p, p).transpose(0, 1, 2, 4, 3, 5).\
            reshape(n,d, ih, iw)
        
        return result
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name='beta',
            shape=(input_channels,),
            initializer=np.zeros
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name='gamma',
            shape=(input_channels,),
            initializer=np.ones
        )

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, d, h, w)), output values

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/

        self.flatten = lambda x, d: x.transpose(1, 0, 2, 3).reshape(d, -1)
        self.unflatten = lambda flatten, n,d,h,w: flatten.reshape(d, n, h, w).transpose(1, 0, 2, 3)

        n,d,h,w = inputs.shape
        flatten = inputs.transpose(1, 0, 2, 3).reshape(d, -1)
        if self.is_training:
            
            if self.running_mean is None:
                self.running_mean = 0
            if self.running_var is None:
                self.running_var = 0

            
            mu = np.mean(flatten, axis = 1)
            var = np.var(flatten, axis = 1)

            flatten = (flatten - mu[...,None])/np.sqrt(var[...,None] + eps)
            
            self.x_norm = self.unflatten(flatten, n, d, h, w)
            
            self.var = var
            

            self.running_mean = self.running_mean*self.momentum + (1 - self.momentum)*mu #(self.running_mean*self.count + mu)/next_count
            self.running_var = self.running_var*self.momentum + (1 - self.momentum)*var # (self.running_var*self.count + var)/next_count

            
        else:
            
            flatten = (flatten - self.running_mean[...,None])/np.sqrt(self.running_var[...,None] + eps)
            
        result = flatten.reshape(d, n, h, w)
        result = self.gamma[...,None, None, None]*result + self.beta[...,None,None,None]
        


        return result.transpose(1, 0, 2, 3)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of channels
                (h, w) - image shape
        """
        # your code here \/

        # n,d,h,w = grad_outputs.shape
        
        # grad_outputs = self.flatten(grad_outputs, d)

        self.beta_grad = np.sum(grad_outputs, axis=(0, 2, 3))#.reshape(d) # мб другая ось
        self.gamma_grad = np.sum(grad_outputs*self.x_norm, axis=(0, 2, 3))

        # dL/dV---------------------------------------
        dx_norm = grad_outputs*self.gamma[None, ..., None, None]


        mult = 1/np.sqrt(self.var[None,:,None,None] + eps)
        dvar = (dx_norm*self.x_norm).mean(axis=(0, 2, 3))[None,:,None,None]
        dx1 = dx_norm*mult
        dx2 = dx_norm.mean(axis=(0, 2, 3))[None,:,None,None]*mult #dvar[None,:,None,None]*2*x_centerd 
        dx3 = self.x_norm*dvar*mult
        # print('dx1---------', dx1)
        # print('dx2----------', dx2,'-----------')
        # print('dx3----------', dx3,'-----------')
        dx =  dx1 - dx2 - dx3
        return dx
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (np.prod(self.input_shape),)

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, d, h, w)), input values

            :return: np.array((n, (d * h * w))), output values

                n - batch size
                d - number of input channels
                (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

            :return: np.array((n, d, h, w)), dLoss/dInputs

                n - batch size
                d - number of units
                (h, w) - input image shape
        """
        # your code here \/
        n,d,h,w = self.forward_inputs.shape
        return grad_outputs.reshape(n,d,h,w)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values

            :return: np.array((n, ...)), output values

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        inputs = inputs.copy()
        if self.is_training:
            rand = np.random.uniform(size=inputs.shape)
            self.forward_mask = rand < self.p
            inputs[self.forward_mask] = 0
        else:
            inputs[self.forward_mask] *= (1-self.p)
        return inputs
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

            :return: np.array((n, ...)), dLoss/dInputs

                n - batch size
                ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        grad_outputs = grad_outputs.copy()
        grad_outputs[self.forward_mask] = 0
        return grad_outputs
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    loss = CategoricalCrossentropy()
    optim = SGDMomentum(1e-3, 0.1)
    model = Model(loss, optim)

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Conv2D(kernel_size=3, input_shape=(3,32,32), output_channels=32))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Dropout(0.2))


    model.add(Conv2D(kernel_size=3, output_channels=64))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2))

    model.add(Conv2D(kernel_size=3, output_channels=128))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(2))

    model.add(Conv2D(kernel_size=3, output_channels=64))
    model.add(ReLU())
    model.add(BatchNorm())
    model.add(Pooling2D(4))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(10))
    model.add(Softmax(10))



    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, batch_size = 16, epochs = 10, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model

# ============================================================================
