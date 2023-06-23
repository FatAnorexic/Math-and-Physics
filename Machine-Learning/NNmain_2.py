import numpy as np
import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import sine_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., weight_regularizer_l2=0.,
                 bias_regularizer_l1=0., bias_regularizer_l2=0.):

        #we put inputsXnuerons, so that for every forward pass, we do not need to transpose the matrix, it's already
        #been done for us in the initialization
        self.weights=0.1*np.random.randn(n_inputs, n_neurons)

        #The first pass of biases is the shape, so we will use tuples.
        self.biases=np.zeros((1, n_neurons))

        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2

    def forward(self, inputs):
        self.output=np.dot(inputs, self.weights)+self.biases
        # During our forward pass we'll want to inputs were
        self.inputs = inputs
    #next we add a backward pass for layer_dense
    def backward(self,dvalues):
        #gradients on param's
        self.dweights=np.dot(self.inputs.T,dvalues)
        self.dbiases=np.sum(dvalues, axis=0,keepdims=True)

        if self.weight_regularizer_l1>0:
            dL1=np.ones_like(self.weights)
            dL1[self.weights<0]=-1
            self.dweights+=self.weight_regularizer_l1*dL1

        if self.weight_regularizer_l2>0:
            self.dweights+=2*self.weight_regularizer_l2*self.weights

        if self.bias_regularizer_l1>0:
            dL1=np.ones_like(self.biases)
            dL1[self.biases<0]=-1
            self.dbiases+=self.bias_regularizer_l1*dL1

        if self.bias_regularizer_l2>0:
            self.dbiases+=2*self.bias_regularizer_l2*self.biases

        #gradient on values
        self.dinputs=np.dot(dvalues,self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        #Again, we need to make sure our inputs are stored in memory
        self.inputs=inputs
    def backward(self,dvalues):
        #Since we are modifying the original value, we'll make a copy first
        self.dinputs=dvalues.copy()
        #zero gradient where values were less than or equal to 0
        self.dinputs[self.inputs<=0]=0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values=np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output=probabilities

    def backward(self,dvalues):
        #Create an unitialized array
        self.dinputs=np.empty_like(dvalues)

        #now to enumerate outputs and gradients into a 2D array of Jacobians and dinputs
        for i, (single_output,single_dvalues) in enumerate(zip(self.output, dvalues)):
            #flatten output array
            single_output=single_output.reshape(-1,1)
            #calculate Jacobian matrix of output
            jacobian_matrix=np.diagflat(single_output)-np.dot(single_output, single_output.T)

            #Calculate the sample-wise gradient and add it to the array of sample gradients
            self.dinputs[i]=np.dot(jacobian_matrix,single_dvalues)

class Loss:
    def calculate(self, output, y):
        sample_losses=self.forward(output,y)
        data_loss=np.mean(sample_losses)
        return data_loss

    def regulariztion_loss(self,layer):
        #it is 0 by default
        regularization_loss=0

        if layer.weight_regularizer_l1>0:
            regularization_loss+=layer.weight_regularizer_l1*np.sum(np.abs(layer.weights))

        if layer.weight_regularizer_l2>0:
            regularization_loss+=layer.weight_regularizer_l2*np.sum(layer.weights*layer.weights)

        if layer.bias_regularizer_l1>0:
            regularization_loss+= layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))

        if layer.bias_regularizer_l2>0:
            regularization_loss+=layer.bias_regularizer_l2*np.sum(layer.biases*layer.biases)

        return regularization_loss
class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples=len(y_pred)
        y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape)==1:
            correct_confidences=y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) ==2:
            correct_confidences=np.sum(y_pred_clipped*y_true, axis=1)
        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self,dvalues,y_true):
        #number of samples
        samples=len(dvalues)
        #number of labels in each sample-we'll use the first sample to count them
        labels=len(dvalues[0])
        #if labels are sparse, turn them into a one-hot vector
        if len(y_true.shape)==1:
            y_true=np.eye(labels)[y_true]

        #calculate the gradient
        self.dinputs=-y_true/dvalues
        #normalize the gradient
        self.dinputs=self.dinputs/samples

#Softmax classifier-combined softmax activation and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():
    #create activation and loss function
    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossEntropy()

    #forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output=self.activation.output
        return self.loss.calculate(self.output,y_true)

    def backward(self,dvalues,y_true):
        #number of samples
        samples=len(dvalues)

        #if samples are one-hot encoded,
        #turn them into discrete values
        if len(y_true.shape)==2:
            y_true=np.argmax(y_true, axis=1)
        #We still need those dvalues, so copy
        self.dinputs=dvalues.copy()

        #Calculate the gradient
        self.dinputs[range(samples), y_true]-=1
        #normalize the gradient
        self.dinputs=self.dinputs/samples

class Optimizer_SGD:
    #initialize optimizer-set settings,
    #learning rate is 1 for this optimizer
    def __init__(self,learning_rate=1., decay=0.,momentum=0.):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum

    #call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1./(1+self.decay*self.iterations))

    #update param's
    def update_params(self,layer):
        #if we use momentum
        if self.momentum:
            #if the layer does not contain momentum arrays, create them and populate with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights)
                layer.bias_momentums=np.zeros_like(layer.biases)
            #now build weight updates with momentum-take previous updates * retain factor and update with current
            #gradient
            weight_updates=self.momentum*layer.weight_momentums-self.current_learning_rate*layer.dweights
            layer.weight_momentums=weight_updates

            #do the same for biases
            bias_updates=self.momentum*layer.bias_momentums-self.current_learning_rate*layer.dbiases
            layer.bias_momentums=bias_updates
        #else we use vanilla SGD
        else:
            weight_updates=-self.current_learning_rate*layer.dweights
            bias_updates=-self.current_learning_rate*layer.dbiases

        #updates using either vanilla or momentums based changes
        layer.weights+=weight_updates
        layer.biases+=bias_updates

    #call once after any parameter updates
    def post_update_params(self):
        self.iterations+=1

class Optimizer_AdaGrad:
    #initialize optimizer-set settings,
    #learning rate is 1 for this optimizer
    def __init__(self,learning_rate=1., decay=0.,epsilon=1e-7):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon

    #call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1./(1+self.decay*self.iterations))

    #update param's
    def update_params(self,layer):
        #if there are no cached arrays create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.biases)

        #update the cache with square of the current grad
        layer.weight_cache+=layer.dweights*layer.dweights
        layer.bias_cache+=layer.dbiases*layer.dbiases

        #vanilla SGD with normalization
        layer.weights+=-self.current_learning_rate*layer.dweights/ \
                       (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases+=-self.current_learning_rate*layer.dbiases/ \
                      (np.sqrt(layer.bias_cache)+self.epsilon)

    #call once after any parameter updates
    def post_update_params(self):
        self.iterations+=1

class Optimizer_RMSProp:
    #a lr of 1 is far too large, most NN use a default of 0.001
    def __init__(self, learning_rate=0.001,decay=0., epsilon=1e-7,rho=0.9):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.rho=rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1./(1+self.decay*self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.biases)

        #update cache with squared current grad
        layer.weight_cache=self.rho*layer.weight_cache+(1-self.rho)*layer.dweights*layer.dweights
        layer.bias_cache=self.rho*layer.bias_cache+(1-self.rho)*layer.dbiases*layer.dbiases

        layer.weights+=-self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases+=-self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations+=1

class Optimizer_Adam:
    def __init__(self, learning_rate=0.001,decay=0.,epsilon=1e-7,
                 beta_1=0.9, beta_2=0.999):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.epsilon=epsilon
        self.beta_1=beta_1
        self.beta_2=beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1./(1.+self.decay*self.iterations))

    def update_params(self, layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums=np.zeros_like(layer.weights)
            layer.weight_cache=np.zeros_like(layer.weights)
            layer.bias_momentums=np.zeros_like(layer.biases)
            layer.bias_cache=np.zeros_like(layer.biases)

        layer.weight_momentums=self.beta_1*layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums=self.beta_1*layer.bias_momentums+(1-self.beta_1)*layer.dbiases

        weight_momentums_corrected=layer.weight_momentums/(1-self.beta_1**(self.iterations+1))
        bias_momentums_corrected=layer.bias_momentums/(1-self.beta_1**(self.iterations+1))

        layer.weight_cache=self.beta_2*layer.weight_cache+(1-self.beta_2)*layer.dweights**2
        layer.bias_cache= self.beta_2*layer.bias_cache+(1-self.beta_2)*layer.dbiases**2

        weight_cache_corrected=layer.weight_cache/(1-self.beta_2**(self.iterations+1))
        bias_cache_corrected=layer.bias_cache/(1-self.beta_2**(self.iterations+1))

        layer.weights+=-self.current_learning_rate*weight_momentums_corrected/\
                       (np.sqrt(weight_cache_corrected)+self.epsilon)

        layer.biases+=-self.current_learning_rate*bias_momentums_corrected/ \
                      (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params(self):
        self.iterations+=1

class Layer_Dropout:
    #init
    def __init__(self,rate):
        #Store rate, we invert it|as an example for dropout of 0.1 we need success rate of 0.9
        self.rate=1-rate

    def forward(self, inputs):
        self.inputs=inputs  #save input values
        #genarate and save a scaled mask
        self.binary_mask=np.random.binomial(1,self.rate, size=inputs.shape)/self.rate
        #Apply mask to output values
        self.output=inputs*self.binary_mask

    def backward(self,dvalues):
        #gradient on values
        self.dinputs=dvalues*self.binary_mask


class Activation_Sigmoid:
    def forward(self,inputs):
        #save inputs and calc/save output of sigmoid
        self.inputs=inputs
        self.output=1/(1+np.exp(-inputs))

    def backward(self,dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output
class Loss_BinaryCrossentropy(Loss):
    #forward pass
    def forward(self,y_pred,y_true):
        #add clipping to prevent log(0)
        y_pred_clipped=np.clip(y_pred,1e-7, 1-1e-7)
        #calculate sample-wise loss
        sample_losses=-(y_true*np.log(y_pred_clipped)+(1-y_true)*np.log(1-y_pred_clipped))
        sample_losses=np.mean(sample_losses, axis=-1)
        return sample_losses

    def backward(self,dvalues,y_true):
        #number of samples
        samples=len(dvalues)
        #number of outputs in every sample,using the first sample to count them
        outputs=len(dvalues[0])
        #clip dvalues to prevent 1/0
        clipped_dvalues=np.clip(dvalues,1e-7,1-1e-7)
        #calculate the gradient for dinputs
        self.dinputs=-(y_true/clipped_dvalues-(1-y_true)/(1-clipped_dvalues))/outputs
        #normalize the gradient
        self.dinputs=self.dinputs/samples
#Since we're no longer using classifacation labels and wish to predict a scalar value, we're going to use a linear
#function as the output layer. This func does not modify its input and simply passes it to the output|y=x|
class Activation_Linear:
    def forward(self,inputs):
        self.inputs=inputs
        self.output=inputs

    #The derivative of x is 1, thus our backward pass is 1*dvalues=dvalues-chain rule
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()

class Loss_MeanSquaredError(Loss):  #as usualy this class inherits from our generic loss function
    def forward(self,y_pred, y_true):
        sample_losses=np.mean((y_true-y_pred)**2, axis=-1)
        return sample_losses
    def backward(self,dvalues, y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=(-2*(y_true-dvalues))/outputs
        self.dinputs=self.dinputs/samples
class Loss_MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses=np.mean(np.abs(y_true-y_pred), axis=-1)
        return sample_losses
    def backward(self,dvalues, y_true):
        samples=len(dvalues)
        outputs=len(dvalues[0])
        self.dinputs=np.sign(y_true-dvalues)/outputs
        self.dinputs=self.dinputs/samples

#Regression model training
X,y=sine_data()

#Since we are going to be using this later, it makes sense to make this a class model


model=Model()
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64,1))
model.add(Activation_Linear())

model.set(loss=Loss_MeanSquaredError, optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3))
model.train(X,y, epochs=10000, print_every=100)


