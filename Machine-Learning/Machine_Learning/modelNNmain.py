import numpy as np
import pickle
import copy

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1=0., weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):

        #we put inputsXnuerons, so that for every forward pass, we do not need to transpose the matrix, it's already
        #been done for us in the initialization
        self.weights=0.1*np.random.randn(n_inputs, n_neurons)

        #The first pass of biases is the shape, so we will use tuples.
        self.biases=np.zeros((1, n_neurons))

        self.weight_regularizer_l1=weight_regularizer_l1
        self.weight_regularizer_l2=weight_regularizer_l2
        self.bias_regularizer_l1=bias_regularizer_l1
        self.bias_regularizer_l2=bias_regularizer_l2

    def forward(self, inputs, training):
        # During our forward pass we'll want to inputs were
        self.inputs = inputs
        self.output=np.dot(inputs, self.weights)+self.biases

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

    def get_parameters(self):
        return self.weights, self.biases

    def set_parameters(self, weights, biases):
        self.weights=weights
        self.biases=biases

class Layer_Dropout:
    #init
    def __init__(self,rate):
        #Store rate, we invert it|as an example for dropout of 0.1 we need success rate of 0.9
        self.rate=1-rate

    def forward(self, inputs, training):
        self.inputs=inputs  #save input values
        #genarate and save a scaled mask

        if not training:
            self.output=inputs.copy()
            return
        self.binary_mask=np.random.binomial(1,self.rate, size=inputs.shape)/self.rate
        #Apply mask to output values
        self.output=inputs*self.binary_mask

    def backward(self,dvalues):
        #gradient on values
        self.dinputs=dvalues*self.binary_mask

class Layer_Input:
    def forward(self,inputs, training):
        self.output=inputs

class Activation_ReLU:
    def forward(self, inputs, training):
        # Again, we need to make sure our inputs are stored in memory
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self,dvalues):
        #Since we are modifying the original value, we'll make a copy first
        self.dinputs=dvalues.copy()
        #zero gradient where values were less than or equal to 0
        self.dinputs[self.inputs<=0]=0

    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    def forward(self, inputs, training):
        self.inputs=inputs
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

    def predictions(self,output):
        return np.argmax(output, axis=1)

class Activation_Sigmoid:
    def forward(self,inputs,training):
        #save inputs and calc/save output of sigmoid
        self.inputs=inputs
        self.output=1/(1+np.exp(-inputs))

    def backward(self,dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output

    def predictions(self, outputs):
        return (outputs>0.5)*1

class Activation_Linear:
    def forward(self,inputs, training):
        self.inputs=inputs
        self.output=inputs

    #The derivative of x is 1, thus our backward pass is 1*dvalues=dvalues-chain rule
    def backward(self,dvalues):
        self.dinputs=dvalues.copy()

    def predictions(self,outputs):
        return outputs

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

class Loss:
    #set trainable layers

    def regulariztion_loss(self):
        #it is 0 by default
        regularization_loss=0

        for layer in self.trainable_layers:

            if layer.weight_regularizer_l1>0:
                regularization_loss+=layer.weight_regularizer_l1*np.sum(np.abs(layer.weights))

            if layer.weight_regularizer_l2>0:
                regularization_loss+=layer.weight_regularizer_l2*np.sum(layer.weights*layer.weights)

            if layer.bias_regularizer_l1>0:
                regularization_loss+= layer.bias_regularizer_l1*np.sum(np.abs(layer.biases))

            if layer.bias_regularizer_l2>0:
                regularization_loss+=layer.bias_regularizer_l2*np.sum(layer.biases*layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regulariztion=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        self.accumulated_sum+=np.sum(sample_losses)
        self.accumulated_count+=len(sample_losses)
        if not include_regulariztion:
            return data_loss
        return data_loss, self.regulariztion_loss()
    def calculate_accumulated(self, *,included_regularization=False):
        data_loss=self.accumulated_sum/self.accumulated_count
        if not included_regularization:
            return data_loss
        return data_loss, self.regulariztion_loss()
    def new_pass(self):
        self.accumulated_sum=0
        self.accumulated_count=0

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
    def backward(self,dvalues,y_true):
        #number of samples
        samples=len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        #We still need those dvalues, so copy
        self.dinputs=dvalues.copy()

        #Calculate the gradient
        self.dinputs[range(samples), y_true]-=1
        #normalize the gradient
        self.dinputs=self.dinputs/samples

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

class Accuracy:
    def calculate(self,predictions,y):
        comparisons=self.compare(predictions, y)
        accuracy=np.mean(comparisons)
        self.accumulated_sum+=np.sum(comparisons)
        self.accumulated_count+=len(comparisons)
        return accuracy
    def calculate_accumulated(self):
        accuracy=self.accumulated_sum/self.accumulated_count
        return accuracy
    def new_pass(self):
        self.accumulated_sum=0
        self.accumulated_count=0

class Accuracy_Categorical(Accuracy):
    def init(self,y):
        pass
    def compare(self,predictions, y):
        if len(y.shape)==2:
            y=np.argmax(y,axis=1)
        return predictions==y

class Accuracy_Regression(Accuracy):
    def __init__(self):
        #calculate the precision property
        self.precision=None
    #calculates precision value based on passed in ground truth
    def init(self, y,reinit=False):
        if self.precision is None or reinit:
            self.precision=np.std(y)/250
    def compare(self, predictions, y):
        return np.absolute(predictions-y)<self.precision

#Since we are going to be using this later, it makes sense to make this a class model
class Model:
    def __init__(self):
        self.layers=[]
        self.softmax_classifier_output=None

    def add(self,layer):
        self.layers.append(layer)

    def set(self,*, loss=None,optimizer=None, accuracy=None):    #The * tells the compiler that these are keyword args, and need to be set
        if loss is not None:
            self.loss=loss

        if optimizer is not None:
            self.optimizer=optimizer

        if accuracy is not None:
            self.accuracy=accuracy

    def finalize(self):
        self.input_layer=Layer_Input()
        layer_count=len(self.layers)
        self.trainable_layers=[]
        for i in range(layer_count):
            #If it's the first layer, the previous object is the input layer
            if i==0:
                self.layers[i].prev=self.input_layer
                self.layers[i].next=self.layers[i+1]
            #all layers except first and last
            elif i<layer_count-1:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.layers[i+1]
            #Last layer, the next is loss
            else:
                self.layers[i].prev=self.layers[i-1]
                self.layers[i].next=self.loss
                self.output_layer_activation=self.layers[i]

            #If layer contains the 'weights' attribute, it's trainable-add it to the list of trainable layers
            #we do not need to check for biases-checking weights is enough
            if hasattr(self.layers[i],'weights'):
                self.trainable_layers.append(self.layers[i])
            if self.loss is not None:
                self.loss.remember_trainable_layers(self.trainable_layers)

        #if output activation is softmax and loss function is CCE, create an object of combined activation
        #and loss function containing faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output=Activation_Softmax_Loss_CategoricalCrossentropy()

    def train(self, X, y,*,epochs=1,batch_size=None, print_every=1, validation_data=None):
        self.accuracy.init(y)
        train_steps=1

        if validation_data is not None:
            validation_steps=1
            X_val,y_val=validation_data

        if batch_size is not None:
            train_steps=len(X)//batch_size #//is integer division
            if train_steps*batch_size<len(X):
                train_steps+=1

            if validation_data is not None:
                validation_steps=len(X_val)//batch_size

                if validation_steps*batch_size<len(X_val):
                    validation_steps+=1
        for epoch in range(1,epochs+1):
            print(f'epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None:
                    batch_X=X
                    batch_y=y
                else:
                    batch_X=X[step*batch_size:(step+1)*batch_size]
                    batch_y=y[step*batch_size:(step+1)*batch_size]

                output=self.forward(batch_X, training=True)
                data_loss, regularization_loss=self.loss.calculate(output,batch_y, include_regulariztion=True)

                loss=data_loss+regularization_loss

                predictions=self.output_layer_activation.predictions(output)
                accuracy=self.accuracy.calculate(predictions,batch_y)

                self.backward(output,batch_y)
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step==train_steps-1:
                    print(f'step: {step}, acc: {accuracy:.3f}, loss: {loss:.3f} (data loss: {data_loss:.3f}, '+
                          f'reg loss: {regularization_loss:.3f}), lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss= self.loss.calculate_accumulated(included_regularization=True)
            epoch_loss=epoch_data_loss+epoch_regularization_loss
            epoch_accuracy=self.accuracy.calculate_accumulated()

            print(f'training, acc: {epoch_accuracy:.3f}, loss: {epoch_loss:.3f} ('+
                  f'data_loss:{epoch_data_loss:.3f}, reg_loss: {epoch_regularization_loss:.3f}), '+
                  f'lr: {self.optimizer.current_learning_rate}')

            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)

    def forward(self,X, training):
        self.input_layer.forward(X, training)
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output

    def backward(self, output, y):

        #if softmax classifier
        if self.softmax_classifier_output is not None:
            #first call backward method on the combined method to set dinputs property
            self.softmax_classifier_output.backward(output,y)

            #since we'll not call backward methos of the last layer which is softmax, set dinputs in the object
            self.layers[-1].dinputs=self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            return
        #first call the backward method on loss, this will set dinputs property that the last layer will try to access
        self.loss.backward(output,y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def evaluate(self, X_val, y_val,*,batch_size=None):
        validation_steps=1
        if batch_size is not None:
            validation_steps=len(X_val)//batch_size
            if validation_steps*batch_size<len(X_val):
                validation_steps+=1

        self.loss.new_pass()
        self.accuracy.new_pass()
        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            output = self.forward(batch_X, training=False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()
        print(f'validation, acc: {validation_accuracy:.3f}, loss: {validation_loss:.3f}')

    def get_parameters(self):
        parameters=[]
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters

    def set_parameters(self, parameters):
        for parameter_set, layer in zip(parameters,self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self,path):
        with open(path,'wb')as f:
            pickle.dump(self.get_parameters(),f)

    def load_parameters(self,path):
        with open(path,'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self,path):
        model=copy.deepcopy(self)
        model.loss.new_pass()
        model.accuracy.new_pass()

        model.input_layer.__dict__.pop('output',None)
        model.loss.__dict__.pop('dinputs',None)

        for layer in model.layers:
            for property in ['inputs', 'output','dinputs','dweights','dbiases']:
                layer.__dict__.pop(property,None)
        with open(path,'wb') as f:
            pickle.dump(model,f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model=pickle.load(f)
        return model

    def predict(self,X,*,batch_size=None):
        prediction_steps=1
        if batch_size is not None:
            prediction_steps=len(X)//batch_size
            if prediction_steps*batch_size<len(X):
                prediction_steps+=1

        output=[]
        for step in range(prediction_steps):
            if batch_size is None:
                batch_X=X
            else:
                batch_X=X[step*batch_size:(step+1)*batch_size]

            batch_output=self.forward(batch_X,training=False)
            output.append(batch_output)

        return np.vstack(output)

