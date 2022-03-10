import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x))>=0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        finished = False
        multiplier = 1

        while finished==False:
            allEquals = True
            for x,y in dataset.iterate_once(1):
                #print(x)
                #print(y)
                curResult = self.get_prediction(x)
                #If the program comes in this if this means that we must continue iterating.
                if curResult != nn.as_scalar(y):
                    allEquals = False
                    direction = nn.Constant(nn.as_scalar(y)*x.data) 
                    self.w.update(direction,multiplier)
            finished = allEquals

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate =-0.005
        self.epsilon = 0.02
        self.first_layer_mat = nn.Parameter(1,100)
        self.first_bias = nn.Parameter(1,100)
        self.second_layer_mat = nn.Parameter(100,1)
        self.second_bias = nn.Parameter(1,1)
        self.parameters=[self.first_layer_mat,self.first_bias,self.second_layer_mat,self.second_bias]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        #Computations of the matrices of each level.
        first_level = nn.Linear(x,self.first_layer_mat)
        first_level_biased = nn.AddBias(first_level,self.first_bias)
        first_relued = nn.ReLU(first_level_biased)
        second_level = nn.Linear(first_relued,self.second_layer_mat)
        second_biased = nn.AddBias(second_level,self.second_bias)
        return second_biased

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        averageLess = False
        loss = float(10000000000)
        while averageLess == False:
            loss = 0.0
            number = 0
            for x,y in dataset.iterate_once(1):
                number+=1
                curLoss = self.get_loss(x,y)
                #Add each curLoss to loss and update each matrice by using gradient descent
                loss+=nn.as_scalar(curLoss)
                gradients = nn.gradients(curLoss,self.parameters)
                self.first_layer_mat.update(gradients[0],self.learning_rate)
                self.first_bias.update(gradients[1],self.learning_rate)
                self.second_layer_mat.update(gradients[2],self.learning_rate)
                self.second_bias.update(gradients[3],self.learning_rate)
                self.parameters=[self.first_layer_mat,self.first_bias,self.second_layer_mat,self.second_bias]
            if loss<self.epsilon*number:
                break 


class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.learning_rate = -0.005
        self.accuracy = 0.97
        self.first_layer = nn.Parameter(784,300)
        self.first_bias = nn.Parameter(1,300)
        self.second_layer = nn.Parameter(300,100)
        self.second_bias = nn.Parameter(1,100)
        self.third_layer = nn.Parameter(100,30)
        self.third_bias = nn.Parameter(1,30)
        self.fourth_layer = nn.Parameter(30,10)
        self.fourth_bias = nn.Parameter(1,10)
        #list of every parameters which may be changed by using gradient 
        self.parameters = [self.first_layer, self.first_bias,self.second_layer,self.second_bias,self.third_layer,self.third_bias,self.fourth_layer,self.fourth_bias]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        #Computation of each matrices on each level.
        first_level = nn.Linear(x,self.first_layer)
        first_level_biased = nn.AddBias(first_level,self.first_bias)
        first_relued = nn.ReLU(first_level_biased)
        second_level = nn.Linear(first_relued,self.second_layer)
        second_biased = nn.AddBias(second_level,self.second_bias)
        second_relued = nn.ReLU(second_biased)
        third_level = nn.Linear(second_relued,self.third_layer)
        third_layer_biased = nn.AddBias(third_level,self.third_bias)
        third_relued = nn.ReLU(third_layer_biased)
        fourth_level = nn.Linear(third_relued,self.fourth_layer)
        fourth_biased = nn.AddBias(fourth_level,self.fourth_bias)
        return nn.ReLU(fourth_biased)

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x,y in dataset.iterate_once(10):
                curLoss = self.get_loss(x,y)
                #Count each gradient of each parameter and update them at the same time.
                gradients = nn.gradients(curLoss,self.parameters)
                self.first_layer.update(gradients[0],self.learning_rate)
                self.first_bias.update(gradients[1],self.learning_rate)
                self.second_layer.update(gradients[2],self.learning_rate)
                self.second_bias.update(gradients[3],self.learning_rate)
                self.third_layer.update(gradients[4],self.learning_rate)
                self.third_bias.update(gradients[5],self.learning_rate)
                self.fourth_layer.update(gradients[6],self.learning_rate)
                self.fourth_bias.update(gradients[7],self.learning_rate)
                self.parameters = [self.first_layer, self.first_bias,self.second_layer,self.second_bias,self.third_layer,self.third_bias,self.fourth_layer,self.fourth_bias]
            #If the validation on the dataset is more than the self.accuracy cycle will be finished
            if dataset.get_validation_accuracy() >= self.accuracy:
                break;

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = -0.01
        self.total_languages = 5
        self.connect_dimension = 200
        #Bias of the first level
        self.initial_bias = nn.Parameter(1,self.connect_dimension)
        self.final_bias = nn.Parameter(1,self.total_languages)
        #Matrice initializations for each level of neural networks
        self.initial = nn.Parameter(self.num_chars,self.connect_dimension)
        self.connect = nn.Parameter(self.connect_dimension,self.connect_dimension)
        self.final = nn.Parameter(self.connect_dimension, self.total_languages)
        self.parameters = [self.initial,self.connect,self.final,self.initial_bias,self.final_bias]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #First layer initialization
        h_value = nn.Linear(xs[0],self.initial)
        h_value = nn.AddBias(h_value, self.initial_bias)
        h_value = nn.ReLU(h_value)
        for ch in xs[1:]:
            #Each step of neural network
            h_value = nn.Add(nn.Linear(ch,self.initial),nn.Linear(h_value,self.connect))
            h_value = nn.AddBias(h_value,self.initial_bias)
            h_value = nn.ReLU(h_value)
        #Write the answers in answer_value.
        answer_value = nn.Linear(h_value,self.final)
        answer_value = nn.AddBias(answer_value,self.final_bias)
        return answer_value

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        validation = 0
        while validation < 0.85:
            for xs,y in dataset.iterate_once(10):
                loss = self.get_loss(xs,y)
                #update each matrice by using gradient descent.
                gradients = nn.gradients(loss,self.parameters)
                self.initial.update(gradients[0],self.learning_rate)
                self.connect.update(gradients[1],self.learning_rate)
                self.final.update(gradients[2],self.learning_rate)
                self.initial_bias.update(gradients[3],self.learning_rate)
                self.final_bias.update(gradients[4],self.learning_rate)
                self.parameters = [self.initial,self.connect,self.final,self.initial_bias,self.final_bias]
            validation = dataset.get_validation_accuracy() 
        