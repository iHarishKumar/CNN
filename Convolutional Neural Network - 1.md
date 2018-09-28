



# Neural Network

Consider a simple neural network as shown below.

![](http://www.astroml.org/_images/fig_neural_network_1.png)

With the above image we can deduce three important things:

1. Input layer.
2. Hidden Layer.
3. Output Layer.

Lets get a bit more detailed about these three things.



##### Input Layer

This layer is where the network takes the input in the form of an image, word2vec, etc depending on the application. Lets get bit mathematical. For more information please refer to resources section at the end.



##### Hidden Layer

This is where the magic happens. Lets take an example of a Gaussian RBF (Just to be safe as I've paper on it so I can copy of the content). Lets see how do RBF function looks.

![<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>Y</mi><mi>m</mi></msub><mo>&#xA0;</mo><mo>=</mo><mo>&#xA0;</mo><mo>&#xA0;</mo><munderover><mrow><mo>&#x2211;</mo><msub><mi>w</mi><mi>m</mi></msub><mi>f</mi><mfenced><mrow><mo>&#x2225;</mo><mi>x</mi><mo>-</mo><msub><mi>c</mi><mi>m</mi></msub><mo>&#x2225;</mo></mrow></mfenced></mrow><mrow><mi>m</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover></math>](https://lh3.googleusercontent.com/lgrAL3XfqChF4oi3aLCCj5Zz7xLagjQOfOg6S16GWSPGnLJtikkE6TPOMT8piR7ydKKWJdxzcQHA348J3mE6xvD9m05KOpweF5afA5WjnJW5aknXongA1A30rQLzptnTMdOzBFUt)



where Y<sub>m</sub> is the final output, w<sub>m</sub> is the weight value associated with it, c<sub>m</sub> is the vector for the neuron m.

So, this __**f(x)**__ can be any activation function. Here, we use Gaussian function. So, the equation now looks like below.

![<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mi>Y</mi><mi>m</mi></msub><mo>&#xA0;</mo><mo>=</mo><mo>&#xA0;</mo><munderover><mrow><mo>&#x2211;</mo><msub><mi>w</mi><mi>m</mi></msub><msup><mi>e</mi><mfenced><mfrac><mrow><mo>&#x2225;</mo><mi>x</mi><mo>-</mo><msub><mi>x</mi><mi>m</mi></msub><msup><mo>&#x2225;</mo><mn>2</mn></msup></mrow><mrow><mn>2</mn><msup><mi>&#x3C3;</mi><mn>2</mn></msup></mrow></mfrac></mfenced></msup></mrow><mrow><mi>m</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover></math>](https://lh6.googleusercontent.com/eNY1oFMJ-La2ve0isGz4SeyM1sSBaytkWos1sVUAIGlgfomlD6EXT2pDQJH4aBKNpI-Gc6e8-QwCPdYZ2exv6DT-DASJYqLY01vNo75xhee1Ykmswiqi5B5snjVw9Nvdt_0zWZOo)





This can have multiple layers. We need to make sure that number of hidden layers is around 30%-60% of the total dataset size. If we have more than that, then we can have overfitting of the function and if it is below that it would be underfitting.

##### Output Layer

This layer is the final output of all the hidden layers. Here, we only have one vector which is the weighted sum of the previous layers. 



# Convolution Neural Networks(CNN)

A Wikipedia definition goes like this:

```
In mathematics convolution is a mathematical operation on two functions to produce a third function that expresses how the shape of one is modified by the other.
```

As the definition suggests, the Convolution Neural Networks are kind of similar. Lets get started.

First things first, CNN's belongs to a class of deep, feed-forward neural networks which are mostly used in analysing images.



####Design/Architecture

Consider the following gif,

![](https://raw.githubusercontent.com/iamaaditya/iamaaditya.github.io/master/images/conv_arithmetic/full_padding_no_strides_transposed.gif)

Here, the blue region  acts as an input layer and the green region is an output layer. 

Take away points from the GIF:

1. The input layer in CNN's is typically an image of MxN dimensions. 
2. The sliding window is called the **Kernel/Filters** which is typically a product of weight of the cell and the activation function or any standard function.
3. Each cell in the green region corresponds to the summation of each cell in the sliding window. This is also called **Feature Map**.
4. One kernel function is used to detect a specifice feature.
5. If we want to detect multiple features from the same input, we need to have multiple channels which will have the corresponding output.
6. The dark blue region where the covolution takes place is called **receptive field**.

(Try to relate this with the previous explaination on Neural Network. Hope it clarifies things).

This kind of convolutinon can happen multiple times depending on the use case.



#### 3X3 Convolution

As explained in the previous section, this kernel can operate in any dimension (but less than input image size). Typically the best practice is to use **3x3** convolution. The benefits of using this size is as follows:

1. It helps in detecting more features.
2. The output size of an input is kind of bigger (< dimensions of input) and will have a chance of learning the features which were missed in the first attempt.



#### 1x1 Convolution

This kind of convolution is typically used for reducing the dimensionality of the matrix without actually loosing much information. It can also be used in increasing the dimension, but it is advised not to do the same as it will have lot of redundant information. The following GIF gives a better understanding of it.

![](https://cdn-images-1.medium.com/max/1600/1*37xcqtruaRrQRAKRSYwJBg.png)



#### Receptive Field

This is the region of the input where the sliding window/Kernel points. It is the weighted vector with the size equal to the size of the Kernel.



#### Epochs

The sample picture of CNN which was shown before is just one iteration. We need to do multiple iterations or epochs so that we learn the features better without missing the details in the input as much as possible. This helps in training the network to have better understanding of the features.



#### Pooling in CNN

Consider an image of size 100x100 and a filter of 3x3. Let the desired target be 10x10. The output image size after each epoch will be 98x98, 96x96,……. 10x10. This kind of calculation takes lot of CPU time.

In order to reduce this computation time, the cocept of pooling is used.

Considering the same image and kernel size, the image size after fifth epoch will be 90x90. Now, let the pool size be 9x9, this reduces the output size to 10x10. With this we got the desired output in 6 epochs where as without pooling we will be arriving at the desired output only after 45 epochs. 

In CNN's we typically use Max Pooling. The following image will be helpful in better understanding of Max Pooling.

![](https://d3ansictanv2wj.cloudfront.net/Figure_4-87c227113cdd0b73d842267404d4aa00.gif)



#### Activation Function

It is always helpful in having a good activation function which useful in bridging the gap between the input and the output variables. It introduces the non-linearity in the network. The output layer is the product of input layer, weight associated with it and the activation function(as explained in the Neural Network section) which serves as an input in the corresponding layers in the stack.

In CNN's we typically use **ReLU** as an activation function as it helps in masking out the negative values in a huge matrix. The ReLU function looks like this.

​								        	**f(x) = max(0, x)**



#### Feature Map

This is the building block of the CNN. The feature map for a 28x28 image looks something like this.

![img](https://github.com/iHarishKumar/CNN/blob/master/images/Feature%20Map.png)



#### Feature Engineering (older computer vision)

In good olden days, people were inserting the relationship between the pixels manually to the machine learning algorithms. This technique would not allow the machine to learn by itself. It was also because of the minimal computation power before (will revisit this in the following sections). With this kind of approach, there was a possibility of human errors. 



#### Some formulas for CNN

* Total number of parameters:

  ​					**(NxMxL+1)xK**

  Where **NxM** is the kernel size, **K** is the feature map as output, **L** is feature maps at the input.

* Output size of CNN:

  ​					**((I + 2xP - F) / S) + 1)**

  Where **I** is input size(assuming square), **P** is padding size, **S** is the stride size, **F** is the filter size.



# Sources

* https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f
* https://smerity.com/articles/2016/architectures_are_the_new_feature_engineering.html
* https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
* https://stackoverflow.com/questions/42786717/how-to-calculate-the-number-of-parameters-for-convolutional-neural-network
* For writing formulas, have used MathType extension in Google Docs.
* For creating this document, have used Typora application.

(All the infromation is based on my understanding. Please do suggest me if I can improve in the topics and understanding of the concepts).

