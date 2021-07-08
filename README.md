# CNN_for_ImageProcessing-Using_Keras
Image classification is the process of segmenting images into different categories based on their features. A feature could be the edges in an image, the pixel intensity, the change in pixel values, and many more

#### What is Image Classification?<hr>
* Image classification is the process of segmenting images into different categories based on their features. A feature could be the edges in an image, the pixel intensity, the change in pixel values, and many more.A same individual person however varies when compared across features like the color of the image, position of the face, the background color, color of the shirt, and many more. 
* The biggest challenge when working with images is the uncertainty of these features. To the human eye, it looks all the same, however, when converted to data you may not find a specific pattern across these images easily.

* An image consists of the smallest indivisible segments called pixels and every pixel has a strength often known as the pixel intensity. Whenever we study a digital image, it usually comes with three color channels, i.e. the Red-Green-Blue channels, popularly known as the “RGB” values. Why RGB? Because it has been seen that a combination of these three can produce all possible color pallets. Whenever we work with a color image, the image is made up of multiple pixels with every pixel consisting of three different values for the RGB channels
<br><br><br>
#### Understanding Image Dimensions<hr>
    image.shape
    (400, 400, 3)
    Note:

The Shape of the image is 400 x 408 x 3 where 400 represents the height, 400 the width and 3 represents the number of color channels. When we say 400 x 400 it means we have 1,60,000 pixels in the data and every pixel has a R-G-B value hence 3 color channels

<br><br><br>
    image[0][0]
    array([193, 159, 113], dtype=uint8)
    Note:
          image [0][0] provides us with the pixel and 193, 159, 113 are the R-G-B values
          
<br><br><br>        
#### The output of gray.shape is 400 x 400. What we see right now is an image consisting of 1,60,000 odd pixels but consists of one channel only.

When we try and covert the pixel values from the grayscale image into a tabular form this is what we observe.
    import numpy as np
    data = np.array(gray)
    flattened = data.flatten()
    flattened.shape

        Output: (192600,)
We have the grayscale value for all 192,600 pixels in the form of an array.

<br><br><br>
flattened
array([149, 147, 160, ..., 156, 137,  53], dtype=uint8)
Note a grayscale value can lie between 0 to 255, 0 signifies black and 255 signifies white.

* Now if we take multiple such images and try and label them as different individuals we can do it by analyzing the pixel values and looking for patterns in them. However, the challenge here is that since the background, the color scale, the clothing, etc. vary from image to image, it is hard to find patterns by analyzing the pixel values alone. Hence we might require a more advanced technique that can detect these edges or find the underlying pattern of different features in the face using which these images can be labeled or classified. This where a more advanced technique like CNN comes into the picture.

<br><br><br>
#### What is CNN?<hr>

* CNN or the convolutional neural network (CNN) is a class of deep learning neural networks. In short think of CNN as a machine learning algorithm that can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image, and be able to differentiate one from the other.

* CNN works by extracting features from the images. Any CNN consists of the following:
  1 - The input layer which is a grayscale image
  2 - The Output layer which is a binary or multi-class labels
  3 - Hidden layers consisting of convolution layers, ReLU (rectified linear unit) layers, the pooling layers, and a fully connected Neural Network

It is very important to understand that ANN or Artificial Neural Networks, made up of multiple neurons is not capable of extracting features from the image. This is where a combination of convolution and pooling layers comes into the picture. Similarly, the convolution and pooling layers can’t perform classification hence we need a fully connected Neural Network.

Before we jump into the concepts further let’s try and understand these individual segments separately.
![TEST_IMAGE]('https://miro.medium.com/max/2000/0*BVil_XCudTACe0vD.jpeg')




