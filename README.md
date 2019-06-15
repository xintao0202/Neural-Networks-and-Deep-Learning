# Neural-Networks-and-Deep-Learning
Coursera Neural Networks and Deep Learning by deeplearning.ai
https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome
1. Introduction to deep learning
  - AI is the new electricity: AI is powering personal devices in our homes and offices, similar to electricity
  - Why deep learning take off: we have access to a lot more data; Deep learning has result in significant improvements; we have access to                                 a lot more computational power
  - Diagram iterating over different ML ideas (idea, experiment, code): 
        Being able to try out ideas quicly allows iterate more quickly;
        Faster computation can help speed up iterating a good idea;
        Recent progress in deep learning algorithms allows us train faster
  - Use insight from previous problem: first try use previous problem without need to iterate multiple times
  - Relu Activitaion functions: a=max(0,z)
  - structured data: referes to any data and resides in a fixed field within a record or file, e.g. data in relational database and spreed sheet; unstructured data is all that can't be so readly classified and fit into a neat box, e.g. photos, videos, pdf, webpage, powerpoint, email...
  - Why RNN (recurrent Neutral Network used for machine learning: it can be trained as a supervised learning problem); applicable when the input/output is a sequence; represet the recurrent process of idea->code->experiment->idea->...
  - Performance of algorithm vs amount of data: large NN has better performance with same amount of data
  - increasing the size of a NN does not hurt performance and may help significantly; increasing the training set size doesn't hurt the performance, and it may help significantly
  
  2.1 Neutral Network Basics
  - what does a neuron compute: computes a linear function (z=Wx+b) followed by an activation function
  - Logistic loss: L(y^,y)= -(ylog(y^)+(1-y)log(1-y^))
  - reshape function: e.g. 32 by 32 image with 3 colors: x=img.reshape((32*32*3,1))
  - Broadcasting
  		 <br/>a = np.random.randn(2, 3) # a.shape = (2, 3)
       <br/>b = np.random.randn(2, 1) # b.shape = (2, 1)
       <br/>c = a + b # c.shape=(2,3), + is element-wise operation, first b will copied 3 times to become (2,3)
       <br/><br/>
       a = np.random.randn(4, 3) # a.shape = (4, 3)
       <br/>b = np.random.randn(3, 2) # b.shape = (3, 2)
       <br/>c = a*b # the computation cannot happen because size don't match                                                             
       <br/>a = np.random.randn(3, 3)
       <br/>b = np.random.randn(3, 1)
       <br/>c = a * b
       <br/>What will be c? This will invoke broadcasting, so b is copied three times to become (3,3), and ∗ is an element-wise product         so c.shape = (3, 3).
  - Nx feature, X=[x1,x2,....xm]: the dimension of X is (Nx,m)
  - matrix multiplication: 
  <br/>a = np.random.randn(12288, 150) # a.shape = (12288, 150)
  <br/>b = np.random.randn(150, 45) # b.shape = (150, 45)
   <br/>c = np.dot(a,b) # c.shape=(12288,45)
  - Vectorization:
      <br/># a.shape = (3,4)
      <br/># b.shape = (4,1)
      <br/>for i in range(3):
        <br/>for j in range(4):
          <br/>c[i][j] = a[i][j] + b[j]
      <br/>How do you vectorize this?
      <br/>c = a + b.T
   - computation graph: u=a*b; v=a*c; w=b+c; J=u+v=w  ==> J=(a-1)*(b+c) 
  
  2.2 Logistic Regression with a Neural Network mindset project
  - [Logistic Regression with a Neural Network mindset project Link](Logistic+Regression+with+a+Neural+Network+mindset+v5.ipynb)
