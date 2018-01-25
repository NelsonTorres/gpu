<h3>[In construction]</h3>

CUDA C++ neural network implementation with RPROP 

currently network runs only one hidden layer (needs improvement using templates)

<h4> To execute after compilation</h4>
 ./gpu &lt;neuron in hidden layer> &lt;threads> &lt;number of epochs> <br/>
  
  * blocks executed = #threads/1024 +1 <br/>
  * threads executed = #threads/#blocks
