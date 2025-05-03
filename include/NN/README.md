Two Parent objects that will drive the creation of the Nueral Net:

# Layer

- This object will create a single Layer (Input, output, activation)
  - When creating a network you will use a vector these objects. 
  - These objects arrtibutes will be checked inside the Network Class to make sure the inputs and outputs line up.


# Network

- This object will create a network that flows information from layer to layer.
  - It will receive an array of Layer Objects and create a vectorized network based on the attributes of the layers.
  - This object will apply the BackPropogration and ForwardPropation Classes/Methods within its Training and Predict Method.
