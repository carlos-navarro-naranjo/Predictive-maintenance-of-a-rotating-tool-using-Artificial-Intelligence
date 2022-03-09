# Predictive-maintenance-of-a-rotating-tool-using-Artificial-Intelligence
It employs a simple but effective Artificial Neural Network Model to predict whether a planned combination of conditions will result in a failure in advance of operating in those conditions. It returns Multiple metrics and a confusion matrix to visualize the performance of the model.

Dataset: 
Dataset developed for predictive maintenance of a rotating tool. The idea is to predict whether a planned combination of conditions will result in a failure in advance of operating in those conditions. 
If a failure is predicted, one would service the system before running it. 

The Excel file contains the following columns:

- Column Label	Description
- UDI	A unique identifier for each data example (i.e., each row of dataset)
- Product ID	Alphanumeric code for product serial number
- Type	Categorization of the product as low (L), medium (M) or high (H) quality
- Air Temperature [K]	Temperature of ambient air
- Process Temperature [K]	Temperature of process
- Rotational Speed [rpm]	Tool speed
- Torque [Nm]	Torque applied by tool
- Tool Wear [min]	How long the tool has been in operation
- Target	Binary variable indicating whether a failure occurred. 1 is failure, 0 is no failure.
- Failure Type	String describing failure if one occurred ("No Failure" otherwise)
- The dataset has a total of 10,000 examples, about 3.3% of which are examples of failures.
 





One Hot Encoding using my own defined function before entering the ANN: 

Column 'Type' contains an alphabetic code that we cannot easily input into something like an ANN. I wrote a function called “encode_one_hot”  that produces a one-hot encoding of this input feature.
one-hot encoding uses a series of binary inputs to represent each of the possible discrete input values. In this case, there are three possible values: 'L', 'M', and 'H'. 
My function  accepts a Pandas DataFrame object as input, adds three new columns to the DataFrame and returns the result. Each of the new columns are specified as follows:
•	Column label 'L'. Rows in the column are 1 if 'Type' is 'L' and 0 otherwise.
•	Column label 'M'. Rows in the column are 1 if 'Type' is 'M' and 0 otherwise.
•	Column label 'H'. Rows in the column are 1 if 'Type' is 'H' and 0 otherwise


Artificial Neural Network Model:

(1 hidden layers with 20 nodes, and activation function: tanh)

