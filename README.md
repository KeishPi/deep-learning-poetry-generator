# deep-learning-poetry-generator
Neural Network model using the Keras library to generate Shakespearean poems

- See Andromeda_Final_Report.pdf for detailed information about our neural network models and how to run them.
- The README in the Project-Files directory also has instructions on how to run the programs.
- We've created a website so anyone could get generated poems from our trained model.  See it here:
   http://ec2-18-217-70-169.us-east-2.compute.amazonaws.com/
   
   This project was the Capstone project for Keisha Arnold, Carrie Treeful, and Jacob Karcz.  We set out to create a poetry generator using a design based on deep learning and neural networks.  It was primarily research based as none of us had any experience with the subject matter or the associated tools and libraries.  This steep learning curve turned out to be one of the main challenges   as there was a lot to learn in a limited amount of time.  After several  weeks, we were able to successfully create a neural network model using the Keras library and train it on Shakespearean sonnets.  This model produces text output that is close to a Shakespearean sonnet, but the model training doesn’t capture a sonnet’s distinct rhyming and meter structure.  To do that we also wanted to implement a model in TensorFlow by attempting to  replicate Miyamoto and Cho’s description of a gated word-char   LSTM*.  Though their paper was quite detailed, there weren’t many similar models implemented in TensorFlow.  That coupled with learning the TensorFlow API made it a slow and at times frustrating process.  Unfortunately, although we were tantalizingly close we   could not get this model to output text so it remains a work in progress.
   * Miyamoto, Y. and Cho, K. (2016). Gated Word-Character Recurrent Language Model. [online] Available at: https://arxiv.org/abs/1606.01700


