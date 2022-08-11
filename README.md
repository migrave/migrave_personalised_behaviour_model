# sequence-learning-dataset
Data Collection and Analysis for the Sequence Learning task

Code from : https://github.com/TsiakasK/sequence-learning-dataset

TO-DO:
* Look into HCI files.
* The representation of the data is sparse e.g.: [0.25, 1.0, 0.0, 0.0, -1.0] 1.0 , that means [length, no_feedback, feedback 1, feedback 2, previous score], can be substituted by [length, feedback_type, previous score]. - One hot encoding seems to be much better: https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
* Performance simulation:
  * There is very few data, maybe it is better to use random forest, gaussian processes or sth similar, instead of neural network for creating user models? Tried but do not neural network seem to overfit the best.
  * Data has to be normalized before feeding into the neural network. - Done, improves Neural Network training.
  * Changed activation functions to ELU in NN to reduce problem of vanishing gradient (actually not sure, can tanh cause vanishing gradient?), or avoid dead relu.
  * Simplify the problem -> change regression problem into classification problem. Suppose user can give good answer and bad answer [-1, 1] and user can be disengaged, neutral or engaged [-1, 0, 1]. For that I can just get the majority vote for the certain state. - Better approach is to regress the probability of success for the user, when it comes to engagement, it would be indeed better, but it is not possible with the provided dataset (all engagement values are in the middle of the range).
  * **Does the normalization for small amount of data make sense?, do normalization only for NN and SVM!**
* Feedback simulation:
  * **Remove the normalization of results!**
  * **Does the normalization for small amount of data make sense?, do normalization only for NN and SVM!**
  * I can change it to regression problem of the probability that the user is engaged/disengaged.
* Q-learning:
  * Do not obtain reward from engagement, use is for Q-augmentation! - **Update: Q-augmentation makes training worse!**
  * Guidance learning with Shared Control is really fast (optimal policy after 1 game), but only 3/4 states are explored.
  * Guidance learning with Shared Control and pretrained policy is slower than single guidance learning but still fast (optimal policy after 11 games) for the engaged user. Unfortunatelly, policy is not adjusted for the disengaged user (return is very low: 3/-1 (for the engaged user it is around 34)). This is also the case when policy is pretrained and no guidance learning is applied. Maybe Q-augmentation is the solution?
* MigrAVE dataset:
  * No other kernel for GP works better (DotProduct*RationalQuadratic is slightly better but less smooth), nn overfits but gives bad peaks (to the bottom) in difficultylevel 1
  * Try different rewards to encourage robot to give more difficult sequences, right now it is stuck with sequence 5.
  * No update of the policy (guidance+pretrained) is the limited capability of exploring the state space and updating q-table (we visit only those states that are pointed out by the supervisor), thus old Q in maxQ'-Q might hinder the learning, that is why reward shaping might help here (we decided for big reward 1/alpha when the correction is given).