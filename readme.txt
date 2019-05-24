The task is to discriminate between 10 mobile device users. For each user, there are 300 accelerometer signal recordings of 1.5 seconds in length. The accelerometer signal is recorded at 100 Hz, thus containing 150 values. The values are recorded on three different axes: x, y, z. Each example is provided in a .csv file with the following format:

timestamp,x-value,y-value,z-value
1399425143089,0.486023,5.588067,7.275979
1399425143098,0.469264,5.566519,7.316082
1399425143108,0.420781,5.550358,7.382521
...
1399425144576,-0.527323,6.221333,6.872556

The provided data set is divided into a training set (200 examples per user) and a test set (100 examples per user). The goal is to train a machine learning model on the training set and apply the model on the test set.
