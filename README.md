This biometric authentication system works is based on the FaceNet by google.
I use simple one-two neural network over face embeddings from FaceNet.

You need a few photos of a person (next I will call them "right person"), the system passes photos through FaceNet and now theese are the vectros of size 512 with float values. Also you need to get the embeddings of other people, need only about 30 identities. These people will be "wrong".

Then I generate the "secret key" of each person, it's just a random binary vector of length 256.

Then I train a one layer "nn" to predicts secret keys. In fact one layer with 512 inputs and 256 outputs is just an ensemble of logistic regression. I train this nn without gradient descent, but with LSM (least squares method, look it at wikipedia).

The result system is FaceNet + one linear layer. It predicts the right secret key of the "right" person and the random binary vector on any other face.

Probability of type I error usualy less than 0.01.
Probability of type II error is always 0.
So this system never lets strange people and almost lets known people.