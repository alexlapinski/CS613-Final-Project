For the Water Treatment Plant data:

First use RBF Kernel & One-Class SVM (with hyperplane).

Perform a rough-search of parameters for rbf kernel.
parameters to search for "nu" and "gamma"

Run through 5 iterations in narrowing down the best search space for nu and gamma.
Run through 30 steps within the parameter boundaries, cross validation such that we have training sizes of at least 50.
Result of Rough-Search: Best Parameters w/ f1 = 0.827989498832: {'nu': 0.25, 'gamma': 0.0001220703125}

Then since we used a logspace for the nu values and gamma values, we'll take the log of these best nu and gamma values.
And re-compute a new logspace varying only +1/-1 from the exponent returned from the log.

This new search space is our fine-search and lets us zero in on the exact best params
for the rbf kernel, using the water-treatment data and one-class svm.
We get (after our 10th iteration):

Cross Validate, nu:2^-5.48275862069->2^-3.48275862069; gamma:2^-15.4482758621->2^-13.4482758621 search w/ rbf
Best Parameters w/ f1 = 0.92448962245: {'nu': 0.045807388855747595, 'gamma': 4.367724483401814e-05}


---

Do the same thing as above, but for linear kernel
Now, rather than tuning nu and gamma, we just need to tune 'nu'.

We get the best parameter as:
Best Parameters w/ f1 = 0.653157685775: {'nu': 0.045807388855747595}


---

Do the same thing as above, but for poly kernel
Now rather than tuning nu and gamma, we tune nu, gamma, coef and degree.

Here, we don't have 30 steps for the polynomial degree, and instead look at the integer range from min to max values.



---
Do the same thing as above, but for sigmoid kernel
For this we are tuning 'nu', gamma and coef.


---
