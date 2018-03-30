# eye_detector
eye detection using SVM and HOG 

![picture](test/1.png)

![picture](test/2.png)

First, I used a support vector machine to do binary classification to divide the eye data and not the eye data.

And I created a detector by creating windows of various sizes.

In the given test picture, the eye was detected correctly, although it was a bit inaccurate.

But when I tried other pictures, I did not get good results.

It seems to be caused by a very small amount of data sets.(about 400 eye data and about 300 non-eye data)

It was difficult to get a dataset, so there was a limit when I made it myself.

I think support vectors machines will perform very well and will produce good results if the dataset is large enough.

Finally, I tried to make a simple neural net, but it seems that it does not work well because of lack of time. I'll update the results later.
