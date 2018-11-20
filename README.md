# ML Applications Auditory Streaming EEG

Logistic regression, linear SVMs, decision trees and nearest neighbour classifiers were compared to determine the best method to discriminate between 'standard' and 'deviant' stimuli responses in EEG data collected during an auditory streaming experiment. The standard and deviant responses were extracted from the EEG time series, and each sample was individually transformed into its frequency domain using Fourier transforms, The frequencies and amplitudes were used as features to learn from. Despite having a very disproportionate representation of both classes (around 19000 standard samples vs 600 deviants), logistic regression resulted in <10% error rates in classification of deviant stimuli responses, and no errors for classifying standards. This varied depending on the experimental setup.

![Multiple classifiers were tested on the data](images/mmn_classify.PNG?raw=true "Classifier method")
