The repository, algorithmic-bias, contains materials related to the Confusion Matrix Dashboard:
http://www.saund.org/cm-dashboard/CMDashboard.html

The Confusion Matrix Dashboard is an interactive tool for viewing the relationship between
prediction scores, distributions of positive and negative outcomes, decisision thresholds,
and the Confusion Matrix.  The Confusion Matrix Dashboard also plots Positive Prediction Ratio Curves
and calculates a PPRS (Positive Prediction Ratio Score).

See also the article,
 "Algorithmic Bias and the Confusion Matrix Dashboard:
 How a Confusion Matrix Behaves Under Distributions of Prediction Scores"


Repository Contents:

compasAnalysis.py     - python code for evaluating the COMPAS/ProPublica data, reporting results, and building our own predictive model

compasAnalysis.ipynb  - Jupyter notebook for walking through use of the code

appleSnacks.py        - python code for generating synthetic data about kids' preferences for an apple snack versus other snack.

appleSnacks.ipynb     - Jupyter notebook for running the appleSnacks data generator and computing PPRS.

data                  - Broward Recidivism data produced by ProPublica.
                        Also Titanic data exported by a separate module that adds a bit of
			feature engineering to the Kaggle Titanic data set.
			



