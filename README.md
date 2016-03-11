# MachineLearningProject

Few things need to be set up before runing the code
 - install scikit-learn package
   - (sudo) pip install -U scikit-learn
 - In MachineLearningProject folder, create a new folder named 'data', download all files in dropbox directory MSF-ML-RA-Project/results/rbm_random_forest/
   - So the layout is MachineLearning/data/rbm_random_forest/(AD, BP, CL...)
 - To add new strategy:
   -create new strategy class in strategy folder
   -SimpleStrategy is an example to look at
 - To generate portfolio, use RBMRamdomforest
 - Use Main.py to create results for all instruments
