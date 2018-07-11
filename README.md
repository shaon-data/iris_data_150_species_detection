# iris150_data_analysis 
## Download 
For downloading use 
       `git clone https://github.com/ShaonMajumder/iris150_data_analysis.git` 

## Database
### Link: https://archive.ics.uci.edu/ml/datasets/iris
| Question | Answer |
| ------------ | ------------ |
| Data Set Characteristics | Multivariate |
| Number of Instances | 150 |
| Area | Life |
| Attribute Characteristics | Real |
| Number of Attributes | 4 |
| Date Donated | 1988-07-01 |
| Associated Tasks | Classification |
| Missing Values? | No |



### Dataset information
The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant. One class is linearly separable from the other 2; the latter are NOT linearly separable from each other. 

### Attribute Information:
Number of Attributes: 4 numeric, predictive attributes and the class
Predicted attribute: class of iris plant. 
1. sepal length in cm 
2. sepal width in cm 
3. petal length in cm 
4. petal width in cm 
5. class: 
-- Iris Setosa 
-- Iris Versicolour 
-- Iris Virginica

### Summary Statistics:
| property | Min | Max | Mean | SD | Class | Correlation |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| sepal length: | 4.3 | 7.9 | 5.84 | 0.83 | 0.7826 | |
| sepal width: | 2.0 | 4.4 | 3.05 | 0.43 | -0.4194 | |
| petal length: | 1.0 | 6.9 | 3.76 | 1.76 | 0.9490 |high |
| petal width: | 0.1 | 2.5 | 1.20 | 0.76 | 0.9565 | high |

   Class Distribution: 33.3% for each of 3 classes.

## Challenges
Deadline: 8/7/2018

In Depth explanation on each step reffering why steps and model had been taken in Jupyter Notebook
1. EDA
2. Hypothesis Test
3. Linear Regression *
4. Classification *
5. Kmeans *
3. Decisions
6. Graphical Visualization or representation

স্কাটার ম্যাট্রিক্স দেখে তারপর সবচেয়ে বেশি কোরিলেটেড প্যারামিটার আগে বসিয়ে লেস সিগ্নেফিকেন্ট গুলো হায়ার ডায়মেনশনে দেয়া যেতে পারে। এতে ওগুলোর অসরল রৈখিক সম্পর্ক, উচ্চ মাত্রায় সরলরৈখিক হওয়ার সম্ভবনা থাকে।

Challenge Credit: Sabbir vai