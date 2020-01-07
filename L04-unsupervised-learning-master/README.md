# L04 Unsupervised Learning

Download the repository as a zip folder and begin an R project for this lab. The zip folder will contain instructions (repeated below) and a template to get an Rmd file started.

# Overview

The goal of this lab is to continue practicing the application of unsupervised learning techniques.

# Datasets 

We will be utilizing the `USArrests` data (*USArrests.csv*) which is contained in the **data** subdirectory. Students are able to access the appropriate codebook using `USArrests`. We we also be using the `college_reshaped.csv` dataset which contains both categorical and numerical data (found in **data** subdirectory). The dataset was formed using the `College` dataset from the `ISLR` package.

# Exercises

Please complete the following exercises. The document should be neatly formatted. 

<br>

#### Exercise 1 (Ex. 9 Section 10.7 pg 416)
Consider the `USArrests` data. Perform hierarchical clustering on the states.

a. Using hierarchical clustering with complete linkage and Euclidean distance, cluster the states.

<br>

b. Cut the dendrogram at a height that results in three distinct clusters. Which states belong to which clusters? *Challenge: Maybe plotting a map and filling by cluster membership might be a good idea.*

<br>

c. Hierarchically cluster the states using complete linkage and Euclidean distance, after scaling the variables to have standard deviation one.

<br>

d. What effect does scaling the variables have on the hierarchical clustering obtained? In your opinion, should the variables be scaled before the inter-observation dissimilarities are computed? Provide a justification for your answer.


<br><br>

#### Exercise 2
Consider the the `college_reshaped.csv` dataset. Scale the numerical features so that they have a standard deviation of one.

a. Run $K$-means on the data. 
Try different numbers of clusters $K$. 
Does a specific value of $K$ tend to produce better or more distinct clusters?

<br>

b. Run hierarchical clustering. Try different numbers of clusters, and use both the Euclidean distance and complete linkage as dissimilarity metrics. 
Be sure that the number of clusters you use in this exercise is similar to the number of clusters you tried in part (a).
What sort of clusters result? 

<br>

c. Run spectral clustering using the radial kernel. Set the number of clusters for the algorithm equal to the number of clusters you found useful in parts (a-b). Do you obtain different clusters than those algorithms?

<br>

d. Use the `cluster` package (specifically the `daisy()` & `pam()`) to perform clustering. Again, use the same number of clusters you used on part (a). Do you obtain different clusters?

<br>

e. Discuss how similar cluster membership is for parts (a-d). What are some reasons that clusters are similar? Why would they be different? In your opinion, do clusters from any one algorithm seem better or more intuitive for this data?