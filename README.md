# Image-correlation
Image analysis and basic machine learning work

</p>
Xinqi Chen @17/04/2023 

## Table of Content
- [Image-correlation](#iamge-correlation)
  - [Abstract](#abstract)
  - [Overview](#overview)
  - [Theoretical Background](#theoretical-background)
  - [Algorithm Implementation and Development](#algorithm-implementation-and-development)
  - [Computational Results](#computational-results)
  - [Summary and Conclusions](#summary-and-conclusions)
  - [Acknowledgement](#acknowledgement)
  
## Abstract
This project explores the use of eigenvectors and singular value decomposition (SVD) for analyzing and representing a dataset of images. The dataset consists of images represented as pixel intensities, and the goal is to compute the eigenvectors and SVD modes that capture the most significant variation in the images. The project uses Python and popular numerical libraries such as NumPy and SciPy for implementing the algorithms and visualizing the results.
  
## Overview
The project concerns the correlation of the yalefaces data set, with computing the eigenvectors and eigenvalues of the covariance matrix or the matrix formed by multiplying the dataset with its transpose. Then, analyze and visualize the eigenvectors and SVD modes to understand their significance and interpret the results.
  
## Theoretical Background
Eigenvalues and eigenvectors are fundamental concepts in linear algebra that have applications in various fields, including image processing and computer vision. Eigenvectors represent the directions in which a matrix stretches or compresses a vector, while eigenvalues represent the scaling factors along those directions. In the context of image analysis, eigenvectors can capture the significant modes of variation in a dataset of images, and eigenvalues represent the amount of variation captured by each eigenvector.

Singular value decomposition (SVD) is another powerful technique in linear algebra that decomposes a matrix into three matrices: U, S, and V^T, where U and V are orthogonal matrices and S is a diagonal matrix with singular values. SVD can be used to compute the eigenvectors and eigenvalues of a dataset matrix, and it has applications in dimensionality reduction, image compression, and data analysis.

## Algorithm Implementation and Development 
The project uses Python and the following libraries for implementing the algorithms:

NumPy: for numerical computations, including matrix operations, eigenvalue and eigenvector computations, and singular value decomposition.

SciPy: for additional numerical computations, including image processing and visualization.

The main steps of the algorithm implementation are as follows:

Load the dataset of images and preprocess them, such as normalizing pixel intensities and reshaping the images into a suitable format.

Compute the covariance matrix or the matrix formed by multiplying the dataset with its transpose. Find wich two image are most highly correlated and most highly uncorrelated.
```ruby
i, j = np.unravel_index(np.argmax(C - np.eye(C.shape[0])*np.max(C)), C.shape)
k, l = np.unravel_index(np.argmin(C + np.eye(C.shape[0])*np.max(C)), C.shape)
```

Compute the eigenvectors and eigenvalues of the covariance matrix or the dataset matrix using NumPy's eig function.
```ruby
# Create the matrix Y
Y = X.T @ X

# Compute the first six eigenvectors with the largest magnitude eigenvalue
eigvals, eigvecs = np.linalg.eig(Y)
idx = np.argsort(eigvals)[::-1]
eigvecs = eigvecs[:, idx[:6]]
```

Compute the SVD of the dataset matrix using NumPy's svd function.

Analyze and visualize the results, including plotting the eigenvectors and SVD modes to understand their significance and interpreting the results.
```ruby
# Percentage of variance captured by each mode
variance = (S ** 2) / (S ** 2).sum()
```

The implementation code is provided in the project's Python script, along with comments to explain the key steps and computations.

## Computational Results
### Part a
The correlation matrix of the Yaleface data set is plotted as below:

![part a result](https://github.com/Mmmo-C/Image-correlation/blob/main/results/a.png)

### Part b
The two most highly correlated images and the two most highly uncorrelated images are:

![part b result](https://github.com/Mmmo-C/Image-correlation/blob/main/results/b.png)

### Part c
The selected 10x10 correaltion matrix [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005] is plotted as below:
![part c result](https://github.com/Mmmo-C/Image-correlation/blob/main/results/c.png)

### Part f
The norm of difference of the first eigenvector v1 from (d) with the first SVD mode u1 from (e) is: 1.419911450662362e-15

### Part g
The the percentage of variance captured by each of the first 6 SVD modes is:[0.72927567 0.15281763 0.02566745 0.01877525 0.00639306 0.00592431]

The first 6 SVD modes can be plotted as below:
![part g result](https://github.com/Mmmo-C/Image-correlation/blob/main/results/g.png)

## Summary and Conclusions
This project demonstrates the use of eigenvectors and SVD for analyzing and representing a dataset of images. The computed eigenvectors and SVD modes can capture the significant modes of variation in the images and provide a useful basis for representing the data. The project's implementation code and results can serve as a starting point for further exploration and experimentation with eigenvectors, SVD, and image analysis.

## Acknowledgement
- [Numerical Recipes in C: The Art of Scientific Computing, Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (2007).]
- [ChatGPT](https://platform.openai.com/)
- [Scipy Documentation](https://docs.scipy.org/doc/scipy/)

