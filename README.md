# Prerequisite
* python3.10

# Environment Setup

```sh
conda create --name myenv --file spec-file.txt
```

# How to train?
```sh
conda activate myenv
python train.py
```

# How to inference?
```sh
python inference.py --input <image-path> --model <model.pth path>
```

# Project Overview
![]("./result.jpg")

This is a simple implementation of FacialKeypointsDetection(aka. FKD). I used a data from
[Facial-Keypoint-Detection-Udacity-PPB](https://github.com/ParthaPratimBanik/Facial-Keypoint-Detection-Udacity-PPB?tab=readme-ov-file).

We had a very small training dataset of only 3,462 samples, so we couldn’t build a model with high complexity. Even with a simple model, overfitting was severe: the training loss was extremely low, while the validation loss was very high.

To address this overfitting issue, we applied the following:
	1.	Reduce model complexity by lowering the number of filters (thus fewer parameters).
	2.	Add Dropout layers and adjust their dropout rates.
	3.	Perform data augmentation.
	4.	Adjust the learning rate.
	5.	Adjust the batch size.

Applying each of these individually did not yield a significant effect, but after combining some of them and increasing the number of FC layers at the end, we resolved the overfitting problem.

## Final Solution

We reduced the number of layers in the model and decreased the number of filters, then increased the FC layers to two stages. We also added a Dropout layer before passing through the FC layers to prevent overfitting. Our rationale was that using only one FC layer to directly map the CNN features to coordinates—without any intermediate non-linear step—would cause the model to overfit more easily. We assumed the single FC weight matrix would “memorize” the patterns excessively. After that, we lowered the learning rate and introduced a weight decay value. Weight decay is an L2 regularization method that penalizes the weight gradient updates so that they decay over time. When w grows large, the L2 term increases, causing the loss to rise and discouraging the model from inflating the weights unnecessarily

```latex
\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{\partial \mathcal{L}}{\partial \mathbf{w}} - \alpha \lambda \mathbf{w}
```

Finally, we adjusted the batch size. When the batch size is large, gradient estimates—based on the average over more samples—are more stable, and convergence can be faster. However, because there are fewer optimizer updates overall, the gradient direction can be determined in a single large step. Conversely, if the batch size is small, fewer samples per update mean larger variance in the gradient, which makes the model less “certain” and thus can improve generalization performance.
