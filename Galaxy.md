---
layout: page
title: Galaxy 
order: 5
permalink: /galaxy/
---

* TOC
{:toc}


# Least Squares

## Subjects Example

Let's dive into an interesting question: What does it mean to take points in 3D and project them onto a 2D plane? Imagine you have a table with marks in Physics, Math, and Programming.

$$
\text{Programming} = \alpha \cdot \text{Physics} + \beta \cdot \text{Math} + \text{Some noise}
$$

Now, if you're given the marks in Physics, Math, and Programming, can you plot them on a plane? What would that look like?

Observe the following table:

![Marks in Physics, Math, and Programming](/machinelearning/assets/images/image1.png)

Take a look at this plot. You'll notice that not all the points lie perfectly on the plane, but they are very close to it. This suggests that we can approximate these points using this plane.

![3D Points Approximated by a Plane](/machinelearning/assets/images/image5.png)

But how do we find the best possible plane? Well, to do that, we need to minimize the distance of the points from the plane.

![Minimizing the Distance from the Plane](/machinelearning/assets/images/image6.png)

Can you think of how we might achieve that? How would you go about finding the plane that best fits the data?

## Projection of Points

Now, let's shift our focus to the concept of projection. Observe the following graph: some points lie on or close to a plane.

![Points Lying Close to a Plane](/machinelearning/assets/images/image2.png)

But what if we want to find the shortest distance of a specific point, say point A, from a given line like $$\( y = \frac{x}{2} \)$$? How would you do that?

The answer is simple: it's the perpendicular drop from point A onto the line.

![Perpendicular Drop from Point A to the Line](/machinelearning/assets/images/image3.png)

This is known as the projection of vector $$\( p \)$$ onto the line $$\( y = \frac{x}{2} \)$$.

But how can you figure out a generic method for finding this projection?

The dot product offers a straightforward approach. Consider a point \( v \) on this line and take the dot product of $$\( v \)$$ with $$\( p \)$$. For instance, if $$\( v = (1, 2) \)$$, the dot product gives us the necessary information to determine the projection.

Would you like to see the math behind it? Here's how it works:

$$
v \cdot (\alpha v - b) = 0
$$

$$
\alpha \cdot v^T v = v^T b
$$

$$
\alpha = \frac{v^T b}{v^T v}
$$

Substituting the values:

$$
\alpha = \frac{\begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 3 \end{bmatrix}}{\begin{bmatrix} 2 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \end{bmatrix}} = \frac{7}{5} = 1.4
$$

So, the projection of point A onto the line $$\( y \)$$ is given by:

$$
B = (\alpha \cdot 2, \alpha \cdot 1) = (2.8, 1.4)
$$

![Projection of Point A on the Line](/machinelearning/assets/images/image4.png)

This is a simple yet powerful method. But there's more to it.

Let's think about matrices. What if we take this concept further? Consider the equation:

$$
A(A \cdot \bar{x} - b) = 0
$$

To solve for $$\( \bar{x} \)$$, we rearrange it as:

$$
A^T A \bar{x} = A^T b
$$

$$
\bar{x} = (A^T A)^{-1} A^T b
$$

And then:

$$
A \bar{x} = A(A^T A)^{-1} A^T b
$$

You see how everything falls into place, right? But here's a critical point: this works perfectly if \( A \) is invertible. Now, think about the real world—most matrices are invertible, aren’t they? So, we don’t usually have to worry about invertibility here. But what would happen if a matrix wasn't invertible? How would that affect our calculations?


# QR Factorization



