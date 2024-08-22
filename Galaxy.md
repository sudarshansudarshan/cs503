---
layout: page
title: Galaxy 
order: 5
permalink: /galaxy/
---

* TOC
{:toc}


# Motivation for ML

## 1) HES-Subjects Example

Let's start with a simple question: Given the marks of a student in English and Hindi, can we predict the marks they might get in Sanskrit? What if we try to see it as a linear combination? 



Think about it like this—can we find a combination of English and Hindi marks that equals the Sanskrit marks? For example, could it be that:

- `2 * English + 5 * Hindi = Sanskrit`  
- Or maybe `3 * English + 1 * Hindi = Sanskrit`?

Now, here's something to ponder: we only looked at two subjects, but what happens if we throw in more subjects? What if we have 3, 4, 5, or even 100 subjects? Do you see how the complexity of this question ramps up significantly?

As the number of subjects increases, it becomes harder and harder to find the right combination. The challenge here is to figure out if we can express the marks in one subject as a linear combination of the others. How would you approach this problem when dealing with multiple subjects?

---

## 2) Radio Example: Parameter Tuning

Let's take a trip back to the 1930s. Imagine you're tuning an old radio with a knob controller. To find the right station, you'd have to slowly tweak the knob, trying to tune into the clearest signal. So in a way, you were performing an algorithm to connect to the best station. 

Now, what do you think happens when you turn the knob clockwise? What if you hear loud noises? Naturally, you'd switch direction and try turning it anticlockwise, right? And if you start hearing faint music, you'd keep turning in that direction to zero in on the best sound.

This is very similar to what we do in Machine Learning. We try to tune our parameters to reduce noise or loss and increase the score. Can you see how this old-school radio tuning is just like parameter tuning in ML?

You start with one direction, assess the output (in this case, the sound), and then decide whether to continue or adjust your approach. The goal is always to find that sweet spot where everything sounds just right, or in ML terms, where the model performs optimally.

---

## 3) Binary Search: The Laptop Thief

Imagine you're given a 4K resolution video that runs for 4.5 hours. The video shows a study table, and at the beginning, there's a laptop on the table. But by the end of the video, the laptop is gone—someone has stolen it. Here's the challenge: you need to find the exact moment when the laptop was taken. But watching the entire 4.5 hours of footage seems impractical, doesn't it?

What's the best way to tackle this? 

A smart approach would be to jump to the middle of the video and see if the laptop is still there. If it is, then the theft must have occurred in the second half. If it's not, then the first half is where the action happened. 

What would you do next? You'd continue this process, checking the middle of the relevant section each time until you pinpoint the exact moment when the laptop was picked up.

Why is this method so efficient? It's because you're using a binary search technique, which allows you to find the moment in `log(n)` time. Instead of scanning through every second of the video, you're drastically cutting down the time it takes to find the key moment. Can you see why this method is so powerful?

---



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

But what if we want to find the shortest distance of a specific point, say point A, from a given line like $$ y = \frac{x}{2} $$? How would you do that?

The answer is simple: it's the perpendicular drop from point A onto the line.

![Perpendicular Drop from Point A to the Line](/machinelearning/assets/images/image3.png)

This is known as the projection of vector $$ p $$ onto the line $$ y = \frac{x}{2} $$.

But how can you figure out a generic method for finding this projection?

The dot product offers a straightforward approach. Consider a point $$ v $$ on this line and take the dot product of $$ v $$ with $$ p $$. For instance, if $$ v = (1, 2) $$, the dot product gives us the necessary information to determine the projection.

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

So, the projection of point A onto the line $$ y $$ is given by:

$$
B = (\alpha \cdot 2, \alpha \cdot 1) = (2.8, 1.4)
$$

![Projection of Point A on the Line](/machinelearning/assets/images/image4.png)

This is a simple yet powerful method. But there's more to it.

Let's think about matrices. What if we take this concept further? Consider the equation:

$$
A(A \cdot \bar{x} - b) = 0
$$

To solve for $$ \bar{x} $$, we rearrange it as:

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


![QR Factorization](/machinelearning/assets/images/note1.png)

![QR Factorization](/machinelearning/assets/images/note2.png)

![QR Factorization](/machinelearning/assets/images/note3.png)

![QR Factorization](/machinelearning/assets/images/note4.png)

_Credits: Lakshay_




---

_First draft notes by Aashik Arun Bobade. For any corrections, contact: bobadeaashik@gmail.com._
