
layout: page
title: Galaxy 
order: 5
permalink: /galaxy/
---


* TOC
{:toc}


# Motivation for ML

## HES-Subjects Example

Let's start with a simple question: Given the marks of a student in English and Hindi, can we predict the marks they might get in Sanskrit? What if we try to see it as a linear combination? 

![HES](/machinelearning/assets/images/English.png)


Think about it like this — can we find a combination of English and Hindi marks that equals the Sanskrit marks? For example, could it be that:

- `0.2 * English + 0.8 * Hindi = Sanskrit`  
- Or maybe `0.6 * English + 0.4 * Hindi = Sanskrit`?

Now, here's something to ponder: we only looked at two subjects, but what happens if we throw in more subjects? What if we have 3, 4, 5, or even 100 subjects? Do you see how the complexity of this question ramps up significantly?

As the number of subjects increases, it becomes harder and harder to find the right combination. The challenge here is to figure out if we can express the marks in one subject as a linear combination of the others. How would you approach this problem when dealing with multiple subjects?

---

## Radio Example: Parameter Tuning

Let's take a trip back to the 1930s. Imagine you're tuning an old radio with a knob controller. To find the right station, you'd have to slowly tweak the knob, trying to tune into the clearest signal. So in a way, you were performing an algorithm to connect to the best station. 

Now, what do you think happens when you turn the knob clockwise? What if you hear loud noises? Naturally, you'd switch direction and try turning it anticlockwise, right? And if you start hearing faint music, you'd keep turning in that direction to zero in on the best sound.

This is very similar to what we do in Machine Learning. We try to tune our parameters to reduce noise or loss and increase the score. Can you see how this old-school radio tuning is just like parameter tuning in ML?

You start with one direction, assess the output (in this case, the sound), and then decide whether to continue or adjust your approach. The goal is always to find that sweet spot where everything sounds just right, or in ML terms, where the model performs optimally.

---

## Binary Search: The Laptop Thief

Imagine you're given a 4K resolution video that runs for 4.5 hours. The video shows a study table, and at the beginning, there's a laptop on the table. But by the end of the video, the laptop is gone—someone has stolen it. Here's the challenge: you need to find the exact moment when the laptop was taken. But watching the entire 4.5 hours of footage seems impractical, doesn't it?

What's the best way to tackle this? 

A smart approach would be to jump to the middle of the video and see if the laptop is still there. If it is, then the theft must have occurred in the second half. If it's not, then the first half is where the action happened. 

What would you do next? You'd continue this process, checking the middle of the relevant section each time until you pinpoint the exact moment when the laptop was picked up.

Why is this method so efficient? It's because you're using a binary search technique, which allows you to find the moment in `log(n)` time. Instead of scanning through every second of the video, you're drastically cutting down the time it takes to find the key moment. Can you see why this method is so powerful?

---

## COVID Testing: Minimizing the Number of Tests

Consider a scenario where a clinic needs to test a group of 10 people for COVID-19. Individual testing is expensive, so the clinic needs an efficient strategy to minimize costs. How can they do this?

The best approach is to minimize the number of tests while still accurately identifying infected individuals. Here's how it can be done:

1. **Initial Batch Testing**: The lab assistant divides the 10 test samples into two groups of 5. She then mixes the samples of each group and tests them in two separate test tubes.
   - If the first batch tests positive, she knows that someone in this group is infected. The second batch, which tests negative, can be ignored—saving 5 tests!

2. **Further Subdivision**: Now, the assistant takes the positive batch and divides it into smaller groups, perhaps 2 and 3 samples each, and tests them again. By following this process, she narrows down the infected individual(s).

3. **Efficiency**: This method is highly efficient because it halves the search space in every iteration. The assistant only conducts the minimum number of tests necessary to identify the infected person(s).

Can you figure out the minimum number of tests required for any number of samples, $$n$$? 

It's `log(n)`. This is where you can see the binary search in action!

To make it even more efficient, you could experiment with different batch sizes, like grouping $$k$$ samples together instead of just 2. This could further reduce the number of tests required.


---

## Finding $$\sqrt{11}$$ 

Finding the square root of 11 is equivalent to finding the root of the equation     

$$x^2 - 11 =0 $$

Note that one can observe that the root lies between 3 and 4, simply because considering $$ f(x) = x^2 - 11 $$, $$f(3)<0$$ and $$f(4)>0$$.

This means the answer lies between 3 and 4.

One can further observe that the root lies between 3 and 3.5 and not between 3.5 and 4.

We can keep reducing the size of the interval this way. 

Every time, we reduce the interval by half. 

It is easy to see that we can find the value of $$\sqrt{11}$$ with increased accuracy with time.

**Observe: The rate at which we converge to the answer is logarithmic (why?)**
**Reason: Every step leads to a reduction in one of the numbers by half**

---


# Markov Processes

Let's start by looking at a simple Markov process involving two states: Happy and Stressed.

![Markov Process with Two States: Happy and Stressed](/machinelearning/assets/images/markov1.png)

In the figure above, we have two states: Happy and Stressed.

Here’s what we know:
- 70% of people in the Happy state will become Stressed after some time (one iteration), while 30% will stay Happy.
- 50% of people in the Stressed state will become Happy after some time (one iteration), while 50% will stay Stressed.

Now, consider an initial scenario where there are 1000 people in the Happy state and 0 in the Stressed state. What will the distribution of people be after several iterations? Will it remain the same?

Let's try to do this for two iterations:

![Distribution After Two Iterations](/machinelearning/assets/images/markov2.png)
![Distribution After Two Iterations](/machinelearning/assets/images/markov3.png)
![Distribution After Two Iterations](/machinelearning/assets/images/markov4.png)

Do you notice how the distribution changes after two iterations?

Will it keep changing, or will it eventually stabilize?

Yes, it will converge after some time. But do you understand why?

To get a better understanding, you can write a program to see if this distribution converges over time. Here’s a simple code snippet to help you explore this:

```python
import numpy as np
A = np.array([[0.3, 0.5],[0.7, 0.5]]) # Transition matrix
v = np.array([1000, 0]) # Initial distribution
def iterate_until_convergence(A, v, tolerance=1e-6, max_iterations=10):
    for _ in range(max_iterations):
        v_next = A @ v
        if np.allclose(v, v_next, atol=tolerance):
            return v_next
        v = v_next
    return v
final_distribution = iterate_until_convergence(A, v)
total_population = np.sum(final_distribution)
percentage_distribution = final_distribution / total_population * 100
print(final_distribution, percentage_distribution)

```

Notice how the distribution changes with each iteration. Eventually, you'll see that it stops changing—it converges.

## Matrix Method

There's another way to approach this: through matrix operations.

Think of the probability of transitioning from one state to another as a matrix, and the initial distribution of people in both states as a vector.

Here's what the matrix looks like:

$$
\text{Transition Matrix: } M = 
\begin{pmatrix}
0.7 & 0.5 \\
0.3 & 0.5
\end{pmatrix}
$$

_Notice that the sum of the columns is 1._

And here’s the initial distribution vector:

$$
\text{Initial Vector: } v_0= 
\begin{pmatrix}
1000 \\
0
\end{pmatrix}
$$

Now, to predict the distribution after one iteration, you multiply the matrix with the vector:

$$
\text{New Distribution: } v_1 = 
\begin{pmatrix}
0.7 & 0.5 \\
0.3 & 0.5
\end{pmatrix}
\begin{pmatrix}
1000 \\
0
\end{pmatrix}
=
\begin{pmatrix}
700 \\
300
\end{pmatrix}
$$

So, to find the eventual distribution, would you keep performing matrix multiplications repeatedly?

![EVD](/machinelearning/assets/images/EVD1.png)

But do you realize that you don't need to do the matrix multiplication all the time? You just need to find $$M^{n}$$.

But $$M$$ is a matrix, so how easily can you compute $$M^{n}$$?

This is where EVD (Eigenvalue Decomposition) comes to our rescue! 

EVD, or Eigenvalue Decomposition, allows us to express the matrix $$M$$ in terms of its eigenvalues and eigenvectors.

We know that:

$$
A v = \lambda v
$$

where $$v$$ is the eigenvector and $$\lambda$$ is the corresponding eigenvalue.

![EVD](/machinelearning/assets/images/EVD2.png)

Multiplying a matrix by a vector $$v$$ $$n$$ times is equivalent to raising the matrix to the power $$n$$.

![EVD](/machinelearning/assets/images/EVD3.png)


Now, do you realize the importance and power of eigenvectors? They can be used to find any power of $$A$$ in no time.

Over time, this matrix multiplication will lead to a steady-state distribution, where further iterations will not change the result. This is the point of convergence.

So for the above example, we get the final distribution to be:

$$
\begin{bmatrix}
416.666368 \\
583.333632
\end{bmatrix}
$$


## Application of Markov Matrices: Opening a New Restaurant

Imagine you want to start a new restaurant in the city. Where should you situate it? Near the railway station? Near the IT hub? Near schools? 

The best way to determine the ideal location is by using the Markov matrix technique. Here's the idea:

By analyzing the movement of people throughout the day, you can figure out where the maximum number of people eventually gather. This will be the hotspot for your restaurant.

Using a Markov matrix, you can model the probabilities of people moving from one location to another over time. As you apply the matrix iteratively, you'll see that the distribution of people converges to a steady state. 

The location with the highest steady-state probability is where people are most likely to gather, making it the best spot for your restaurant. This method allows you to make a data-driven decision, increasing the chances of your restaurant's success.

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

---

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

---


# QR Factorization


![QR Factorization](/machinelearning/assets/images/note1.png)

![QR Factorization](/machinelearning/assets/images/note2.png)

![QR Factorization](/machinelearning/assets/images/note3.png)

![QR Factorization](/machinelearning/assets/images/note4.png)

_Credits: Lakshay_

---


# Hoeffding's inequality


**Question: Walk into Amritsar and estimate the ratio of men to women for a population of $$10^6$$.**
Write a code and observe this.



```
# create a list of length L of 1 million of randomly distributed numbers between one and zero with probability 0.21

import random

l = 10000
result = [0 if random.random() < 0.21 else 1 for _ in range(l)]
print(result)
```

```
#create another list with ith entry being the proportion of 0s seen so far, meaning, upto the ith position in the original list
# this can be done by using the cumulative sum of the original list, and then dividing by the index

result3 = [sum(result[:i])/i if i != 0 else 0 for i in range(1, l)]
print(result3)
```

```
# create_list(p_zero, n): "Generate a Python function in Colab to create a list of length n where each element is either 0 or 1, with a given probability p_zero for zeros."
def create_list(p_zero, n):
    import random
    return [random.choices([0, 1], weights=[p_zero, 1-p_zero])[0] for _ in range(n)]
# cumulative_proportion_ones(L): "Create a Python function in Colab that takes a list of 0s and 1s and returns the cumulative proportion of 1s at each position."
def cumulative_proportion_ones(L):
    return [sum(L[:i+1])/(i+1) for i in range(len(L))]

# create parameters p_zero, n, n_small, window_size
p_zero = 0.21
n = 10000
n_small = 10
window_size = 5

#Plotting Cumulative Proportion: "Write Python code in Colab to plot the cumulative proportion of 1s from a list generated with a specific zero probability using Matplotlib."
import matplotlib.pyplot as plt
L = create_list(p_zero, n)
plt.plot(cumulative_proportion_ones(L))
plt.show()

# Error Calculation: "Generate Python code in Colab to calculate and plot the error between the cumulative proportion of 1s and the expected value over time."
# Calculate errors
errors = [abs(prop - (1-p_zero)) for prop in cumulative_proportion_ones(L)]

# Plot errors
plt.plot(errors)
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Error between Cumulative Proportion of 1s and Expected Value')
plt.show()

```


You have a biased coin. How or when will you get to know that the coin is biased, or how many steps will it take to know that the coin is biased?

It is given by:

$$
Pr(|\mu - \gamma| > \epsilon) \leq \frac{2}{e^{2 \epsilon^{2} N}}
$$

![Hoeffding’s Inequality](/machinelearning/assets/images/he.png)

Here, $$\epsilon$$ can be anything like 0.001 or 0.01, but on the right-hand side, $$\epsilon$$ is squared:

$$
\epsilon^2 \to 0
$$ 

This makes it very small, pulling the denominator of the right-hand side closer to 1, as you might expect! But **N** (the number of samples) pulls it up.

![Hoeffding’s Inequality](/machinelearning/assets/images/he2.png)

So, the chances of the difference between the true mean and your estimated mean being greater than $$\epsilon$$ are:

$$
\leq \frac{1}{e^{2 \epsilon^{2} N}}
$$

How do we apply this logic to our question?

You can see that with a very small **_N_**, convergence occurs quickly. Do you realize that the whole of this process is dependent on **N** and that there is an exponential dip?


**Hoeffding's Inequality and its Relation with _N_**

```
import random
import matplotlib.pyplot as plt
import time

def create_list(p_zero, n):
    L = []
    for _ in range(n):
        if random.random() < p_zero:
            L.append(0)  # Probability p_zero of appending 0
        else:
            L.append(1)  # Probability 1 - p_zero of appending 1
    return L

def cumulative_proportion_ones(L):
    cumulative_proportion = []
    cumulative_sum = 0
    for i in range(len(L)):
        cumulative_sum += L[i]
        cumulative_proportion.append(cumulative_sum / (i + 1))
        # print(f"Progress: {i}, {cumulative_sum/(i+1)}")
    return cumulative_proportion

# Parameters
p_zero = 0.21  # Probability of 0
n = 100000   # Length of the list for full plot
n_small = 100000 # Length of the list for error calculation
window_size = 10  # Number of errors to include in each average calculation

# Create the list
L = create_list(p_zero, n)

# Calculate the cumulative proportion of 1's
cumulative_ones = cumulative_proportion_ones(L)

# Plot the cumulative proportion of 1's over time
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(cumulative_ones, label=f'Proportion of 1s (p_zero = {p_zero})')
plt.axhline(y=1 - p_zero, color='r', linestyle='--', label='Expected Proportion of 1s')
plt.xlabel('Number of Elements')
plt.ylabel('Cumulative Proportion of 1s')
plt.title('Cumulative Proportion of 1s Over Time')
plt.legend()

# Smaller list for error calculation
L_small = create_list(p_zero, n_small)

# Calculate the cumulative proportion of 1's
cumulative_ones_small = cumulative_proportion_ones(L_small)

# Expected proportion of 1's
expected_proportion = 1 - p_zero

# Calculate the error between the cumulative proportion and the expected proportion
error = [abs(cumulative_ones_small[i] - expected_proportion) for i in range(n_small)]

# Plot the error over time
plt.subplot(1, 2, 2)
plt.plot(error, label='Error between Cumulative Proportion and Expected Proportion', color='orange')
plt.xlabel('Number of Elements')
plt.ylabel('Error')
plt.title('Error Between Cumulative Proportion of 1s and Expected Value')
plt.legend()
plt.show()

```

<!-- **Question: You are given a coin whose bias, which is unknown. How many times should you toss the coin so that you are confident of your estimate with a margin of 0.01?** -->


Project:
A company has 100 employees, each rated in 10 distinct skills. Each employee's rating in a skill reflects their experience, expertise, and comfort in that skill, with values ranging from 0 to 100. The company wants to divide its employees into two groups of 50 each, such that the total skill levels for each group are identical across all 10 skills.

Your task is to divide the employees into two equal-sized groups (50 each) such that for every skill, the sum of the skill ratings in Group 1 is equal to the sum of the skill ratings in Group 2.

![Hoeffding’s Inequality](/machinelearning/assets/images/he1.png)


# Perceptron Learning Algorithm

## Update Rule

![PLA](/machinelearning/assets/images/PLA1.jpg)
![PLA](/machinelearning/assets/images/PLA2.jpg)
![PLA](/machinelearning/assets/images/PLA3.jpg)
![PLA](/machinelearning/assets/images/PLA4.jpg)
![PLA](/machinelearning/assets/images/PLA5.jpg)
![PLA](/machinelearning/assets/images/PLA6.jpg)


![PLA](/machinelearning/assets/images/PLA7.png)
![PLA](/machinelearning/assets/images/PLA8.png)

# SVD

![PLA](/machinelearning/assets/images/SVD1.png)

<video width="640" height="360" controls>
  <source src="https://github.com/sudarshansudarshan/machinelearning/assets/images/lena.mp4?raw=true" type="video/mp4">
</video>

![PLA](/machinelearning/assets/images/SVD2.png)
![PLA](/machinelearning/assets/images/SVD3.png)
![PLA](/machinelearning/assets/images/SVD4.png)
![PLA](/machinelearning/assets/images/SVD5.png)



---

_First draft notes by Aashik Arun Bobade. For any corrections, contact: bobadeaashik@gmail.com._
