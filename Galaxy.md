---
layout: page
title: Galaxy 
order: 5
permalink: /galaxy/
---

* TOC
{:toc}

### 1. Birthday Paradox
Given a bunch of 30 students in a class, there is a high chance that you have 2 people with the same birthday. Instead of 30, if the class has n students, what are the chances that you will find two people with the same birthday? 

>Try writing a piece of python code which plots the probability Vs n 

We observe that the probability of we not having two people with the same birthday in a class of $$n$$ people is given by : 

$$\frac{365}{365}.\frac{364}{365}.\frac{363}{365}\cdots \frac{(365-n)}{365} $$   

(There is a minor error in the above product. Can you identify?)

Let us call this $$f(n)$$.

$$f(n)=\left({1-\frac{0}{365}}\right).\left({1-\frac{1}{365}}\right).\left({1-\frac{2}{365}}\right)... \left({1-\frac{n}{365}} \right) $$

We know that $$\left( 1-x\right)\approx e^{-x}$$


Therefore, 

$$f(n) \approx e^{\frac{-1}{365}}.e^{\frac{-2}{365}}.e^{\frac{-3}{365}}...e^{\frac{-n}{365}} $$   

$$f(n)\approx e^{\frac{-n(n+1)/2}{365}}$$

This simply tells us that for around 50 students in a class
$$f(50)\approx e^{\frac{-50(51)/2}{365}}=0.03$$

The probability of we finding two people with the same birthday is $$0.97$$

- Can you observe this empirically for 50 students? is the probability of success 0.97?
- Can you think of some application of this idea?
- When you put 50 balls in 365 baskets, uniformly at random, do you think that you can surely find a basket with atleast 2 balls? 
- What do you have to say about birthday paradox's relevance on planet Neptune where one year comprises of 60,000 days :-) ?



### 2. Finding $$\sqrt 10$$ 

Finding square root of 10 is equivalent to finding the root of the equation     

$$x^2 - 10 =0 $$

Note that one can observe that the root lies between 3 and 4, simply because considering $$ f(x) = x^2 - 10 $$, $$f(3)<0$$ and $$f(4)>0$$.

This means the answer lies between 3 and 4.

One can further observe that the root lies between 3 and 3.5 and not between 3.5 and 4.

We can keep reducing the size of the interval this way. 

Everytime, we reduce the interval by half. 

It is easy to see that we can find the value of $$\sqrt{10}$$ with increased accuracy with time.

**Observe: The rate at which we converge to the answer is logarithmic (why?)**

### 3. Finding GCD
``` python
def gcd(a,b):
	if (a<b):
		return gcd(b,a)
	else:
		if (b==0):
			return a
		else:
			return gcd(b,a%b)
```

Note that the above snippet of code finds the GCD of two numbers in lightening speed. It only takes logarithm in the $$ min(a,b) $$.

**Reason: Every step leads to a reduction in one of the numbers by half**


