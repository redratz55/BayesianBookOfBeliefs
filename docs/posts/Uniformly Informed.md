---
date: 2026-06-21
categories:
  - puzzling-problems

---

# A Uniform Problem

Today's blog we'll tackle the Fiddler on the Proof puzzle for this weekend. 

<!-- more -->

Fiddler on the Proof is Zach Wissner-Gross' spiritual successor to his longstanding 538 series The Riddler.

If you aren't a subscriber to his substack I'd recommend you do, the puzzle's are quite nice. 

## The Puzzle:
I think the random number generator on my calculator might be malfunctioning. Oh no!

Under normal conditions, it should generate random numbers between 0 and 1. But my suspicion is that the calculator is “tanked,” meaning it only generates random numbers between 0 and some value 0 < a < 1. Beyond that, I have no knowledge regarding the value of a. At the moment, it’s equally likely to be any value from 0 to 1.

As an experiment, I ask the calculator to generate one random number. It produces a value of exactly 0.5. (While this is, admittedly, infinitely unlikely, let’s roll with it!)

Based on this result, what can I expect the value of a to be, on average?

## Solving

We have a uniform random-number generator that should generate a number

$$
X\sim \operatorname{Uniform}(0,1).
$$

However, we suspect that the generator is faulty and instead generates numbers uniformly between (0) and some unknown upper bound (A), where (0<A<1).

We perform one draw and observe

$$
X=0.5.
$$

We want to find

$$
\mathbb E[A\mid X=0.5].
$$

Our model is

$$
A\sim \operatorname{Uniform}(0,1),
\qquad
X\mid A=a\sim \operatorname{Uniform}(0,a).
$$

The general conditional density of (X) given (A=a) is

$$
\frac1a\mathbf 1_{{0\le x\le a}}.
$$

After observing (X=0.5), the likelihood as a function of (a) is

$$
\frac1a\mathbf 1_{{0.5\le a\le1}}.
$$

Any value (a<0.5) is impossible because a generator supported on ([0,a]) could not have produced (0.5). For $(a\ge0.5)$, the density of a uniform distribution on ([0,a]) is

$$
\frac{1}{a-0}=\frac1a.
$$

We now combine the likelihood with the prior to obtain the posterior and then compute its expectation.

Because

$$
A\sim \operatorname{Uniform}(0,1),
$$

the prior density is

$$
f_A(a)=\mathbf 1_{{0<a<1}}.
$$

Equivalently, $(f_A(a)=1)$ on ((0,1)) and (0) otherwise.

Unnormalized posterior

By Bayes' rule,

$$
f_{A\mid X}(a\mid0.5)
\propto
f_{X\mid A}(0.5\mid a)f_A(a).
$$

Therefore,

$$
f_{A\mid X}(a\mid0.5)
\propto
\frac1a\mathbf 1_{{0.5\le a\le1}}.
$$

To obtain a proper probability density, we need a normalizing constant (C) such that

$$
C\int_{0.5}^{1}\frac1a,da=1.
$$

Thus,

$$
C[\ln a]_{0.5}^{1}=1,
$$

so

$$
C\bigl(\ln 1-\ln 0.5\bigr)=1.
$$

Because

$$
\ln 1-\ln 0.5=\ln 2,
$$

we have

$$
C\ln 2=1 \implies C=\frac1{\ln 2}.
$$

The normalized posterior density is therefore

$$
\frac{1}{a\ln 2}
\mathbf 1_{{0.5\le a\le1}}.
$$

Equivalently,

$$
\begin{cases}
\dfrac{1}{a\ln 2}, & 0.5 \le a \le 1,\\
0, & \text{otherwise}.
\end{cases}
$$

Finally, the posterior expectation is

$$
\int_{0.5}^{1}
a\frac{1}{a\ln 2},da.
$$

The (a) terms cancel, giving

$$
\frac1{\ln 2}\int_{0.5}^{1}1,da.
$$

Therefore,

$$
\frac{0.5}{\ln 2}=
\frac{1}{2\ln 2}
\approx
\boxed{0.7213}.
$$