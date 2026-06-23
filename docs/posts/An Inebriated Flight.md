---
date: 2026-05-01
categories:
  - puzzling-problems

---

# An Inebriated Flight
Part 3 of our drunk passenger problems.

In this problem we'll bring our series to a close and find a way to generalize the expected value calculation to multiple drunk passengers.
<!-- more -->

# The Problem

Consider 50 passengers boarding an airplane with 50 seats, where passenger $i$ is assigned to seat $i$ for $i = 1, \dots, 50$.

The first 10 passengers are drunk and choose their seats uniformly at random from the set of all unoccupied seats, without regard to their assigned seats.

Each subsequent passenger $r = 11, \dots, 50$ follows the rule:

- If seat $r$ is unoccupied, passenger $r$ sits in seat $r$.

- Otherwise, passenger $r$ chooses a seat uniformly at random from the set of remaining unoccupied seats. 


What is the expected number of passengers sitting in the correct seats?


# Solution
Let

\[
N = 50
\]

be the total number of passengers, and let

\[
D = 10
\]

be the number of drunk passengers.

The first \(10\) passengers choose randomly, so by linearity of expectation for each drunk passenger, the probability of sitting in their assigned seat is

\[
\frac{1}{50}.
\]

Therefore, the expected number of drunk passengers sitting correctly is

\[
10 \cdot \frac{1}{50} = \frac{1}{5}.
\]

Now consider a sober passenger \(r\), where

\[
11 \le r \le 50.
\]

When passenger \(r\) boards, the only seats that still matter are

\[
\{1,2,\dots,10\} \cup \{r,r+1,\dots,50\}.
\]

The seats

\[
\{11,12,\dots,r-1\}
\]

have already resolved. They either contain their assigned passenger, or they have been consumed by the displacement chain.

So the relevant unresolved set has size

\[
10 + (50-r+1).
\]

Passenger \(r\) is displaced if the chain reaches one of the original drunk seats before it reaches seat \(r\). There are \(10\) original drunk seats, so

$$
 P(\text{passenger } r \text{ is displaced}) =\frac{10}{10 + 50 - r + 1}.
$$

Thus,

$$
P(\text{passenger } r \text{ sits correctly}) =1 -\frac{10}{10 + 50 - r + 1}.
$$

Simplifying,

$$
P(\text{passenger } r \text{ sits correctly})=\frac{50-r+1}{10 + 50-r+1}.
$$

Therefore, the expected number of correct sober passengers is

\[
\sum_{r=11}^{50}
\frac{50-r+1}{10+50-r+1}.
\]

Including the drunk passengers, the total expected number of passengers sitting correctly is

$$
E[\text{correct}]=\frac{10}{50}+\sum_{r=11}^{50}\frac{50-r+1}{10+50-r+1}.
$$

Equivalently,

$$
E[\text{correct}]=\frac{1}{5}+\sum_{r=11}^{50}\frac{51-r}{61-r}.
$$

Now let

\[
k = 51-r.
\]

As \(r\) runs from \(11\) to \(50\), \(k\) runs from \(40\) down to \(1\). So we can rewrite the expectation as

$$
E[\text{correct}]=\frac{1}{5}+\sum_{k=1}^{40}\frac{k}{k+10}.
$$

Numerically, this gives

\[
E[\text{correct}]
\approx 24.518.
\]

So the expected number of passengers sitting in their assigned seats is

\[
\boxed{24.518}
\]


