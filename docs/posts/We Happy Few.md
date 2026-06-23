---
date: 2026-04-30
categories:
  - puzzling-problems

---

# We Happy Few
This is part 2 of our drunk passenger problems. 

Today we'll see how we can extend the our part 1 to determine the probability of any passenger being displaced.
<!-- more -->

# The Problem

Consider 100 passengers boarding an airplane with 100 seats, where passenger $i$ is assigned to seat $i$ for $i = 1, \dots, 100$.

The first passenger is drunk and chooses their seats uniformly at random from the set of all unoccupied seats, without regard to their assigned seat.

Each subsequent passenger $r = 2, \dots, 100$ follows the rule:

- If seat $r$ is unoccupied, passenger $r$ sits in seat $r$.

- Otherwise, passenger $r$ chooses a seat uniformly at random from the set of remaining unoccupied seats. 


A passenger is considered happy if they are seated in their assigned seat. 

What is the expected number of happy passengers?


# Solution

If we can reach back to the previous version of this problem we were asked to find the probability that passenger 100 sat in their correct assigned seat. 

In that problem we were able to identify that the 100th passengers outcome is only affected by two seats outcomes:

$$
\{1, 100\}
$$

Or passenger 100's probability only depends on the actions of the drunk passenger, $D$, and their unresolved seat. 

What if we were considering the probabilities for the 99th passenger?

Well, this passenger still depends on the position of the drunk. However, they also are concerned with the set of unresolved seats. 

That is the drunk passenger could choose to sit in their seat, or the 99th or the 100th. In the event of sitting in the 100th, the 99th is resolved successfully and passenger 100 fails. So we can observe there are 2 possible outcomes that are successful for the 99th passenger:

Let $N$ be the number of total passengers, we have the probability that a passenger, $r$, is displaced as: 

$$
P(displaced) = \frac{D}{D + N - r + 1}
$$

For the 99th this resolves to:

$$
P_{99}(displaced) = \frac{1}{100 - 99 + 2} = \frac{1}{3}
$$

We can observe the sample space for a given passenger is the union of the drunk passenger's seat and the unresolved seats that the drunk passenger could choose from: 

$$
\{D\} \cup \{r, \dots, N\}
$$

Now to compute the number of happy passengers we can just compute:

$$
E[Displaced] = \frac{99}{100} + \sum_{k=2}^{N} \frac{1}{102 - k} = 5.177
$$

Giving us our ultimate result: 

$$
E[Happy] = 100 - E[Displaced] = 100-5.177 
$$

\[
    \approx \boxed{94.823}
\]




