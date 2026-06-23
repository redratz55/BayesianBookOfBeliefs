---
date: 2026-04-29
categories:
  - puzzling-problems

---

# A Displaced Passenger
Welcome to Puzzling Problems. This will be a recurring section on fun math problems. 

The first problem will actually build into a 3-part problem. We'll start with a basic case and question and work towards a general form. 
<!-- more -->
# The Problem

Consider 100 passengers boarding an airplane with 100 seats, where passenger $i$ is assigned to seat $i$ for $i = 1, \dots, 100$.

The first passenger is drunk and chooses their seats uniformly at random from the set of all unoccupied seats, without regard to their assigned seat.

Each subsequent passenger $r = 2, \dots, 100$ follows the rule:

- If seat $r$ is unoccupied, passenger $r$ sits in seat $r$.

- Otherwise, passenger $r$ chooses a seat uniformly at random from the set of remaining unoccupied seats. 


What is the probability passenger 100 sits in their assigned seat?


# Solution

This is a fun and classic brainteaser. It might seem tempting to try and derive a general solution to the probability that any passenger sits in their assigned seat given a number of drunks. 

But we'll leave that for a later date.

Instead we'll solve this relatively simply by thinking about a simpler case and working up from their. 

Imagine their are two passengers, one of which is drunk. 

We'll label the drunk passenger $D$. 

When $D$ makes their seat selection they only choose between the correct seat or the wrong seat. Which leaves the second passenger either the correct seat or the incorrect seat with probability $p(C) = \frac{1}{2}$. 

If we then add a second sober passenger, then if $D$ chooses the wrong seat, they displace one of the other passengers. 

Assuming they displace the second passenger, this passenger now acts as the new drunk and we are back to the single drunk passenger case. 

We can see this pattern will hold as we increase the number of sober passengers. 

In essence, the $D$ either selects their correct assigned seat, or they displace another creating a chain of displaced passengers. 

For the 100th passenger, the choice of $D$ only matters if one of two seats are chosen: 

$$
\{1, 100\}
$$

As we'll see in part 2, we can generalize this choice to passenger r given $D$ for $r > D$. 

Considering our set of choices we arrive at the correct probability:

$$
p(\text{Correct Seat}) = \frac{1}{2}
$$

