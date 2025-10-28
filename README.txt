Matthew West & Miguel Mateo Osorio Vela

https://github.com/mwest17/proj3-logic-planning

Engineering Process:

For Q1, the construction of the first 3 sentences familiarized us with the syntax and operations 
we used during the project. The key part of this question was the entails(), here we had to ask ourselves: 
What is an Unsatisfiable_condition( or inconsistent) between a premise and conclusion that are supposed to be true?
Unsatisfiable_condition = premise & ~(conclusion)
We searched for those models, if we dont find such model we can say the premise entails the conclusion.

For Q2- the key logic was in the implementation of at most one() where we created a list where we stored all
pairs of literal. In each pair, at most one element could be true, in other words: ~(l1 & l2)
Then we applied DeMorgans law and got: ~l1 | ~l2, pairs were appended using that expression.
If one of the pairs outputs FALSE it meant that at that pair both elements were true, which violates the atmost one principle.

Q3 was straightforward until we encountered a problematic edge case: t=0. We can not call the successors axiom from this timestep
because it will try to reference time =-1, which doesnt make sense. We handled it in check_location_satifiablity()
by appending to KB what we knew its valid without calling the successor axiom. Additionally, we implemented a condtional of t!=0 before appending succesors.

Q4 was also pretty straightforward. We first implemented the pseudocode, which got us most of the way there. 
The main issue was determining the proper way to implement the transition model.

Q5 followed a similar structure to Q4. The majority of our work for that was in implementing the goal state and the food 
succesor axiom (and how the pacman successor axiom still played a part).

Q6 was a challenge, as the helper functions were not entirely clear at first. Implementing the logic in the actual Q6 function with placeholder helper functions
made it easier to then determine what the helper functions needed to do.

Q7 was even more of a challenge. We went through a few iterations of findProvableWalls. 
Our main issue came from misunderstanding what logic the sensors added to the model. 
Once we were checking the entailment of the correct blocking statements, the function worked as intended.


AI Use:
Matthew - I used AI for python syntax.

Miguel Mateo - I used AI as a suplemental resource to brainstorm