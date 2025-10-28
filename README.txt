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

Q4- Q7

AI Use:
Matthew - 

Miguel Mateo - I used AI as a suplemental resource to brainstorm