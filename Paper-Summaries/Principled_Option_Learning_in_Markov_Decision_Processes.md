**Principled Option Learning in Markov Decision Processes**

This uses the standard MDP formulation I'm familiar with, but with costs instead
of rewards (ugh). It will do for now. The paper is proposing how to learn
"options". These are individual policies by themselves that solve some subtask.

There are several tricky things about this paper:

- How can we formalize the MDP? It would be very helpful to have rules for when
  we can use options, and what we can assume from them. Also, how do prior
  options come into play?

- I don't understand why the "option" definition is one that says there exists a
  termination state. Shouldn't it just be a subpolicy that we follow with a
  higher-level controller?

- How can we extend this beyond the grid domain which looks very constrained?

- Fastest way to understand these update equations? Some of them are not clear
  to me and are taken straight from previous work.

There are several definitions of "options" stated in the paper. Let's combine
them as follows:

- An option is a policy in a special domain where the agent can choose the
  halting action $a_{\rm term}$ that terminates the episode.
- An option is a pre-learned routine behavior that can be invoked by a
  high-level controller to solve some subtask. [Section 3]
- A good option should therefore be used by many subtasks and it should be
  similar to the optimal solution for each of these subtasks. [Section 3]

And don't forget, THETA = SUBTASKS!!! It's not for anything else!

Fortunately I have some code to run for this.
