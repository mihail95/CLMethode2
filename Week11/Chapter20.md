# Semantic Role Labeling

## Semantic Roles
#### Thematic Roles
* **agents** - The volitional causer of an event (breaker/opener)
* **theme** - The participant most directly affected by an event (window/door)

## Diathesis Alternations
#### Thematic Grid
$\theta$-grid / case frame\
*The set of thematic role arguments taken by a verb*

#### Verb alternations / diathesis alternations
Multiple structure realizations

## Semantic Roles: Problems with Thematic Roles
* often roles need to be fragmented
* hard to generalize over

**Solutions**: either **more** or **less** roles

#### Generalized semantic roles (less roles)
More abstract roles like *proto-agent* or *proto-patient*

#### Specific roles (more roles)
Assign each verb/group of similar verbs a separate role - **PropBank** or **FrameNet**

## The Proposition Bank
Each sense of each verb is given a name:\
**Arg0** - proto-agent\
**Arg1** - proto-patioent\
**Arg2** - benefactive, instrument, attribute or end state\
**Arg3** - start point, benefactive, instrument, attribute\
**Arg4** - end point\
**ArgMs** - modification / adjunct meanigs

## FrameNet
Roles are specific to a **frame**

## Semantic Role Labeling (SRL)
### Feature-based Algorithm
### Neural Algorithm

## Selectional Restrictions
### Representing Selectional Restrictions
* Restrict what kinds of arguments a predicate could take
### Selectional Preferences
* Represent the restrictions as preferences, to avoid too strict constraints
### Selectional Association
The relative contribution of a WordNet class to the general selectional preference of a verb (How strong a class is associated with a given verb)
### Selectional Preference via Conditional Probability
### Evaluating Selectional Preferences

## Primitive Decomposition of Predicates
