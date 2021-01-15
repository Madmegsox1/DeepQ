## `Agent()` :
#### Field Summary:

| Type  | Field |
| ------------- | ------------- |
| float  | Learning rate  |
| float  | Gamma  |
| int | Number of actions |
| float | Epsilon |
| int | batch size |
| int | input dimensions |
| float | epsilon end |
| int | memory size |
| string | model save name |
#### Constructor Summary
`Agent(gamma, epsilon, lr, input_dims, n_actions, mem_size, batch_size, epsilon_end)`

## `Agent.choose_action()`:
#### Field Summary:
| Type  | Field |
| ------------- | ------------- |
| environment variables (depends on what you are trading the agent on)  | observation |
#### Constructor Summary
`Agent.choose_action(observation)`

## `Agent.store_transition()`:
#### Field Summary:
| Type  | Field |
| ------------- | ------------- |
| float | state  |
| float | action |
| float | reward |
| float | new state |
| bool | done |

#### Constructor Summary
`Agent.store_transition(state, action, reward, new_state, done)`

## `Agent.learn()`:
#### Field Summary:
| Description  |
| ------------- |
| Runs the data is has collected through the neural network |

#### Constructor Summary
`Agent.learn()`

## `Agent.save_model`:
#### Field Summary:
| Description  |
| ------------- |
| saves the model|

#### Constructor Summary
`Agent.save_model()`

## `Agent.load_model`:
#### Field Summary:
| Description  |
| ------------- |
| loads the model|

#### Constructor Summary
`Agent.loads_model()`
