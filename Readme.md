# Value Iteration Agent for Taxi Environment

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements for this assignment.

```bash
pip3 install -r requirements.txt
```

## Usage

```python
cd agent
python3 agent.py
```

## The Taxi Problem


### Description
    There are four designated locations in the grid world indicated by R(ed),
    G(reen), Y(ellow), and B(lue). When the episode starts, the taxi starts off
    at a random square and the passenger is at a random location. The taxi
    drives to the passenger's location, picks up the passenger, drives to the
    passenger's destination (another one of the four specified locations), and
    then drops off the passenger. Once the passenger is dropped off, the episode ends.

###Map:

        +---------+
        |R: | : :G|
        | : | : : |
        | : : : : |
        | | : | : |
        |Y| : |B: |
        +---------+

### Actions
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east
    - 3: move west
    - 4: pickup passenger
    - 5: drop off passenger

### Observations
    There are 500 discrete states since there are 25 taxi positions, 5 possible
    locations of the passenger (including the case when the passenger is in the
    taxi), and 4 destination locations.

    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi

    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    
### Rewards
    - -1 per step unless other reward is triggered.
    - +20 delivering passenger.
    - -10  executing "pickup" and "drop-off" actions illegally.

### Objective
    Your aim is to train a taxi agent using Value Iteration to pick up the passenger
    and drop him to his destination while maximising the rewards received.

### What to expect
    Fill in only the parts in the agent.py file below the comments with "FILL ME:" instructions
    Do not make any changes to any other methods or classes. While running locally, your agent will be tested on a single testcase. 
    You can debug and see the behaviour of your agent on a pygame window as well as staments printed to your python console or terminal.
    Do not forget to install requirements in the requirements.txt file before proceeding.

### Submission
    Zip the agent.py file such that the file structure is agent.zip/agent.py
    Submit the agent.zip file
