import random


class BlackjackAgent:
    def __init__(self):
        """
        Initializes the Blackjack agent.

        Attributes:
            q_table (dict): A table to store the state-action values.
            learning_rate (float): The rate at which the agent learns from new experiences.
            discount_factor (float): The factor by which future rewards are discounted.
            exploration_rate (float): The rate at which the agent explores random actions.
            actions (list): Possible actions, where 0 is 'stand' and 1 is 'hit'.
        """
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.1
        self.actions = [0, 1]

    def action(self, observation, exploit_only=False):
        """
        Decides the action to take based on the current observation and exploration strategy.

        Parameters:
            observation (object): The current state of the game environment.
            exploit_only (bool): If True, the agent will only exploit known information
                                 without exploration. Default is False.

        Returns:
            int: The action chosen by the agent (0 for 'stand', 1 for 'hit').
        """
        if not exploit_only and random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.actions)

        state = self._get_state(observation)
        return self._get_best_action(state)

    def learn(self, action, observation, reward, terminated, next_observation):
        """
        Updates the agent's knowledge based on the outcome of its action.

        Parameters:
            action (int): The action taken by the agent.
            observation (object): The current state of the game environment before the action.
            reward (float): The reward received after taking the action.
            terminated (bool): Whether the game has ended after the action.
            next_observation (object): The state of the game environment after the action.

        Returns:
            float: The error in the Q-table update, representing how much the Q-value has changed.
        """
        current_state = self._get_state(observation)
        next_state = self._get_state(next_observation)

        # Update Q-table
        old_value = self.q_table.get((current_state, action), 0)
        next_max = max(self.q_table.get((next_state, a), 0) for a in self.actions)

        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * next_max
        )
        self.q_table[(current_state, action)] = new_value

        error = abs(old_value - new_value)
        return error

    def _get_state(self, observation):
        """
        Transforms the observation into a state representation.

        Parameters:
            observation (object): The current state of the game environment.

        Returns:
            str: A string representation of the state.
        """
        return str(observation)

    def _get_best_action(self, state):
        """
        Determines the best action for a given state based on the Q-table.

        Parameters:
            state (str): The current state of the game.

        Returns:
            int: The best action to take in the given state.
        """
        values = [self.q_table.get((state, a), 0) for a in self.actions]
        return self.actions[values.index(max(values))]


# Additional methods can be added for specific Blackjack strategies and calculations
