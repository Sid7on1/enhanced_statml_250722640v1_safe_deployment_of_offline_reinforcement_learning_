import logging
import threading
import time
import json
from typing import List, Dict, Tuple

import numpy as np
from agent import Agent
from message_bus import MessageBus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiAgentCommunicator:
    """
    Multi-agent communicator class for facilitating communication between multiple agents.

    ...

    Attributes
    ----------
    agents : list of Agent objects
        List of agent objects for communication.
    message_bus : MessageBus object
        Message bus for publishing and subscribing to topics.
    config : dict
        Configuration settings for the communicator.
    running : bool
        Flag indicating if the communicator is running.
    lock : threading.Lock
        Lock for thread safety.

    Methods
    -------
    start()
        Start the multi-agent communication process.
    stop()
        Stop the multi-agent communication process.
    send_message(sender, message)
        Send a message from a sender agent to other agents.
    receive_message(agent)
        Receive messages for a specific agent.
    update_agent_status(agent, status)
        Update the status of an agent.
    is_running()
        Check if the communicator is running.

    """

    def __init__(self, agents: List[Agent], message_bus: MessageBus, config: Dict):
        """
        Initialize the MultiAgentCommunicator object.

        Parameters
        ----------
        agents : list of Agent objects
            List of agent objects for communication.
        message_bus : MessageBus object
            Message bus for publishing and subscribing to topics.
        config : dict
            Configuration settings for the communicator.

        """
        self.agents = agents
        self.message_bus = message_bus
        self.config = config
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        """
        Start the multi-agent communication process.

        Subscribes to necessary topics and starts consuming messages.

        """
        with self.lock:
            if self.running:
                raise RuntimeError("Communicator is already running")

            self.running = True
            logger.info("Starting multi-agent communication...")

            # Subscribe to required topics
            self.message_bus.subscribe(self.config['input_topic'], self.receive_message)

            # Start consuming messages
            self.message_bus.start_consuming()

    def stop(self):
        """
        Stop the multi-agent communication process.

        Unsubscribes from topics and stops consuming messages.

        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Communicator is not running")

            self.running = False
            logger.info("Stopping multi-agent communication...")

            # Unsubscribe from topics
            self.message_bus.unsubscribe(self.config['input_topic'], self.receive_message)

            # Stop consuming messages
            self.message_bus.stop_consuming()

    def send_message(self, sender: Agent, message: Dict) -> None:
        """
        Send a message from a sender agent to other agents.

        Parameters
        ----------
        sender : Agent object
            Agent sending the message.
        message : dict
            Message to be sent.

        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Communicator is not running")

            logger.info(f"Sending message from agent {sender.name}: {message}")

            # Add sender information to the message
            message['sender'] = sender.name

            # Publish the message to the output topic
            self.message_bus.publish(self.config['output_topic'], message)

    def receive_message(self, message: Dict) -> None:
        """
        Receive messages for a specific agent and forward them accordingly.

        Parameters
        ----------
        message : dict
            Message received from the message bus.

        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Communicator is not running")

            logger.info(f"Received message: {message}")

            # Get the recipient agent
            recipient = next((agent for agent in self.agents if agent.name == message['recipient']), None)

            if recipient is None:
                logger.warning(f"Recipient agent {message['recipient']} not found")
                return

            # Forward the message to the recipient agent
            recipient.receive_message(message)

    def update_agent_status(self, agent: Agent, status: str) -> None:
        """
        Update the status of an agent.

        Parameters
        ----------
        agent : Agent object
            Agent for which the status needs to be updated.
        status : str
            New status of the agent.

        """
        with self.lock:
            if not self.running:
                raise RuntimeError("Communicator is not running")

            logger.info(f"Updating status of agent {agent.name} to {status}")

            # Update the status of the agent
            agent.status = status

    def is_running(self) -> bool:
        """
        Check if the communicator is running.

        Returns
        -------
        bool
            True if the communicator is running, False otherwise.

        """
        with self.lock:
            return self.running

def calculate_velocity_threshold(agent: Agent, message: Dict) -> float:
    """
    Calculate the velocity threshold for an agent based on the message received.

    Parameters
    ----------
    agent : Agent object
        Agent for which the velocity threshold needs to be calculated.
    message : dict
        Message containing the necessary information.

    Returns
    -------
    float
        Calculated velocity threshold.

    """
    # Extract values from the message
    mass = message['mass']
    angle = np.radians(message['angle'])
    coefficient = message['coefficient']

    # Apply the formula from the research paper
    velocity_threshold = (2 * mass * agent.gravity * np.sin(angle)) ** 0.5 / coefficient
    return velocity_threshold

def apply_flow_theory(agent: Agent, message: Dict) -> Dict:
    """
    Apply flow theory to an agent based on the message received.

    Parameters
    ----------
    agent : Agent object
        Agent to which flow theory will be applied.
    message : dict
        Message containing the necessary information.

    Returns
    -------
    dict
        Updated message with flow theory applied.

    """
    # Extract values from the message
    target_position = np.array(message['target_position'])
    current_position = np.array(agent.position)

    # Calculate the flow field
    flow_field = self.calculate_flow_field(agent, target_position)

    # Apply the flow field to the agent
    new_velocity = agent.velocity + flow_field

    # Update the message with the new velocity
    message['velocity'] = new_velocity.tolist()

    return message

def calculate_flow_field(agent: Agent, target_position: np.array) -> np.array:
    """
    Calculate the flow field for an agent based on its current state and the target position.

    Parameters
    ----------
    agent : Agent object
        Agent for which the flow field needs to be calculated.
    target_position : numpy array
        Target position towards which the agent needs to move.

    Returns
    -------
    numpy array
        Calculated flow field.

    """
    # Implement the flow field calculation algorithm from the research paper
    # ...

    return flow_field

# Example usage
if __name__ == '__main__':
    # Initialize agents and message bus (dummy implementation)
    agents = [Agent(name="Agent 1", position=(0, 0), velocity=(1, 0)),
              Agent(name="Agent 2", position=(5, 0), velocity=(0, 1))]
    message_bus = MessageBus()

    # Create a configuration dictionary (replace with actual configuration)
    config = {
        'input_topic': 'agent_input',
        'output_topic': 'agent_output'
    }

    # Create a MultiAgentCommunicator object
    communicator = MultiAgentCommunicator(agents, message_bus, config)

    # Start the multi-agent communication
    communicator.start()

    # Simulate sending messages between agents
    message = {
        'recipient': 'Agent 2',
        'sender': 'Agent 1',
        'target_position': (10, 10)
    }

    communicator.send_message(agents[0], message)

    # Update agent status
    communicator.update_agent_status(agents[0], 'idle')

    # Stop the multi-agent communication
    communicator.stop()