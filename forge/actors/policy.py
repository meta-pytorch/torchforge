class PolicyInterface(Actor, ABC):
    """Abstract interface for policies."""

    @endpoint
    @abstractmethod
    async def generate(self, request: Observation) -> Action:
        """Generate an action given a state/request."""
        pass

    @endpoint
    @abstractmethod
    async def update_weights(self):
        """Update the policy weights."""
        pass
