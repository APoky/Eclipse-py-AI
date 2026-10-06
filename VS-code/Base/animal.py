from abc import ABC, abstractmethod

class Animal(ABC):
    """
    This is the virtual class (Abstract Base Class).
    It cannot be instantiated directly.
    """
    
    def __init__(self, name):
        self.name = name
    @abstractmethod
    def make_sound(self):
        """
        A virtual method that MUST be implemented by all subclasses.
        """
        pass
    
    def walk(self):
        """
        A concrete method that subclasses can optionally override.
        """
        return f"{self.name} is walking."

# ❌ ERROR EXAMPLE: Trying to instantiate the ABC directly will raise an error.
# abstract_animal = Animal("Generic") # Raises TypeError
