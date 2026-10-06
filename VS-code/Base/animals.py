from animal import Animal

class Dog(Animal):
    """
    Concrete class that successfully implements the virtual method.
    """
    def make_sound(self):
        # Implementation for the required virtual method
        return "Woof! Woof!"

class Cat(Animal):
    """
    Concrete class that implements the virtual method and overrides a concrete one.
    """
    def make_sound(self):
        # Implementation for the required virtual method
        return "Meow!"
        
    def walk(self):
        # Optional override of the concrete method from the parent
        return f"{self.name} is gracefully stalking."

# 🟢 SUCCESS: We can create instances of the concrete subclasses
fido = Dog("Fido")
whiskers = Cat("Whiskers")

print(f"{fido.name}: {fido.make_sound()}")
print(f"{whiskers.name}: {whiskers.walk()}")

# ❌ ERROR EXAMPLE: A class that forgets to implement 'make_sound' would fail here:
# class Bird(Animal):
#     pass
# bird = Bird("Tweety") # Raises TypeError: Can't instantiate abstract class Bird with abstract method make_sound
