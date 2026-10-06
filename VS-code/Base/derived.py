from base import Vehicle
class Car(Vehicle):
    """
    The derived class (Car) inherits from Vehicle.
    It adds specific attributes and overrides a method.
    """
    
    # The __init__ method of the derived class
    def __init__(self, brand, color, model, horsepower):
        # 1. Call the parent constructor to initialize inherited attributes
        super().__init__(brand, color) 
        
        # 2. Initialize new, specialized attributes
        self.model = model
        self.horsepower = horsepower
        
    # Override the inherited method
    def show_info(self):
        # Call the parent's method and add extra car-specific details
        base_info = super().show_info()
        return f"{base_info} It is the {self.model} model with {self.horsepower} HP."

    # New method specific to the Car class
    def drive(self):
        return f"The {self.model} is now driving."
# Create an instance of the derived class
my_car = Car(brand="Toyota", color="Red", model="Camry", horsepower=203)

# Accessing inherited attributes
print(f"Brand: {my_car.brand}") 

# Calling inherited method
print(my_car.start_engine()) 

# Calling the overridden method (it executes the Car version)
print(my_car.show_info()) 

# Calling the new method specific to Car
print(my_car.drive())
