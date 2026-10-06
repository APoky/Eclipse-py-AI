class Vehicle:
    """The base class defining general vehicle attributes and methods."""
    
    def __init__(self, brand, color):
        self.brand = brand
        self.color = color
        
    def start_engine(self):
        return f"The {self.color} {self.brand} engine starts."
        
    def show_info(self):
        return f"This is a {self.color} {self.brand}."