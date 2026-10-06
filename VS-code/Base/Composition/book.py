from Composition.author import Author
class Book:
    """
    Represents a book which is composed of an Author instance.
    (Composition: Book has an Author)
    """
    
    def __init__(self, title, isbn, author_instance):
        self.title = title
        self.isbn = isbn
        # Composition happens here:
        # The 'author' attribute holds an instance of the Author class.
        self.author = author_instance 
        
    def describe_book(self):
        author_name = self.author.get_full_name()
        return (f"Title: {self.title}\n"
                f"Author: {author_name}\n"
                f"ISBN: {self.isbn}")