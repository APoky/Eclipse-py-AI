from Composition.book import Book
from Composition.author import Author

# 1. Create the component instance (The Author)
author_obj = Author("Toni", "Morrison")

# 2. Create the container instance (The Book)
# We pass the author object into the Book's constructor.
book_obj = Book(
    title="Beloved",
    isbn="978-0307275276",
    author_instance=author_obj
)

# 3. Accessing the component's data through the container
print(f"Book title: {book_obj.title}")

# We access the Author's methods and attributes via the 'author' attribute
print(f"Author's Full Name (accessed via Book): {book_obj.author.get_full_name()}")

print("-" * 20)
print(book_obj.describe_book())