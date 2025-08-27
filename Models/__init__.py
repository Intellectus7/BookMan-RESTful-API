# import os, sys, importlib.util

# def import_by_filename(module_name, filename):
#     """Search Models/ for a file and import it"""
#     try:
#         base_dir = os.path.dirname(__file__)
#         for root, _, files in os.walk(base_dir):
#             if filename in files:
#                 file_path = os.path.join(root, filename)
#                 spec = importlib.util.spec_from_file_location(module_name, file_path)
#                 module = importlib.util.module_from_spec(spec)
#                 sys.modules[module_name] = module
#                 sys.path.append(module_name)
#                 spec.loader.exec_module(module)
#                 return module
#     except Exception as e:
#         print(e)
#         raise ImportError(f"Could not find {filename} inside {base_dir}")

# # --- Normal imports ---
# try:
#     from .Book import Book
#     from .Person import Person
#     from .Run import (
#         session, run, update, delete, add,
#         find_table, find_table_by_key,
#         bool_table_check, pwd_context, Base
#     )
# # --- Fallback if run outside package ---
# except ImportError:
#     try:
#         from Book import Book
#         from Person import Person
#         from Run import (
#             session, run, update, delete, add,
#             find_table, find_table_by_key,
#             bool_table_check, pwd_context, Base
#         )
#     except Exception as e:
#         BookModule = import_by_filename("Book", "Books.py")
#         PersonModule = import_by_filename("Person", "Person.py")
#         Run = import_by_filename("Run", "Run.py")

#         # Expose everything 
#         Book = getattr(BookModule, "Book", None)
#         Person = getattr(PersonModule, "Person", None)
#         session = getattr(Run, "session", None)
#         run = getattr(Run, "run", None)
#         update = getattr(Run, "update", None)
#         delete = getattr(Run, "delete", None)
#         add = getattr(Run, "add", None)
#         find_table = getattr(Run, "find_table", None)
#         find_table_by_key = getattr(Run, "find_table_by_key", None)
#         bool_table_check = getattr(Run, "bool_table_check", None)
#         pwd_context = getattr(Run, "pwd_context", None)
#         Base = getattr(Run, "Base", None)

# __all__ = [
#     "Book", "Person",
#     "session", "Base", "pwd_context",
#     "run", "update", "delete", "add",
#     "find_table", "find_table_by_key", "bool_table_check"
# ]
from .Book import Book
from .Person import Person
from .Run import (
    session, run, update, delete, add,
    find_table, find_table_by_key,
    bool_table_check, pwd_context, Base
    )
 __all__ = [
    "Book", "Person",
     "session", "Base", "pwd_context",
     "run", "update", "delete", "add",
     "find_table", "find_table_by_key", "bool_table_check"
 ]