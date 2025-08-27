from flask import Flask, jsonify, request
import importlib.util
import sys, os
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token,
    jwt_required, get_jwt_identity
)
import sys
# Dynamic importer

def import_outside_directory(module_name, file_or_dir_path):
    # If it's a directory, use __init__.py
    if os.path.isdir(file_or_dir_path):
        file_or_dir_path = os.path.join(file_or_dir_path, "__init__.py")

    spec = importlib.util.spec_from_file_location(module_name, file_or_dir_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for {module_name} at {file_or_dir_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

for module in sys.modules:
    print(module)

# Import Helpers + Models
try:
    from BookManAPI.Models import *
    import BookManAPI.Models as Models
    from BookManAPI.Services import Helper as Helper
except Exception as e:
    HELPER_PATH = r"C:\Users\USER\OneDrive\Desktop\BookManAPI\Services\Helper.py"
    MODELS_PATH = r"C:\Users\USER\OneDrive\Desktop\BookManAPI\Models"

    Helper = import_outside_directory("Helper", HELPER_PATH)
    Models = import_outside_directory("Models", MODELS_PATH)
    
Helper.MODEL_STARTED = True
Models.run()
Helper.MODEL_STARTED = True


# Flask setup

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bookman.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = (
    'sjhe9dwde0is_0ia;iuoiouiunueiue9wjd0weuwiduwnid@99s9uscbijvvidfnihehfud^^hieegieovyfifidu9u9ru9rr5%'
)

jwt = JWTManager(app)
crypt = Bcrypt(app)


# Auth routes

@app.route('/auth/register', methods=["POST"])
def register():
    data = request.get_json()
    Username = data.get("Username")
    Password = data.get("Password")

    if not Username or not Password:
        return jsonify({"error": "Username and Password required"}), 400

    if Models.Person.query.filter_by(Username=Username).first():
        return jsonify({"message": "Username already exists"}), 409

    hashed = crypt.generate_password_hash(Password).decode("utf-8")
    new_user = Models.Person(
        PersonId=Helper.key_gen(),
        Username=Username,
        Password=hashed
    )
    Models.add(new_user)
    return jsonify({"message": f"User {Username} created successfully"}), 201


@app.route('/auth/login', methods=["POST"])
def login():
    data = request.get_json()
    Username = data.get("Username")
    Password = data.get("Password")

    user = Models.Person.query.filter_by(Username=Username).first()
    if not user:
        return jsonify({"error": "User not found in database"}), 404

    if not crypt.check_password_hash(user.Password, Password):
        return jsonify({"message": "Bad credentials"}), 401

    token = create_access_token(identity=Username)
    return jsonify(access_token=token)



# Book routes

@app.route('/books', methods=["GET", "POST"])
def books():
    if request.method == "GET":
        books = Models.Book.query.all()
        if not books:
            return jsonify({"error": "No books found"}), 404
        return jsonify([book.to_dict() for book in books])

    elif request.method == "POST":
        data = request.get_json()
        # Require JWT only for POST
        @jwt_required()
        def create_book():
            current_user = get_jwt_identity()
            if not current_user:
                return jsonify({"error": "Unauthorized"}), 401

            new_book = Models.Book(
                BookId=Helper.key_gen(),
                Title=data.get("Title"),
                Description=data.get("Description"),
                Author=current_user,
            )
            Models.add(new_book)
            return jsonify({"message": "Book created successfully"}), 201

        return create_book()


@app.route('/books/<ido>', methods=["GET", "PUT", "DELETE"])
def bookmethod(ido):
    if request.method == "GET":
        # Try multiple lookups: BookId, Title, Author
        book = Models.Book.query.filter_by(BookId=ido).first()
        if not book:
            book = Models.Book.query.filter_by(Title=ido).first()
        if not book:
            book = Models.Book.query.filter_by(Author=ido).first()
        if not book:
            return jsonify({"error": "Book not found"}), 404
        return jsonify(book.to_dict())

    elif request.method == "PUT":
        data = request.get_json()

        @jwt_required()
        def update_book():
            current_user = get_jwt_identity()
            book = Models.Book.query.filter_by(BookId=ido).first()
            if not book:
                return jsonify({"error": "Book not found"}), 404
            if book.Author != current_user:
                return jsonify({"error": "Not Allowed"}), 403

            book.Title = data.get("Title", book.Title)
            book.Description = data.get("Description", book.Description)
            Models.session.commit()
            return jsonify({"message": "Book updated", "book": book.to_dict()})

        return update_book()

    elif request.method == "DELETE":
        @jwt_required()
        def delete_book():
            current_user = get_jwt_identity()
            book = Models.Book.query.filter_by(BookId=ido).first()
            if not book:
                return jsonify({"error": "Book not found"}), 404
            if book.Author != current_user:
                return jsonify({"error": "Not Allowed"}), 403

            Models.session.delete(book)
            Models.session.commit()
            return jsonify({"message": "Book successfully deleted"})

        return delete_book()



if __name__ == "__main__":
    app.run(debug=True)
