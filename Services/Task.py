# services.py
from flask import jsonify
from flask_jwt_extended import create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import Models
from Models import session
import Services.Helper



def register_user(data, crypt):
    Username = data.get("Username")
    Password = data.get("Password")

    if not Username or not Password:
        return jsonify({"error": "Username and Password required"}), 400

    if session.query(Models.Person).filter_by(Username=Username).first():
        return jsonify({"message": "Username already exists"}), 409

    hashed = crypt.generate_password_hash(Password).decode("utf-8")
    new_user = Models.Person(
        PersonId=Helper.key_gen(),
        Username=Username,
        Password=hashed
    )
    session.add(new_user)
    session.commit()
    return jsonify({"message": f"User {Username} created successfully"}), 201


def login_user(data, crypt):
    Username = data.get("Username")
    Password = data.get("Password")

    user = session.query(Models.Person).filter_by(Username=Username).first()
    if not user:
        return jsonify({"error": "User not found in database"}), 404

    if not crypt.check_password_hash(user.Password, Password):
        return jsonify({"message": "Bad credentials"}), 401

    token = create_access_token(identity=Username)
    return jsonify(access_token=token)




def get_books():
    from Models import session
    books = session.query(Models.Book).all()
    if not books:
        return jsonify({"error": "No books found"}), 404
    return jsonify([book.to_dict() for book in books])


def create_book(data, current_user):
    new_book = Models.Book(
        BookId=Helper.key_gen(),
        Title=data.get("Title"),
        Description=data.get("Description"),
        Author=current_user,
    )
    Models.add(new_book)
    return jsonify({"message": "Book created successfully"}), 201


def get_book(ido):
    book = session.query(Models.Book).filter_by(BookId=ido).first()
    if not book:
        book = session.query(Models.Book).filter_by(Title=ido).first()
    if not book:
        book = session.query(Models.Book).filter_by(Author=ido).first()
    if not book:
        return jsonify({"error": "Book not found"}), 404
    return jsonify(book.to_dict())


def update_book(ido, data, current_user):
    book = session.query(Models.Book).filter_by(BookId=ido).first()
    if not book:
        return jsonify({"error": "Book not found"}), 404
    if book.Author != current_user:
        return jsonify({"error": "Not Allowed"}), 403

    book.Title = data.get("Title", book.Title)
    book.Description = data.get("Description", book.Description)
    Models.session.commit()
    return jsonify({"message": "Book updated", "book": book.to_dict()})


def delete_book(ido, current_user):
    book = session.query(Models.Book).filter_by(BookId=ido).first()
    if not book:
        return jsonify({"error": "Book not found"}), 404
    if book.Author != current_user:
        return jsonify({"error": "Not Allowed"}), 403

    Models.session.delete(book)
    Models.session.commit()
    return jsonify({"message": "Book successfully deleted"})
