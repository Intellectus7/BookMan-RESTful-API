from flask import Flask, request
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, jwt_required, get_jwt_identity
import Models
from Services import Helper as Helper
from Services import Task as services  

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///bookman.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = 'super-secret-key'

jwt = JWTManager(app)
crypt = Bcrypt(app)



@app.route('/auth/register', methods=["POST"])
def register():
    return services.register_user(request.get_json(), crypt)


@app.route('/auth/login', methods=["POST"])
def login():
    return services.login_user(request.get_json(), crypt)


@app.route('/books', methods=["GET", "POST"])
def books():
    if request.method == "GET":
        return services.get_books()

    elif request.method == "POST":
        @jwt_required()
        def create():
            return services.create_book(request.get_json(), get_jwt_identity())
        return create()


@app.route('/books/<ido>', methods=["GET", "PUT", "DELETE"])
def bookmethod(ido):
    if request.method == "GET":
        return services.get_book(ido)

    elif request.method == "PUT":
        @jwt_required()
        def update():
            return services.update_book(ido, request.get_json(), get_jwt_identity())
        return update()

    elif request.method == "DELETE":
        @jwt_required()
        def delete():
            return services.delete_book(ido, get_jwt_identity())
        return delete()


if __name__ == "__main__":
    app.run(debug=True)
