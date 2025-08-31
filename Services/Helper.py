import os, sys, importlib.util, random, uuid, secrets, shelve, dbm

# --- Dynamic import helper ---
def import_outside_directory(module_name, file_or_dir_path):
    if os.path.isdir(file_or_dir_path):
        file_or_dir_path = os.path.join(file_or_dir_path, "__init__.py")

    spec = importlib.util.spec_from_file_location(module_name, file_or_dir_path)
    if spec is None:
        raise ImportError(f"Cannot find spec for {module_name} at {file_or_dir_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class localStorage:
    def __init__(self, filename="local_storage"):
        self.filename = filename
        self.passkey = (
            self.filename
            + "4$jduy9y93yhsi03yg_3g9s-+jsehu%=azbajahisols223#ihihdihdsj@ihihsi!hgkshiheljhdroruh^"
        )

    def _open(self, flag="c", writeback=False):
        """Try default dbm, fall back to dumb if needed"""
        try:
            return shelve.open(self.filename, flag=flag, writeback=writeback)
        except dbm.error:
            return shelve.Shelf(dbm.dumb.open(self.filename, flag))

    def setItem(self, key, value):
        with self._open() as db:
            db[key] = value

    def getItem(self, key):
        with self._open() as db:
            val = db.get(key, None)
        # Wrap in reactive object automatically
        if isinstance(val, list):
            return ReactiveList(self, key, val)
        elif val is not None:
            return ReactiveValue(self, key, val)
        return None

    def removeItem(self, key):
        with self._open() as db:
            if key in db:
                del db[key]

    def clear(self):
        with self._open() as db:
            db.clear()

    def generate(self, name, value):
        with self._open(flag="c", writeback=True) as db:
            if name in db:
                return db[name]
            else:
                db[name] = value
                return db[name]

    def getAll(self, passkey):
        if passkey == self.passkey:
            with self._open() as db:
                return dict(db)  # return as normal dict, not Shelf
        else:
            return "Invalid password for the file"


# --- Import Models ---
try:
    current_directory = os.getcwd()
    model_path = os.path.abspath("C:/Users/USER/OneDrive/Desktop/BookManAPI/Models")
    Models = import_outside_directory("Models", model_path)
except ImportError:
    parent_directory = os.path.abspath(os.path.join(os.getcwd(), ".."))
    sys.path.append(parent_directory)
    import Models

from Models import session, Base, pwd_context

MODEL_STARTED = True
storage = localStorage("local_storage")
KEYS = storage.generate("KEYS", [])

def key_gen():
    global MODEL_STARTED
    if not MODEL_STARTED:
        MODEL_STARTED = True
        Models.start()
    # uui = uuid.uuid4()
    # promise = secrets.token_hex(32)
    # hex_uui = uuid.uuid4().hex
    # hex_secrets = secrets.token_urlsafe(32)
    # secret = str(uui) + promise + hex_uui + hex_secrets
    # secret_list = list(secret)
    # index = random.randrange(0, len(secret_list))
    # secret_list[index] = secret_list[index].upper()
    # index2 = index - random.randrange(0, 10)
    # if index2 >= 0:
    #     secret_list[index2] = secret_list[index2].lower()
    # final_secret = "".join(secret_list)
    # if final_secret in KEYS:
    #     return key_gen()
    # KEYS.append(final_secret)
    # return final_secret
    return random.randrange(0, 100000)
def identify(username, password):
    global MODEL_STARTED
    if not MODEL_STARTED:
        MODEL_STARTED = True
        Models.start()
    user = Models.session.query(Models.Person).filter_by(Username=username).first()
    if user and Models.verify_password(password, user.Password):
        return [True, user]
    return False

def update_user(username, **kwargs):
    user = Models.session.query(Models.Person).filter_by(Username=username).first()
    if not user:
        return None
    return Models.update(user, **kwargs)

def delete_record(model_name, key_column, value):
    model = Records.get(model_name)
    if not model:
        raise ValueError("Invalid model name")
    obj = Models.session.query(model).filter(key_column == value).first()
    if obj:
        Models.delete(obj)
        return True
    return False
# --- Utility functions ---

def generate_id():
    return str(uuid.uuid4())

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)