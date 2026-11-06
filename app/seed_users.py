from app.database import Base, engine, SessionLocal
from app.auth import AuthUser, hash_password

# Import AuthUser before create_all so its table is registered
Base.metadata.create_all(engine)

USERS = [
    ("admin", "admin-pass", "admin"),
    ("analyst", "analyst-pass", "analyst"),
    ("viewer", "viewer-pass", "viewer"),
]

db = SessionLocal()
for username, password, role in USERS:
    if db.query(AuthUser).filter_by(username=username).first() is None:
        db.add(AuthUser(username=username, password_hash=hash_password(password), role=role))
db.commit()
db.close()
print("Seeded users: admin, analyst, viewer")
