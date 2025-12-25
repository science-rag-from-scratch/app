import os
import psycopg2
import dotenv
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
import chainlit as cl
from typing import Optional

dotenv.load_dotenv()

DB_CONN = (
    f"dbname={os.environ['POSTGRES_DB']} "
    f"user={os.environ['POSTGRES_USER']} "
    f"password={os.environ['POSTGRES_PASSWORD']} "
    f"host={os.environ['POSTGRES_HOST']}"
)
CHAINLIT_CONN = (
    f"postgresql+asyncpg://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)

# -----------NEW-USER-DATA------------
login = "admin"
password = "1234"
display_name = "Admin"
access = "1"
# ------------------------------------

conn = psycopg2.connect(DB_CONN)
cur = conn.cursor()

cur.execute(
    """
            SELECT MAX(identifier)
            FROM users
            """
)

if cur.fetchall()[0][0]:
    identifier = str(int(cur.fetchall()[0][0]) + 1)
else:
    identifier = "0"

cur.close()
conn.close()

metadata = {
    "username": login,
    "password": password,
    "display_name": display_name,
    "access": access,
}


@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo=CHAINLIT_CONN)


@cl.on_chat_start
async def start_chat():
    pass


@cl.password_auth_callback
async def on_login(username: str, password_1: str) -> Optional[cl.User]:
    data_layer = get_data_layer()
    user = cl.User(identifier=identifier, metadata=metadata)
    await data_layer.create_user(user)
    print(
        f"\nПользователь успешно создан.\nlogin: {login}\npassword: {password}\nid: {identifier}\n"
    )
    return cl.User(identifier=identifier, metadata=metadata)
