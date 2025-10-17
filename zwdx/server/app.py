from zwdx.server.server import Server

if __name__ == "__main__":
    server = Server().instance()
    server.run()