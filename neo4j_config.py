import os

from neo4j import GraphDatabase


def create_neo4j_driver():
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        raise RuntimeError("Set the NEO4J_PASSWORD environment variable.")

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    return GraphDatabase.driver(uri, auth=(username, password))
