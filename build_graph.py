from neo4j_config import create_neo4j_driver

print("==================================================")
print("   UPGRADING NEO4J KNOWLEDGE GRAPH (PHASE 2)")
print("==================================================\n")


def build_advanced_graph():
    with create_neo4j_driver() as driver:
        with driver.session(database="neo4j") as session:
            # 1. Clear the old graph to avoid data conflicts
            print("[*] Wiping old generic graph data...")
            session.run("MATCH (n) DETACH DELETE n")

            # 2. Inject Interstellar (with release year property)
            print("[*] Injecting Node: Interstellar (2014) & Edge: DIRECTED_BY...")
            session.run("""
            MERGE (m:Movie {name: 'Interstellar', year: 2014})
            MERGE (p:Person {name: 'Christopher Nolan'})
            MERGE (m)-[:DIRECTED_BY]->(p)
            """)

            # 3. Inject Christopher Nolan's birth location
            print("[*] Injecting Node: London & Edge: BORN_IN...")
            session.run("""
            MERGE (p:Person {name: 'Christopher Nolan'})
            MERGE (l:Location {name: 'London'})
            MERGE (p)-[:BORN_IN]->(l)
            """)

            # 4. Inject Inception (with release year property)
            print("[*] Injecting Node: Inception (2010) & Edge: DIRECTED_BY...")
            session.run("""
            MERGE (m:Movie {name: 'Inception', year: 2010})
            MERGE (p:Person {name: 'Christopher Nolan'})
            MERGE (m)-[:DIRECTED_BY]->(p)
            """)

            print("\nSUCCESS! Advanced Knowledge Graph built with strict metadata properties.")

if __name__ == "__main__":
    build_advanced_graph()