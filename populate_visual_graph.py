import random
from datasets import load_dataset
from neo4j_config import create_neo4j_driver

dataset = load_dataset("framolfese/2WikiMultihopQA", split="validation")
random.seed(42)
sampled_indices = random.sample(range(len(dataset)), 15) # Loads 15 real Wikipedia questions together

with create_neo4j_driver() as driver:
    with driver.session(database="neo4j") as session:
        session.run("MATCH (n) DETACH DELETE n") # Clear once at the start
        for idx in sampled_indices:
            for triple in dataset[idx]['evidences']:
                subj, rel, obj = triple
                subj = str(subj).replace("'", "\\'")
                rel = str(rel).replace(" ", "_").replace("-", "_").upper()
                obj = str(obj).replace("'", "\\'")
                cypher = f"MERGE (s:Entity {{name: '{subj}'}}) MERGE (o:Entity {{name: '{obj}'}}) MERGE (s)-[:{rel}]->(o)"
                try:
                    session.run(cypher)
                except Exception:
                    continue

print("Done! Run 'MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150' in Neo4j Desktop.")