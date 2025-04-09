# Import relevant modules
from py2neo import Graph

# connect to the graph
graph = Graph("neo4j://url", auth=("id", "password!"), name = "dbname")


query2 = """
        with "match(n:Targets) 
        
        return n.samples as samples, id(n) as atom_id, n.name as gene, n.median_exp as median_exp, n.no_of_unique_samples as no_of_samples, n.levels as levels;" as query

        CALL apoc.export.csv.query(query, "allnodes_features_unpaired.csv", {})
        YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
        """

try:
    result2 = graph.run(query2)
    print(result2)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password!"), name = "dbname")
    result2 = graph.run(query2)
    print(result2)


query1 = """
        with "match(n:Targets)-[r:Top_Events]-(m:Targets) 
       
        return r.common_samples as common_samples, r.num_common_samples as num_samples,  n.name as gene1, m.name as gene2, id(n) as atom_0, id(m) as atom_1, r.level1 as level1, r.level2 as level2;" as query
        CALL apoc.export.csv.query(query, "alledges_features_unpaired.csv", {})
        YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
        """

try:
    result1 = graph.run(query1)
    print(result1)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password!"), name = "dbname")
    result1 = graph.run(query1)
    print(result1)


