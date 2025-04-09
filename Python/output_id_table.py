# Import relevant modules
from py2neo import Graph


# Login to the database
graph = Graph("neo4j://url", auth=("yliu34", "password"), name = "dbname")

query2 = """
        WITH "match(n:nodes_chose) 
        
        return n.name as name, n.median_exp as median_exp, n.no_of_unique_samples as sample_size, n.no_of_samples as sample_size,id(n) as nodeId, n.event_no as event_no" AS query
        
        CALL apoc.export.csv.query(query, "id_gene_map.csv", {})
        
        YIELD file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data
        
        RETURN file, source, format, nodes, relationships, properties, time, rows, batchSize, batches, done, data;
        
        """
try:
    result2 = graph.run(query2)
    print(result2)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("yliu34", "password"), name = "dbname")
    result2 = graph.run(query2)
    print(result2)
