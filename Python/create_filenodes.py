# Import relevant modules
from py2neo import Graph


# Login to the database
graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")

query1 = """
        CALL apoc.periodic.iterate(
        '
        LOAD CSV WITH HEADERS FROM ("file:/url_train.csv") AS row
        RETURN row
        ','
        WITH row.URL AS fileUrl
        MERGE (file:File {url: fileUrl});
        ',
        {batchSize:10000,parallel:true}) YIELD batches, total
        RETURN batches, total;
        """

try:
    result1 = graph.run(query1)
    print(result1)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
    result1 = graph.run(query1)
    print(result1)

