# Import relevant modules
from py2neo import Graph

# cutoff for common samples shared across edges
params = {}
params['n_pt'] = 4

# Login to the database
graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")


query3 = """
         MATCH (m:File)
         DELETE m
         """
try:
    result3 = graph.run(query3)
    print(result3)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
    result3 = graph.run(query3)
    print(result3)


query4 = """
         MATCH (m:Expression) WHERE (m.include = "Y") AND (m.levels <> 1)
         SET m:nodes_chose
         """
try:
    result4 = graph.run(query4)
    print(result4)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
    result4 = graph.run(query4)
    print(result4)


query1 = """
        CALL apoc.periodic.iterate(

        '

        MATCH (m:nodes_chose) 

        WITH collect(m) AS selectedNodes1

        MATCH (n:nodes_chose)

        UNWIND selectedNodes1 as m

        WITH n, m, size(apoc.coll.intersection(n.samples, m.samples)) AS intersection

        WHERE n.name <> m.name AND NOT (n)-[:Top_Events]->(m) AND intersection > $n_pt
        
        RETURN id(n) as NId1, id(m) as NId2

        ',

        '

        MATCH (n:nodes_chose), (m:nodes_chose)

        WHERE id(n) = NId1 AND id(m) = NId2

        CREATE (n)-[r:Top_Events]->(m)

         

        WITH n, m, r, [sample IN n.samples WHERE sample IN m.samples] as matching

        SET r.common_samples = matching,

        r.num_common_samples = size(matching)


        ',

        {batchSize:50000, parallel:true}) YIELD batches, total

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

