# Import relevant modules
from py2neo import Graph

# connect to the graph
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
         MATCH (m:Targets)
         REMOVE m:Targets
         """
try:
    result4 = graph.run(query4)
    print(result4)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
    result4 = graph.run(query4)
    print(result4)


# change the directory 
allgene = pd.read_csv('features_unpaired.csv')
params = {}
params['allgenes'] = list(allgene['x'])

query2 = """
        MATCH (n) WHERE n.name IN $allgenes AND (n.levels <> 1)
        SET n:Targets
        """
try:
    result2 = graph.run(query2, params)
    print(result2)
except:
    print('connection failed.............connect again!!!!!!!!!')
    graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
    result2 = graph.run(query2, params)
    print(result2)


query1 = """
        CALL apoc.periodic.iterate(

        '

        MATCH (n:Targets)

        WITH collect(n) AS selectedNodes1

        MATCH (m:Targets)

        UNWIND selectedNodes1 as n

        WITH n, m, size(apoc.coll.intersection(n.samples, m.samples)) AS intersection

        WHERE n.name <> m.name AND NOT (n)-[:Top_Events]->(m) AND intersection >0
        
        RETURN id(n) as NId1, id(m) as NId2

        ',

        '

        MATCH (n:Targets), (m:Targets)

        WHERE id(n) = NId1 AND id(m) = NId2

        CREATE (n)-[r:Top_Events]->(m)

         

        WITH n, m, r, [sample IN n.samples WHERE sample IN m.samples] as matching

        SET r.common_samples = matching,

        r.num_common_samples = size(matching),

        r.level1 = n.levels,

        r.level2 = m.levels

        ',

        {batchSize:10000, parallel:true}) YIELD batches, total

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
