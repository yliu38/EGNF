# Import relevant modules
from py2neo import Graph


# Login to the database
graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")


for x in range(0,10000):
    # First drop the graph if already exists
    query_drop = """RETURN gds.graph.exists('random_sampling')"""
    try:
        result_drop = graph.run(query_drop)
        parsed_drop = result_drop.evaluate()
    except:
        print('connection failed.............connect again!!!!!!!!!')
        graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
        result_drop = graph.run(query_drop)
        parsed_drop = result_drop.evaluate()

    if (parsed_drop == 1):
        try:
            result = graph.run("CALL gds.graph.drop('random_sampling') YIELD graphName;")
            print("\nDropping query status for a graph named 'random_sampling'\n")
        except:
            print('connection failed.............connect again!!!!!!!!!')
            graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
            result = graph.run("CALL gds.graph.drop('random_sampling') YIELD graphName;")
            print("\nDropping query status for a graph named 'random_sampling'\n")


    # projection
    query1 = """
            // Cypher query
            MATCH (n:nodes_chose:class1)
            WITH n.name AS geneName, rand() AS randomOrder, n
            ORDER BY geneName, randomOrder
            WITH geneName, COLLECT(n)[0] AS randomNode
            unwind(randomNode) as nme
            match (m:nodes_chose:class1) WHERE id(m)=id(nme)
            set m:tmp_class1
            with m
            // Make a Cypher projection
            MATCH (source:tmp_class1)-[r:Top_Events]->(target:tmp_class1)
            WITH gds.graph.project('random_sampling', source, target,{
            sourceNodeProperties: source {source: id(source) },
            targetNodeProperties: target {target: id(target) }}) AS g
            RETURN g.graphName AS graph, g.nodeCount AS nodes
            """

    print("\nMade a subgraph and a Cypher projection named 'random_sampling'\n")
    try:	
    	result1 = graph.run(query1)
    	print(result1)
    except:
        print('connection failed.............connect again!!!!!!!!!')
        graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
        result1 = graph.run(query1)
        print(result1)
        
    
    # degree distribution
    query3 = """
            //Running Degree centrality
            CALL apoc.export.csv.query("CALL gds.degree.stream('random_sampling', {orientation: 'UNDIRECTED'})
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).name AS name, score, nodeId
            ORDER BY score DESC, name DESC", "degree_cen_""" + str(x) + """.csv", {})
            """
    print("\nRunning degree centrality!\n")
    try:
        result3 = graph.run(query3)
        print(result3)
    except:
        print('connection failed.............connect again!!!!!!!!!')
        graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
        result3 = graph.run(query3)
        print(result3)
        
    
    # modularityOptimization
    query8 = """
            //Modularity Optimization
            CALL apoc.export.csv.query("CALL gds.beta.modularityOptimization.stream('random_sampling')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).name AS name, communityId, nodeId
            ORDER BY communityId, name", "Modularity_Optimization_""" + str(x) +""".csv", {})
            """
    print("\nRunning Modularity Optimization algorithm!\n")
    try:
        result8 = graph.run(query8)
        print(result8)
    except:
        print('connection failed.............connect again!!!!!!!!!')
        graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
        result8 = graph.run(query8)
        print(result8)

   
    # delete label
    query4 = """
            //delete tmp_class1
            match (n:tmp_class1)
            remove n:tmp_class1
            """
    print("\nRunning degree centrality!\n")
    try:
        result4 = graph.run(query4)
        print(result4)
    except:
        print('connection failed.............connect again!!!!!!!!!')
        graph = Graph("neo4j://url", auth=("id", "password"), name = "dbname")
        result4 = graph.run(query4)
        print(result4)
   
