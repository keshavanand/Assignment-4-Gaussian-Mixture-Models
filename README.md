Assignment 3

2: Data Loading Analysis

Time to Load Data: It took approximately 0.3 milliseconds to load the data for December 9th and 10th into the DataFrame.

Number of Tasks: The data loading operation was broken down into 2 tasks. This is because Spark divides the data into partitions and assigns a task for each partition. Each task is processed in parallel by the available executors (cores), allowing Spark to optimize performance.

DAG Execution: The DAG (Directed Acyclic Graph) for the data loading process consisted of several stages. The first stage involved reading the CSV files, inferring the schema, and distributing the data across multiple partitions. Here is a screenshot of the DAG:

3: Basic Investigation

The total number of records are 5649.

The inferred schema and data:

4: Filtered Transactions

The total number of transactions that match the criteria (StockCode starts with '227', description contains "ALARM CLOCK", or unit price > 5) are 173





6: Aggregations and DAG Plan Analysis

Total sum, min and max quantity are 554, -96 and 96 respectively.

This job was divided into two stages with id 3 and 4. The stage with id 3 was divided into two tasks which in total took 0.1 milliseconds. And the stage id 4 only created one task and completed it in 38 ms.

Stage 3: Shuffle Write Size / Records: 141.0 B / 2 

Stage 4: Shuffle Read Size / Records: 141.0 B / 2



The number of partitions for a stage is equal to the number of tasks executed in that stage so for stage 3 partitions were 2 (PROCESS_LOCAL) and for stage 4 partitions were only 1 (NODE_LOCAL).



Job DAG



Stage 3 DAG



Stage 4 DAG







7: Non-UK Transactions

The number of non-UK transactions that meet the criteria from point 4 are 22.





