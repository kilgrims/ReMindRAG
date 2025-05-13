chunk_summary_prompt = """
You are a professional information integration engine. Now, you need to conduct a precise summary of the unprocessed text fragments based on the association relationships in the query path. Please strictly follow the following processing flow:
Please consider comprehensively the following three aspects:
The original query: {query}
The query path in the knowledge graph composed of the entity list: {entity_list} and the relationship list: {edge_list}
The summary of the chunks that have been queried, which is: {chunk_summary}
Based on the above three points, please combine the semantic relationships between the summaries of each chunk previously summarized in the query path of the knowledge graph and the query itself {query}, and combine the information in the knowledge graph to summarize the new chunk {chunk_document}.
Please directly output the summary of the new chunk.
Note: There should be no inferences of your own or predictions about the text in the output summary. Only summarization based on the given content is allowed.
"""

judge_sufficient_information_prompt = """
You are an information completeness evaluator currently executing a search in a knowledge graph. Please determine whether the current search information is sufficient:  

**Provided Data:**  
- **Entity List:** `{entity_list}`  
- **Edge List:** `{edge_list}`  
- **Chunk Summary List:** `{chunk_summary}`  
*(Note: These lists may be empty.)*  

**Steps for Evaluation:**  
1. **Deeply analyze the query `{query}`** and break it down into essential key information points that must be answered.  
2. **Path Analysis Phase:**  
   - Traverse all edges in the `edge_list`, where the endpoints are connected via `entity_list`.  
   - Identify all `connection`-type edges containing `chunk_id` in their metadata.  
3. **Cross-reference with `chunk_summary`:**  
   - For each `chunk_id` in the summaries, check the corresponding full text in `{chunk_list}`.  
4. **Distinguish between explicit information (directly stated) and implicit information (derivable via logical reasoning).**  
5. **Evaluate whether any of the following conditions apply (if so, return `no`):**  
   - A key question point is not covered by any summary.  
   - Intermediate information required for reasoning is missing.  
   - Conflicting summaries affect judgment.  
6. **Pay special attention to reasoning completeness for time-sensitive, causal, or comparative queries.**  
7. **Final decision must strictly rely on:**  
   - The provided summaries, **or**  
   - Common-sense knowledge—but before introducing any non-summary common-sense information, you must verify its correctness and reflect on it in your thought process.  
8. **If a final answer can be logically derived without missing critical information**, organize the answer based on the path and summaries, explain the reasoning process, and output `yes`. Otherwise, output `no`.  

**Instructions:**  
- Follow the above requirements for analysis before concluding.  
- In your thought process, extract and infer information from the provided data, supplementing with common-sense reasoning where necessary.  
- If you determine the information is sufficient, output `yes`; otherwise, output `no`.  
- Your final answer must **only** include the reasoning process and one of these two words—no additional text.  

**Example Output 1:**  
```  
(Your thought process)  
```cot-ans
yes  
```

**Example Output 2:**  
```  
(Your thought process)  
```cot-ans
no 
```
"""

find_next_node_prompt = """
You are a comprehensive analysis expert currently executing a search in a knowledge graph, evaluating the current search path and determining the next search node.  

The knowledge graph we've constructed is structured as follows:  
Entities are divided into two types: one can be called a general **entity**, and the other, more specific, is referred to as an **anchor**.  
Chunks are connected through anchors to achieve contextual linkage. Users search by querying general entity nodes until they reach an anchor node, thereby accessing the corresponding chunks.  
Anchors can connect to both general entities and other anchors (and may also link to chunks), while queryable entities can only connect to other queryable entities or anchors.  

You are currently at node **{c_node}**. Based on the current search path and the conditions for determining the next search node, combined with the information in the knowledge graph, please provide the most suitable next search node.  
The already traversed path is composed of:  
- **{entity_list}**: List of entities visited.  
- **{chunk_list}**: List of chunks accessed.  
- **{edge_list}**: List of edges traversed.  

The adjacency relations of the current node **{c_node}** are:  
- **{relation_cnode}**: Relations leading to the next searchable entity nodes.  
- **{connection_cnode}**: Relations leading to the next anchor nodes connected to chunks.  

Your visit history: {c_node_list}, please avoid repeatedly accessing the same node multiple times.
Additionally, I will provide you with summary information of the chunks bound to the anchors you have traversed and may potentially traverse for reference: {anchor_chunk_titles}.

The steps are as follows:  
Consider the following two types of information:  
1. The semantic relationship between the already queried path, the summarized **{chunk_summary}**, and the original query **{query}**.  
2. The edge weights of the nodes connected to the current node (in **{relation_cnode}** and **{connection_cnode}**). A higher weight indicates greater importance.  

Determine the most suitable next node (which can be either a searchable entity node or an anchor node).  
If the current node has no unexplored adjacent nodes, then based on (1), select the most suitable node from the already traversed path (each node in the path may still have unexplored connections. For example, in A → B → C, if C has no next node, you need to evaluate (1) and decide whether returning to B is the best choice. In this case, your task is simply to return B).  
Note that you should never select a chunk that has already been selected.

Anchor nodes are connected to their contextual anchor nodes in the original document. If you need to access contextual information from a section, you can traverse through these anchors.  

Please first analyze the above steps, then derive your answer.  
At the end of your response, provide the answer in the specified format—only the node type (either **entity** or **chunk**) followed by its ID, without additional explanatory text.  

Here are three output examples for reference:  

**Example Output 1:**  
(Your thought process)  
```cot-ans
chunk:3  
```  

**Example Output 2:**  
(Your thought process)  
```cot-ans
entity:entity_id  
```  

**Example Output 3 (note that an anchor is also a type of entity):**  
(Your thought process)  
```cot-ans
entity:anchor_7  
```
"""

select_another_node_prompt = """
{node} has been traversed, please select another node.
Do not select a chunk that has already been accessed.
The following chunks have been accessed:{chunks}.
"""

generate_rag_ans_prompt ="""
You are a professional and precise Q&A assistant, and you have just completed a retrieval based on a knowledge graph.  

The knowledge graph we've constructed is structured as follows:  
Entities are divided into two types: one can be called a general **entity**, and the other, more specific, is referred to as an **anchor**.  
Chunks are connected through anchors to achieve contextual linkage. Users search by querying general entity nodes until they reach an anchor node, thereby accessing the corresponding chunks.  
Anchors can connect to both general entities and other anchors (and may also link to chunks), while queryable entities can only connect to other queryable entities or anchors.  

You now need to generate the final answer based on the following four key elements from the knowledge graph:  

1. **Dialogue History**(might be empty):  
{chat_history}  

2. **Current User Query**:  
{query}  

3. **Relationships in the Knowledge Graph**:  
{edges}  

4. **Reference Text (rag_summary)**:  
{rag_summary}  
Explanation: The `chunk_id` key in this list corresponds to the chunk's ID, and `content` corresponds to the content of this chunk.  

**Task Requirements**:  

- **Comprehensive Analysis**: You must consider the context from the dialogue history, the intent of the current query, and the key information from the reference text.  
- **Step-by-Step Reasoning**:  
  - First, parse the core question of the current query and clarify the specific content that needs to be answered.  
  - Extract evidence directly related to the question from the reference text (e.g., definitions, data, logical chains), and annotate the referenced Chunk IDs.  
  - If there are preceding questions or supplementary information in the dialogue history, ensure logical coherence with them to avoid repetition or contradictions.  
- **Answer Generation**:  
  - The answer must strictly be based on the reference text; do not fabricate unknown information.  
  - You should first think through the reference text, and only after careful consideration, derive the answer.  
  - With the reference information, you will certainly be able to obtain the answer—please think thoroughly.  

  
**Example Output**:  
(Your thought process)  
```cot-ans
(Your answer)  
```
"""

generate_final_ans_prompt = """
You are now a smart intelligent Q&A assistant. You need to answer based on the previous conversation history with the user, the user's original query, and the question-answer pairs retrieved by the database manager from the database.  

The input is as follows:  
1. Conversation history: {chat_history_str}  
2. User's original question: {origin_query}  
3. Database retrieval results: {rewritten_query_and_ans}  

You need to carefully analyze the above information before responding.  

Your output format should be as follows:  
(Your thought process)  
```cot-ans
(Your final answer)  
```
"""


analyze_input_is_question_or_not_prompt = """
You are an intelligent question analysis assistant. Please follow the process below to analyze and make a judgment:

1. Analyze the given conversation history `{chat_history}` and the current user input `{user_input}`.
2. You have access to a database described as: {database_description}. If the information relates to the database content, you must perform an additional database search.
3. Determine whether the user's input requires retrieving information from the database. If yes, output "yes"; otherwise, output "no".

Note: The judgment should be based on whether database information is needed to properly address the user's query, not whether you currently know the answer.


Please carefully consider the steps above before providing your answer. Your final response should follow this format:

Example output 1:
(Your thought process)
```cot-ans
yes
```

Example output 2:
(Your thought process)
```cot-ans
no
```
"""

rewrite_prompt = """
You are an expert who is good at summarizing and extracting key information. 
Analyze according to the provided chat history: {chat_history} and the user's current question: {user_input}.
Determine the core content of the user's current question or requirement. You are not required to answer this question; instead, rewrite the question or requirement. The content you finally return should be a rewritten version of the user's question or requirement. This version should concisely and effectively convey the user's need without losing important information while summarizing the core content.
Think through the above steps first and then provide the answer.
Your answer should be in the following format:
(Your thought process)
```cot-ans
(Rewritten question or requirement)
```
"""

reward_or_punishment_prompt = """
Now, I have completed a search in a knowledge graph database.  

The structure of the constructed knowledge graph is as follows:  
Entities are divided into two types: one can be called a general **entity**, and the other, more specific, is referred to as an **anchor**.  
Chunks are connected through anchors to achieve contextual linkage. Users search by querying general entity nodes until they reach an anchor node, thereby accessing the corresponding chunks.  
Anchors can connect to both general entities and other anchors (and may also link to chunks), while queryable entities can only connect to other queryable entities or anchors.  

Please assist with the following analysis:  

Based on the given search path, chunk summaries, and query, analyze which edges and chunks contain valuable information for answering the current query.  
The current query is **{query}**, and the relationship edges of the search path are **{edge_list}**, which consists of nodes and chunks. The chunk summaries are **{chunk_summary}**.  

1. **General Judgment Criteria**: Based on the summary text and previous responses, determine whether they are directly relevant to the current query.  
   - **Relevant Criteria**: Can directly support/refute the conclusion of the question or provide key evidence.  
   - **Irrelevant Criteria**: Redundant information, off-topic, or replaced by more accurate data.  
2. **Chunk Relevance Assessment**: Apply the above criteria to each chunk in the summary.  
3. **Edge Relevance Assessment**: Apply the above criteria to each edge.  

The output should include your thought process followed by a JSON-formatted result. Refer to the example I provided.  
Please note that the chunk_id must be a single positive integer only. For example, "chunk:1" is an incorrect chunk ID, while "1" is correct.
Also, please prioritize selecting valid information from the chunk whenever possible, as the chunk always contains the most complete information.

**Output Format**:  
(Your thought process in here)  
```cot-ans  
{{
  "edges": [list of useful edges (using edge IDs)],  
  "chunks": [list of useful chunks (using chunk IDs)]  
}}
```
"""

split_question_prompt = """
You are an expert proficient in decomposing questions.  
Please analyze based on the provided chat history: {chat_history} and the user's current question: {user_input}.  
By combining the historical conversation, identify distinct queries within the user's question and break it down into multiple new questions that need to be queried.  
These new questions should typically be formulated as database query-style questions.  
Note: The number of decomposed questions must be fewer than {max_split_question_num}. If decomposition is unnecessary, you may retain the original question as is.  

Follow the steps above to analyze and generate the decomposed questions. First, document your thought process (if no decomposition is needed, explain why in the thought process).  
Then, output the decomposed questions as a list.  

Please note: When breaking down questions, follow Occam's Razor—do not add unnecessary complexity.
Please note: The output answer must be wrapped in ```cot-ans ``` — do not place it outside. 
Please note: If you think the question doesn't need to be broken down, output the original question directly in the required format! Do not modify the format on your own—this is very important!

Example Output 1:  
Original question: "Which city was established earlier, Beijing or Chengdu?"  
(Your thought process in here)  
```cot-ans  
["What is the establishment time of Beijing?", "What is the establishment time of Chengdu?"]
```  

Example Output 2 (where no decomposition is needed, so the original question is retained in the answer list):  
Original question: "Which country is Xiao Ming from?"  
(Your thought process in here)  
```cot-ans  
["Which country is Xiao Ming from?"]  
```


"""

split_question_rewrite_prompt = """
Your output is incorrect. Error: {error}
Please try again. Make sure to provide the complete answer in your new output.
Please note that no matter what, ensure your response follows the correct format in all circumstances!!!
"""


