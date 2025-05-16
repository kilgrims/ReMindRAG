entity_extract_prompt = """
Extract entities and entity relationships from the following text, then output the complete merged statements of entities and entity relationships.  
An entity should be a simple and easy-to-understand word or phrase.  
An entity should be a meaningful word or phrase. For example, in the sentence "David is seventeen years old," "David" is a meaningful entity, but "seventeen years old" is not.  
An entity is a persistent concept with broad significance, not a temporary one. For example, "twenty years old," "one hundred dollars," or "building collapse" cannot be entities.  
At the same time, an entity typically refers to a specific thing or concept with a clear identity or definition. For example, in the sentence "The distance between New York and Boston is not far," "New York" and "Boston" are entities, but "distance" is not.  
When extracting entities, this process should be precise and deliberate, not arbitrary or careless.  
Entity types include organizations, people, geographical locations, events, objects, professional concepts, etc.  
An entity relationship is simply a predicate statement that describes the subject, object, and their relationship.  
Please note that the entities you extract must not include conjunctions like 'or' or 'and'—they should be precise and standalone.

The output format is as follows:  
["entity_1", "entity_2", ..., "entity_n"]

Example 1:  
Given text: This travel guide is very detailed, including introductions to popular attractions, recommendations for local delicacies, and practical transportation guides.  
Output:  
["travel guide", "attractions introduction", "food recommendations", "transportation guide"]

Example 2:  
Given text: In this world, police are supposed to catch thieves.  
Output:  
["police", "thieves"]

Please note: Your final output must strictly follow the required JSON format and should not include any additional content.
If there's no entity found, please output "[]" only.
"""

relation_extract_prompt = """
For the chunk of text I'm about to input, it contains the following named entities: {entity_list}.
Please extract the relationships between these named entities. Each relationship should be a predicate phrase describing the connection between the subject and the object.
For example, in "Tom" "raises" "dog", "raises" is the relationship. After extracting a relationship, combine it with the subject and object to form a complete sentence.

Your final output should be a JSON-formatted list where each sub-list contains three elements:
[The subject of the relationship, The complete relationship sentence, The object of the relationship]

I'll provide some examples next for your reference when generating the output.

Example 1:  
Given text: This travel guide is very detailed, including introductions to popular attractions, recommendations for local delicacies, and practical transportation guides.  
Output:  
[
  ["travel guide", "travel guide includes attractions introduction.", "attractions introduction"],
  ["travel guide", "travel guide includes food recommendations.", "food recommendations"],
  ["travel guide", "travel guide includes transportation guide.", "transportation guide"]
]

Example 2:  
Given text: In this world, police are supposed to catch thieves.  
Output:  
[
  ["police", "police are supposed to catch thieves.", "thieves"]
]

Please note: Your final output must strictly follow the required JSON format and should not include any additional content.
"""

chunk_title_get_prompt = """
You are now a powerful text summarization assistant.  
I will give you a paragraph, and you need to summarize it. The summarization requirements are as follows:  

1. Your summary should allow users to understand what the paragraph is about when they read it.  
2. If there is a part of the paragraph that you cannot summarize well, you may use a simple example in the summary for demonstration.  
3. This summary should function like a table of contents, enabling users to quickly grasp the content of the paragraph and decide whether to read the full text if interested.  
4. Please output the summarized result directly.  
"""


relation_num_error_rewrite_prompt = """
There's an error in your output: relation {relation} has incorrect formatting.
All relations must use this exact format:
['entity_a', 'complete relational sentence (formed as 'subject' + 'relationship statement' + 'object')', 'entity_b']

Your resubmission must include the complete response with both entity_list and relation_triplet.
"""