json_rewrite_prompt = """
Your output is incorrect. The error is: {error}. 
Please try again. When you resubmit, please provide the complete answer.
Reminder: Your response must be in the exact JSON format shown earlier. Do not include anything outside the JSON structure.
"""


check_keys_rewrite_prompt = """
Your output is incorrect. The error is: {error}. 
Please try again. When you resubmit, please provide the complete answer.
Reminder: Your response must be in the exact JSON format shown earlier. Do not include anything outside the JSON structure.
Please note that no matter what, ensure your response follows the correct format in all circumstances!!!!
"""

unpack_ans_rewrite_prompt = """
Your output is incorrect. The error is: {error}.  
Please check if your final output follows the required format.  

Reference format:  
```cot-ans  
(Your final answer)  
```  

Please try again. When retrying, ensure your output uses the specified format—wrap the final answer with ```cot-ans and ```.
When you resubmit, please provide the complete answer. 
"""