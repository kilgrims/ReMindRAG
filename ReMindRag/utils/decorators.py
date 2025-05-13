import json
import re
from functools import wraps
from .prompts import json_rewrite_prompt, check_keys_rewrite_prompt, unpack_ans_rewrite_prompt

def retry_json_parsing(func):
    """
    A decorator to retry the function if JSON parsing fails.
    Handles JSON strings wrapped in ```json ``` markers.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        retries = 0
        max_retries = 3
        last_result = None
        
        while retries < max_retries:
            try:
                result = func(self, *args, **kwargs)
                last_result = result
                
                # Handle JSON string that might be wrapped in ```json ```
                if isinstance(result, str):
                    stripped_result = result.strip()
                    if stripped_result.startswith('```json') and stripped_result.endswith('```'):
                        # Remove the markers and any leading/trailing whitespace
                        stripped_result = stripped_result[7:-3].strip()
                    try:
                        result_json = json.loads(stripped_result)
                    except json.JSONDecodeError:
                        # If stripping markers caused error, try parsing original
                        result_json = json.loads(result)
                else:
                    result_json = result if isinstance(result, dict) else json.loads(str(result))
                
                return result_json
                
            except (json.JSONDecodeError, TypeError) as e:
                retries += 1
                print(f"JSON parsing failed, retrying... ({retries}/{max_retries})")
                if last_result:
                    print("Last output:", last_result)
        
        raise ValueError(f"Failed to parse response as JSON format. Last output: {last_result}")
    return wrapper



def check_keys(*required_keys):
    """
    A decorator to check for required keys in a JSON response with retry mechanism.
    Uses error_chat_history to maintain retry context separately from main chat history.
    Handles JSON strings wrapped in ```json ``` markers.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            max_retries = 3
            error_chat_history = kwargs.pop('error_chat_history', [])
            
            while retries < max_retries:
                try:
                    kwargs['error_chat_history'] = error_chat_history
                    
                    result = func(*args, **kwargs)
                    if not result:
                        print("Warning: No Entity Found")
                    
                    # Handle JSON string that might be wrapped in ```json ```
                    if isinstance(result, str):
                        stripped_result = result.strip()
                        if stripped_result.startswith('```json') and stripped_result.endswith('```'):
                            # Remove the markers and any leading/trailing whitespace
                            stripped_result = stripped_result[7:-3].strip()
                        try:
                            parsed_result = json.loads(stripped_result)
                        except json.JSONDecodeError:
                            # If stripping markers caused error, try parsing original
                            parsed_result = json.loads(result)
                    else:
                        parsed_result = result if isinstance(result, dict) else json.loads(str(result))
                    
                    missing_keys = [key for key in required_keys if key not in parsed_result]
                    if missing_keys:
                        raise KeyError(f"Missing keys: {', '.join(missing_keys)}")
                    
                    return parsed_result

                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    retries += 1
                    print(f"Validation failed, retrying... ({retries}/{max_retries})")
                    
                    if retries < max_retries and 'result' in locals():
                        error_chat_history.extend([
                            {"role": "assistant", "content": json.dumps(result)},
                            {"role": "user", "content": check_keys_rewrite_prompt.format(error=str(e))}
                        ])
                    
                    if retries == max_retries:
                        error_msg = {
                            "error": str(e),
                            "attempts": retries,
                            "last_output": result if 'result' in locals() else None,
                            "error_history": error_chat_history
                        }
                        raise ValueError(json.dumps(error_msg, indent=2))
        return wrapper
    return decorator

def unpack_cot_ans(func):
    """
    A decorator to extract cot-ans blocks with retry mechanism.
    Uses error_chat_history to maintain retry context separately.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        max_retries = 3
        error_chat_history = kwargs.pop('error_chat_history', [])
        
        while retries < max_retries:
            try:
                kwargs['error_chat_history'] = error_chat_history
                
                result = func(*args, **kwargs)

                if not isinstance(result, str):
                    raise TypeError("Response must be a string")

                matches = re.findall(r'```cot-ans(.*?)```', result, re.DOTALL)
                if not matches:
                    raise ValueError("No triple backtick block found")
                
                return matches[-1].strip()

            except (TypeError, ValueError) as e:
                retries += 1
                print(f"Extraction failed, retrying... ({retries}/{max_retries})")
                
                if retries < max_retries and 'result' in locals():
                    error_chat_history.extend([
                        {"role": "assistant", "content": result},
                        {"role": "user", "content": unpack_ans_rewrite_prompt.format(error=str(e))}
                    ])
                
                if retries == max_retries:
                    error_msg = {
                        "error": str(e),
                        "attempts": retries,
                        "last_output": result if 'result' in locals() else None,
                        "error_history": error_chat_history
                    }
                    raise ValueError(json.dumps(error_msg, indent=2))
        return ""
    return wrapper