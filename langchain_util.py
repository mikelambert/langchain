
import re

from typing import Optional

SearchTool = str # a description

def add_system_prompt():
    '''
    This function is used to add the systems prompt to the beginning of the prompt. It is currently not used.
    '''
    return "You're going to talk with a human. Please be helpful, honest, and harmless. Please make sure to think carefully, step-by-step when answering the human."

def add_search_tool_description(tools: list[SearchTool]):
    '''
    This function is used to add the tool description to the beginning of the prompt. It is currently not used.
    '''
    tool_prompt = "<tools>"
    for tool in tools:
            tool_prompt += f"""
<tool_description>
Search Engine Tool
* {tool}
* At any time, you can make a call to the search engine using the following syntax: <search_query>Put your search query here!</search_query>.
* You'll then get results back in <search_result> tags.
* Feel free to call the search engine multiple times with different queries in order to find the information you need.
* Sometimes the search engine will return empty search results, or the search results may not contain the information you need. In such cases, you should try again with a different query.
</tool_description>
"""

    tool_prompt += "</tools>"

    return tool_prompt

def get_search_starting_prompt(raw_prompt: str, search_tool: SearchTool) -> str:
    tools: list[str] = [search_tool]
    return f"""{add_system_prompt()}
{add_search_tool_description(tools)}""" + raw_prompt

# Handling search queries and results

def extract_search_query(text: str) -> Optional[str]:
    match = re.search(r"<search_query>(.*?)</search_query>", text)
    return match.group(1) if match else None
