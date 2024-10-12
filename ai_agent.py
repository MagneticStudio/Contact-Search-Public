# AI Contacts Search Assistant
# Copyright (C) 2023 Alex Furmansky, Magnetic Ventures LLC
#
# This file is part of the AI Contacts Search Assistant.
#
# AI Contacts Search Assistant is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI Contacts Search Assistant is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with AI Contacts Search Assistant. If not, see <https://www.gnu.org/licenses/>.

from typing import TypedDict, List, Dict, Any, Annotated
import os
import re
from openai import OpenAI
import uuid
import sqlite3

from langchain_core.output_parsers import JsonOutputParser

from langchain.pydantic_v1 import BaseModel, Field


import logging
from dotenv import load_dotenv
from langchain_core.tools import tool

from vector_handler import EmbeddingStore
from db_handler import get_database_schema, validate_query, create_or_update_results, fetch_detailed_contact_profile

from langchain_openai import ChatOpenAI
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import tools_condition
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Externalize database name
DATABASE_NAME = os.getenv('DATABASE_NAME', 'contacts.db')

##### DEFINE THE ASSISTANT #####
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Define the primary assistant prompt
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for searching contacts. "
            "You have access to the following database schema: {schema}. "
            "If the user mentions a name, you can use a search tool to find the contact. Always use fuzzy search for names because users may use different spellings. Fuzzy MATCH search can only be used on FTS tables."
            "The general contacts search tool is an SQL search on regular or FTS tables. SQL queries are useful for "
            "searching for contacts by specific attributes like name, email, title, company name, phone number, location, keywords, etc."
            "The vector search tool is a semantic search that compares the natural language query to the long-form text of the contacts using embeddings."
            "Vector search is useful for contextual similarity comparisons across multiple contacts."
            "You should use both the vector search tool and the general contacts search tool together. "

            "When constructing SQL queries, follow these syntax rules for FTS (Full-Text Search):\n"
            "1. Use MATCH for fuzzy searching in FTS tables.\n"
            "2. For OR conditions: Use 'MATCH 'term1 OR term2 OR term3''.\n"
            "3. For AND conditions: Use 'MATCH 'term1 AND term2 AND term3''.\n"
            "4. Combine AND, OR, and phrases: 'MATCH '\"exact phrase\" AND (term1 OR term2)''.\n"
            "5. Use wildcards with asterisk: 'MATCH 'term*''. This is helpful when words have a hyphen or when you want to match a range of words.\n\n"
            "Example 1: To find contacts who attended Harvard:\n"
            "SELECT c.*, e.school FROM contacts c JOIN education_fts e ON c.id = e.contact_id WHERE e.school MATCH 'Harvard';\n"
            "Example 2: To find contacts with the title 'Founder' or 'CEO' in Miami:\n"
            "SELECT c.*, e.title FROM contacts c JOIN experiences_fts e ON c.id = e.contact_id WHERE c.location LIKE '%Miami%' AND e.title MATCH 'Founder* OR CEO';\n"
            "Example 3: Who can I talk to in Los Angeles about raising capital for my startup?\n"
            "SELECT c.*, e.title FROM contacts c JOIN experiences_fts e ON c.id = e.contact_id WHERE c.location LIKE '%Los Angeles%' AND e.title MATCH 'fundraising OR \"venture capital\" OR \"venture investor\"';\n"
            "Example 4: Find contacts who are both founders and involved in AI:\n"
            "SELECT c.*, e.title FROM contacts c JOIN experiences_fts e ON c.id = e.contact_id WHERE e.title MATCH 'founder* AND (AI OR \"artificial intelligence\")';\n"
            "Example 5: Find contacts with variations of the word 'founder':\n"
            "SELECT c.*, e.title FROM contacts c JOIN experiences_fts e ON c.id = e.contact_id WHERE e.title MATCH 'founder* OR cofounder* OR founders* OR \"founding member\"';\n"
            "Use these examples and syntax rules to construct your own queries based on the user's input."

            "Here is a good workflow for your search:\n"
            "1. Begin by ensuring you understand the user's query and the context. If the user's query is vague, you can ask for more information or to broaden the search. You may use tools to help you turn the user's query into a more specific search criteria you will use. "
            "Use the add_new_query tool to add the user's query to the user_queries table. You will need to provide the user's query and the evaluation criteria for the search results (in regular English text). "
            "2. Define the strategy for your search. Consider the tools you have access to. You may need to use several tools in sequence. Explain your reasoning for choosing the tools you will use."
            "Expand keywords in the user's query to related keywords based on your creativity and knowledge. You may also expand the user's query to be more comprehensive, especially when using vector search. "
            "We want to be thorough in our search."
            "Example:"
            "user_query: Who can introduce my sister to in politics?"
            "evaluation_criteria: Contacts working in politics or related fields with relevant experience and connections."
            "3. Decide if the query is a simple query that returns details of a single contact or a complex query that returns multiple contacts. You should confirm your search strategy with the user if it is a complex query. Use regular English to communicate."
            "4. Perform the search. You can use the single_contact_search_tool for simple queries. You can use the many_contacts_search_tool and vector_search_tool for complex queries. The many_contacts_search_tool and vector_search_tool may be used together or separately."
            "When searching for multiple contacts, use the many_contacts_search_tool and vector_search_tool to add a list of contact IDs to the result_sets table. You can then use the evaluate_results_tool to evaluate the results of your search."
            "Use the display_search_results tool to access the list of contact IDs after running evaluate_results_tool to get a nicely formatted version of the results. "
            "You can try multiple search techniques if the first one does not return good results. For example, you can try to broaden the search or use different synonyms of the keywords. "

            "You can divide a query into multiple sequential queries. You can use the new_query tool multiple times to create multiple queries. Here is an example of a complex multi-step query:\n"
            "Sample Query: Which company did Bob and Joe both work at?\n"
            "Strategy: use simple search to find work experiences for Bob and Joe and then compare the results to find the common experiences.\n"
            "Step 1: Use add_queries_tool to add a new query to the user_queries table. Use single_contact_search_tool to find work experiences for Bob\n"
            "Step 2: Use add_queries_tool to add a new query to the user_queries table. Use single_contact_search_tool to find work experiences for Joe\n"
            "Step 3: Compare the results of the two queries to find the common experiences.\n"

            "Sample Query: Which contacts are most similar to Bob?"
            "Strategy: use single_contact_search_tool to get all the information about Bob. Then start a whole new query to find all contacts that are similar to Bob."
            "Step 1: Use add_queries_tool to add a new query to the user_queries table. Use single_contact_search_tool to get all the information about Bob. Write out keywords and summary of Bob to use for the new query."
            "Step 2: Use add_queries_tool to add a new query to the user_queries table. Use both vector_search_tool and many_contacts_search_tool to find similar contacts."
            "Step 3: Use evaluate_results_tool to evaluate the results of your search."
            "Step 4: Use display_search_results to display the results of your search."
            "If your search has zero results, try a different search technique. Be thorough and creative in your search."
            "Only show high and medium relevancy results to the user. Do not show low relevancy results."
            "Remember to phrase your answer as a response based on the user's original query."
            "The format of the results you show to the user is as follows: "
            "contact_name: The name of the contact. "
            "relevancy_score: The relevancy score of the contact (optional). "
            "explanation: Explanation of your answer. "
            "LinkedIn URL: The LinkedIn URL of the contact. "
            
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now(), schema=get_database_schema())

##### DEFINE THE TOOLS #####
# Function to pull the database schema dynamically
@tool
def single_contact_search_tool(query: str, query_id: int) -> Dict[str, Any]:
    """
    Used for simple queries that return specific details of a single contact.

    Args:
        query (str): The SQL query string.
        query_id (int): The ID of the user query.

    Returns:
        Dict[str, Any]: Acknowledgment with the number of results added.
    """
    logging.debug(f"Received query: {query}")
    logging.debug(f"Received query_id: {query_id}")

    # Remove single and double backslashes from the query string
    query = re.sub(r'\\+', '', query)
    logging.debug(f"Sanitized query: {query}")  # Debugging output
    
    if not validate_query(query):
        raise ValueError("Invalid query. Only SELECT statements are allowed.")

    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        # print(f"Raw database query results: {results}")  # Debugging output
        
        # Check if results are empty
        if not results:
            print("No results found for the query.")
        else:
            print(f"Results found: {results}")
        
        
        return {"status": "success.", "results": results}
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

@tool
def many_contacts_search_tool(query: str, query_id: int) -> Dict[str, Any]:
    """
    Execute a search query that returns multiple contacts. Adds the contact IDs to the result_sets table for further evaluation.
    You should limit the number of results to 25.

    Args:
        query (str): The SQL query string.
        query_id (int): The ID of the user query.

    Returns:
        Dict[str, Any]: Acknowledgment with the number of results added.
    """
    logging.debug(f"Received query: {query}")
    logging.debug(f"Received query_id: {query_id}")
    
    # Remove single and double backslashes from the query string
    query = re.sub(r'\\+', '', query)
    logging.debug(f"Sanitized query: {query}")  # Debugging output
    
    # Check if the query already has a LIMIT clause
    if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
        query = f"{query} LIMIT 25"
    logging.debug(f"Final query: {query}")  # Debugging output
    
    if not validate_query(query):
        raise ValueError("Invalid query. Only SELECT statements are allowed.")

    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        # print(f"Raw database query results: {results}")  # Debugging output
        
        # Extract contact IDs directly from the results
        columns = [description[0] for description in cursor.description]
        contact_ids = [dict(zip(columns, row))['id'] for row in results]
        print(f"Extracted contact IDs: {contact_ids}")
        
        num_results = create_or_update_results(query_id, contact_ids)
        print(f"Number of results added to result_sets: {num_results}")
        return {"status": "success.", "results": results}
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

@tool
def vector_search_tool(query: str, query_id: int, model: str = "text-embedding-3-small") -> Dict[str, Any]:
    """
    Perform a vector similarity search and update the result set. Returns maximum of 10 results.

    Args:
        query (str): The search query.
        query_id (int): The ID of the user query.
        model (str): The model to use for generating the query embedding.
        

    Returns:
        Dict[str, Any]: Acknowledgment with the number of results added.
    """
    client = OpenAI()
    embedding_store = EmbeddingStore()
    embedding_store.load_index()

    try:
        response = client.embeddings.create(input=query, model=model)
        query_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for query: {query}\n{e}")
        return {"status": "error", "message": str(e)}

    # Search for similar embeddings
    results = embedding_store.search(query_embedding=query_embedding, k=20)
    # print(f"Raw vector search results: {results}")  # Debugging output

    # Extract contact IDs from the results
    contact_ids = []
    for result in results:
        match = re.match(r'([a-zA-Z0-9]+)_summary_\d+', result[1])
        if match:
            contact_ids.append(match.group(1))
    
    print(f"Extracted contact IDs: {contact_ids}")  # Debugging output

    num_results = create_or_update_results(query_id, contact_ids)
    return {"status": "success", "num_results": num_results}

@tool
def add_new_query(user_query: str, evaluation_criteria: str = None) -> int:
    """
    Prepare a new query entry in the user_queries table.

    Args:
        user_query (str): The original user query.
        evaluation_criteria (str): The evaluation criteria for the search results.

    Returns:
        int: The ID of the newly created query entry.
    """
    conn = sqlite3.connect('contacts.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO user_queries (query_text, query_criteria) VALUES (?, ?)', (user_query, evaluation_criteria))
    query_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return query_id


@tool
def evaluate_results_tool(query_id: int) -> Dict[str, Any]:
    """
    Evaluate the results for a given query ID.

    Args:
        query_id (int): The ID of the user query.

    Returns:
        Dict[str, Any]: Acknowledgment with the number of results evaluated and the relevancy summary.
    """
    db_name = 'contacts.db'
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Fetch the user query and criteria
        cursor.execute('SELECT query_text, query_criteria FROM user_queries WHERE id = ?', (query_id,))
        query_data = cursor.fetchone()
        if not query_data:
            return {"status": "error", "message": "Query ID not found"}
        
        query_text, query_criteria = query_data
        
        # Fetch the result set for the query
        cursor.execute('SELECT id, contact_id FROM result_sets WHERE query_id = ?', (query_id,))
        results = cursor.fetchall()
        
        llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        json_parser = JsonOutputParser()

        system_template = (
            "You are an expert executive assistant. You are helping your client go through the records in their CRM database and look for the most relevant contacts that meet the client's search criteria. Your role is to evaluate each contact and decide whether this contact is relevant for the client's search.\n\n"
            "Here is the client's original question: {query_text}\n"
            "Here is how you will evaluate whether the contact is a good match for the request: {query_criteria}\n\n"
            "You will respond with a JSON object that contains two fields:\n"
            "1. 'explanation': A list of reasons explaining why you believe this contact is a good or bad fit for the user's query. You must include specific facts from the contact's profile in your answer. Be thorough.\n"
            "2. 'relevancy_score': This must be one of: 'Low', 'Medium', or 'High'.\n\n"
            "Here is the full information about this contact: {detailed_contact_profile}\n"
            "{format_instructions}"
            
            ######## NEED TO ADD EXAMPLES HERE #######
        )
        prompt_template = ChatPromptTemplate.from_messages([
            ('system', system_template),
            ('human', '{detailed_contact_profile}')
        ])
        chain = prompt_template | llm | json_parser
        
        num_evaluated = 0
        relevancy_summary = {"Low": 0, "Medium": 0, "High": 0}
        
        for result in results:
            result_id, contact_id = result
            detailed_profile = fetch_detailed_contact_profile(contact_id)
            if not detailed_profile:
                continue
            
            try:
                response = chain.invoke({
                    "query_text": query_text,
                    "query_criteria": query_criteria,
                    "detailed_contact_profile": detailed_profile,
                    "format_instructions": json_parser.get_format_instructions()
                })
                
                # print(f"Response from evaluate_results_tool: {response}")
                
                relevancy_score = response.get('relevancy_score', 'Low')
                explanation = "\n".join(response.get('explanation', []))
                
                cursor.execute('''
                    UPDATE result_sets
                    SET relevancy_score = ?, reasoning = ?
                    WHERE id = ?
                ''', (relevancy_score, explanation, result_id))
                
                relevancy_summary[relevancy_score] += 1
                num_evaluated += 1
            except Exception as e:
                print(f"Error processing result: {e}")
                continue
        
        conn.commit()
        
        return {
            "status": "success",
            "num_evaluated": num_evaluated,
            "relevancy_summary": relevancy_summary
        }
    except sqlite3.Error as e:
        logging.error(f"Database error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

@tool
def display_search_results(query_id: int) -> Dict[str, Any]:
    """
    Display the search results for a given query ID.

    Args:
        query_id (int): The ID of the user query.

    Returns:
        Dict[str, Any]: JSON object containing the contact name, relevancy score, and explanation for each record.
    """
    db_name = 'contacts.db'
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # print(f"Fetching results for query_id: {query_id}")  # Debug print
        
        # Fetch the result set for the query
        cursor.execute('''
            SELECT rs.contact_id, rs.relevancy_score, rs.reasoning, c.full_name
            FROM result_sets rs
            JOIN contacts c ON rs.contact_id = c.id
            WHERE rs.query_id = ?
            ORDER BY CASE rs.relevancy_score
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                WHEN 'Low' THEN 3
                ELSE 4
            END
        ''', (query_id,))
        results = cursor.fetchall()
        
        # print(f"Number of results fetched: {len(results)}")  # Debug print
        
        # Prepare the JSON object
        search_results = [
            {
                "contact_name": result[3],
                "relevancy_score": result[1],
                "explanation": result[2]
            }
            for result in results
        ]
        
        # print(f"Formatted search results: {search_results}")  # Debug print
        
        return {"status": "success", "results": search_results}
    except sqlite3.Error as e:
        print(f"Database error in display_search_results: {e}")  # Debug print
        logging.error(f"Database error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()

# Define the tools
part_1_tools = [
    single_contact_search_tool,
    many_contacts_search_tool,
    vector_search_tool,
    add_new_query, 
    evaluate_results_tool,
    display_search_results
]

# Bind the tools to the assistant
zero_shot_agent_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

# Define helper functions to handle tool errors and pretty print messages
def handle_tool_error(state) -> dict:
    """
    Handle errors in tool execution by adding the error message to the chat history.

    Args:
        state (dict): The current state of the tool execution.

    Returns:
        dict: A dictionary containing the error message.
    """
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    """
    Create a ToolNode with error handling fallbacks.

    Args:
        tools (list): A list of tools to include in the ToolNode.

    Returns:
        dict: A ToolNode with error handling fallbacks.
    """
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def _print_event(event: dict, _printed: set, max_length=1500):
    """
    Pretty print the messages in the graph for debugging.

    Args:
        event (dict): The event containing the messages.
        _printed (set): A set of already printed message IDs to avoid duplicates.
        max_length (int): The maximum length of the message to print.
    """
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

##### DEFINE THE GRAPH #####
# Initialize the graph builder
builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(zero_shot_agent_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()
part_1_graph = builder.compile(checkpointer=memory)



# Main function to run the AI agent directly
def main():
    state = State(messages=[])
    thread_id = str(uuid.uuid4())
    config = {
    "configurable": {

        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}
    _printed = set()

    print("Welcome to the AI Contacts Search Assistant!")
    print("Type your query and press Enter. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break

        logging.debug(f"User input: {user_input}")

        state["messages"].append(("user", user_input))
        events = part_1_graph.stream(state, config, stream_mode="values")
        
        for event in events:
            _print_event(event, _printed, max_length=20000)


if __name__ == '__main__':
    main()