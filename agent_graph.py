import os
import json
from typing import List, Dict, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, END, START
from langchain_google_genai import ChatGoogleGenerativeAI
from google.cloud import bigquery
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

# --- הגדרות סביבה (נדרש עבור LLM) ---
# נניח שה-API Key נמצא ב-GOOGLE_API_KEY בסביבה/ב-.env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# ==========================
# State Definition
# ==========================
class AgentState(MessagesState):
    candidate_tables: List[str] = Field(
        default_factory=list,
        description="A list of table names the agent has identified as potentially relevant."
    )
    tables_to_check: List[str] = Field(
        default_factory=list,
        description="The remaining list of tables to investigate in detail."
    )
    exploratory_results: str = Field(
        default="",
        description="Results of exploratory queries as a string."
    )
    query_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="History of queries and their results."
    )
    
    # Prevent repeated exploration across the same table
    exploration_done: bool = Field(
        default=False,
        description="Whether exploration phase has already been performed for the current table."
    )
    exploration_attempts: int = Field(
        default=0,
        description="How many times exploration was attempted for the current table."
    )
    # For routing after run_query
    last_query_type: Optional[str] = Field(
        default=None,
        description="Type of the last query executed: 'exploratory' or 'final'."
    )
    # Multi-schema management
    schemas_by_table: Dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping of table_name -> detailed schema content."
    )
    current_table: Optional[str] = Field(
        default=None,
        description="The table currently being examined."
    )

# ==========================
# Tools
# ==========================

@tool
def run_query_tool(query: str) -> str:
    """Execute a SQL query on the BigQuery public dataset and return the results."""
    try:
        # NOTE: Client initialization here means it relies on ADC or environment setup.
        client = bigquery.Client() 
        job = client.query(query)
        df = job.result().to_dataframe()
        if df.empty:
            return "[]"
        return df.to_json(orient="records", force_ascii=False)
    except Exception as e:
        return f"Query failed: {e}"


@tool
def json_overview_tool() -> str:
    """Loads the table overview text for table selection."""
    try:
        with open("overview.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "overview.txt not found."


# ==========================
# Helper Functions
# ==========================
def ensure_public_dataset(query: str) -> str:
    """
    Ensures all table references use the correct fully-qualified BigQuery public dataset format.
    Correct format: `bigquery-public-data.thelook_ecommerce`.table_name
    """
    PROJECT_DATASET_PREFIX = "`bigquery-public-data.thelook_ecommerce`."
    DATASET_NAME = "thelook_ecommerce."
    
    # Normalization steps... (Keeping your robust normalization logic)
    query = query.replace(" ``bigquery-public-data", " `bigquery-public-data")
    query = query.replace("``bigquery-public-data", "`bigquery-public-data")
    query = query.replace("`bigquery-public-data`.", "`bigquery-public-data.")
    query = query.replace("bigquery-public-data.bigquery-public-data.", "bigquery-public-data.")
    query = query.replace(f".{DATASET_NAME}`", f".{DATASET_NAME}")
    
    if DATASET_NAME in query and PROJECT_DATASET_PREFIX not in query:
        query = query.replace(
            DATASET_NAME,
            PROJECT_DATASET_PREFIX
        )
        
    if query.endswith("`"):
        if query.count("`") % 2 != 0:
            query = query.strip()[:-1]

    return query


def is_duplicate_query(query_history: List[Dict], q_type: str, q: str) -> bool:
    for entry in query_history:
        if entry.get("type") == q_type and entry.get("query") == q:
            return True
    return False

def keep_first_tool_call(ai_msg: AIMessage) -> AIMessage:
    # Ensure only one tool call is kept in the message (if any)
    tcs = getattr(ai_msg, "tool_calls", None) or []
    if len(tcs) <= 1:
        return ai_msg
    first = tcs[0]
    return AIMessage(content=ai_msg.content, tool_calls=[first])

def schema_file_exists(table_name: str) -> bool:
    # Since we use overview.txt, this function is mostly a placeholder but kept for logic flow
    return True 

def is_query_result_empty_or_failed(result_content: str) -> bool:
    try:
        if isinstance(result_content, str):
            if result_content.strip() == "[]":
                return True
            if "Query failed" in result_content:
                return True
            parsed = json.loads(result_content)
            if isinstance(parsed, dict) and isinstance(parsed.get("data"), list) and len(parsed["data"]) == 0:
                return True
            return False
        else:
            if isinstance(result_content, dict) and isinstance(result_content.get("data"), list):
                return len(result_content["data"]) == 0
            return False
    except Exception:
        return False

def extract_table_names(overview_obj) -> List[str]:
    # Robust extraction logic (kept as is)
    if isinstance(overview_obj, dict):
        candidates = (
            overview_obj.get("tables")
            or overview_obj.get("table_names")
            or overview_obj.get("data")
            or []
        )
    elif isinstance(overview_obj, list):
        candidates = overview_obj
    else:
        candidates = []

    names: List[str] = []
    for item in candidates:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            n = (
                item.get("table_name")
                or item.get("name")
                or item.get("table")
                or item.get("tableName")
            )
            if n:
                names.append(n)
    
    seen = set()
    uniq = []
    for n in names:
        if n not in seen:
            uniq.append(n)
            seen.add(n)
    return uniq

# ==========================
# Graph Nodes
# ==========================

class CandidateTables(BaseModel):
    """The model's selection of the most relevant tables for the user's query."""
    table_names: List[str] = Field(
        description="A list of 3-5 table names that are most relevant to answering the user's question.",
    )


def select_candidate_tables(state: AgentState):
    structured_llm = llm.with_structured_output(CandidateTables)
    system_prompt = """You are an expert data analyst. Your job is to help a user query a database.
You are given a user's question and an overview of all available tables.
Based on this, select the 3-5 most relevant tables that are likely to contain the answer.
If you can't find any relevant tables, respond with an empty list.

Here is the table overview: {table_overview}"""
    overview = json_overview_tool.invoke({})
    prompt = system_prompt.format(table_overview=overview)
    user_question = state["messages"][0].content
    response = structured_llm.invoke([("system", prompt), ("human", user_question)])

    selected = response.table_names or []
    # Simplified filtering since we rely on overview.txt being complete
    filtered = selected 

    if not filtered:
        # If nothing is selected, we must stop the table loop immediately
        return {"candidate_tables": [], "tables_to_check": [], "need_detailed_search": True}
    else:
        return {"candidate_tables": filtered, "tables_to_check": filtered, "need_detailed_search": False}


def loop_entry_point(state: AgentState):
    """A simple node that can also act as a reset point per table."""
    # Reset per-table exploration flags upon entering the loop for the next table
    return {
        "exploration_done": False,
        "exploratory_results": "",
        "exploration_attempts": 0,
        "last_query_type": None
    }


def get_next_schema_to_check(state: AgentState):
    """Pulls the next table from tables_to_check and sets context."""
    tables_list = list(state.get("tables_to_check", []))
    if not tables_list:
        return {"messages": []}

    current_table = tables_list[0]
    remaining_tables = tables_list[1:]

    return {
        "current_table": current_table,
        "tables_to_check": remaining_tables,
        # Injecting the full overview as schema context (as we skip per-table loading)
        "schemas_by_table": {"all_schemas": json_overview_tool.invoke({})} 
    }


def store_schema(state: AgentState):
    """Store the retrieved schema content (from the ToolMessage) into the state."""
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        schema_content = last_message.content
        current_table = state.get("current_table")
        schemas = dict(state.get("schemas_by_table", {}))
        if current_table:
            schemas[current_table] = schema_content
        return {"schemas_by_table": schemas}
    return {}


def generate_query(state: AgentState):
    try:
        system_prompt = """You are an AI assistant designed to generate SQL queries against a real BigQuery public dataset.
Follow these strict rules:

- Always fully qualify table names with project, dataset, and table.
- When referencing the BigQuery public dataset, you **MUST** use the following correct format:
  **`bigquery-public-data.thelook_ecommerce`.table_name**
- **CRITICAL FORMAT RULE:** The backtick (` ) must only enclose the **project and dataset**, and not the table name.
  Correct: `bigquery-public-data.thelook_ecommerce`.users
  Incorrect (will fail): `bigquery-public-data.thelook_ecommerce.`users
  Incorrect (will fail): ``bigquery-public-data.thelook_ecommerce.`users

- If exploratory results are already provided (or exploration_done=True), DO NOT run exploratory queries again.
- When generating the final query, produce exactly one tool call to run_query_tool.
- Do not return more than one tool call in any response.
- Use exact values found in the exploratory results (no guessing).
- Business defaults for "current active customers":
  - Prefer filtering to active subscription status if relevant (e.g., SUB_ACT_IND = 'Y' or a similar status column).
  - For snapshot/history tables with LOGICAL_DATE or similar, default to the latest snapshot:
    LOGICAL_DATE = (SELECT MAX(LOGICAL_DATE) FROM <TABLE_NAME>).
  - When counting customers, use COUNT(DISTINCT CUSTOMER_ID) where applicable.
- If the user explicitly asks for a different time range or status, follow the user; otherwise use the defaults above.
"""

        llm_with_tools = llm.bind_tools([run_query_tool])
        history = state["messages"]
        
        # Context setup
        schemas_by_table = state.get("schemas_by_table", {})
        current_table = state.get("current_table")
        schema_hint = json.dumps(schemas_by_table, ensure_ascii=False)
        dataset_prefix = "thelook_ecommerce."
        current_table_full = f"{dataset_prefix}{current_table}" if current_table else "N/A"

        query_history = list(state.get("query_history", []))
        exploration_done = bool(state.get("exploration_done", False))
        exploratory_results = state.get("exploratory_results", "")

        should_explore = (not exploratory_results) and (not exploration_done) and current_table

        if should_explore:
            # --- Exploratory Path ---
            human_prompt_exploratory = (
                f"Schemas (JSON by table): {schema_hint}\n"
                f"Current table in focus: {current_table_full}\n\n"
                "Generate and run a small exploratory SQL query (single query) "
                "to understand the data better for the current table. "
                "Only one tool call is allowed."
            )
            
            response = llm_with_tools.invoke(
                [("system", system_prompt)] + history + [("human", human_prompt_exploratory)]
            )
            response = keep_first_tool_call(response)

            for tc in getattr(response, "tool_calls", []) or []:
                if tc["name"] == "run_query_tool":
                    args = tc["args"]
                    q = args["query"] if isinstance(args, dict) else json.loads(args)["query"]
                    q = ensure_public_dataset(q)
                    if not is_duplicate_query(query_history, "exploratory", q):
                        query_history.append({"type": "exploratory", "query": q, "result": {}})
                    # Update arguments in tool call
                    if isinstance(args, dict):
                        args["query"] = q
                    else:
                        tc["args"] = json.dumps({"query": q})

            return {
                "messages": [response],
                "query_history": query_history,
                "exploration_done": True,
            }

        # --- Final Query Path ---
        system_prompt_final = (
            system_prompt + 
            "\n\nIMPORTANT: Your next message **must include exactly one tool call** to `run_query_tool` "
            "with a JSON argument {\"query\": \"...\"}. Do not output SQL in markdown or text form; "
            "you must call the tool directly."
        )
        
        human_prompt_final = (
            f"Original question: {history[0].content}\n\n"
            f"Schemas (JSON by table): {schema_hint}\n"
            f"Current table in focus: {current_table_full}\n\n"
            f"Exploratory results (if any): {exploratory_results}\n\n"
            "Now, generate the final SQL query to answer the original question using those results."
        )
        
        final_response = llm_with_tools.invoke(
            [("system", system_prompt_final)] + history + [("human", human_prompt_final)]
        )
        final_response = keep_first_tool_call(final_response)

        for tc in getattr(final_response, "tool_calls", []) or []:
            if tc["name"] == "run_query_tool":
                args = tc["args"]
                q = args["query"] if isinstance(args, dict) else json.loads(args)["query"]
                q = ensure_public_dataset(q)
                if not is_duplicate_query(query_history, "final", q):
                    query_history.append({"type": "final", "query": q, "result": {}})
                
                # Update arguments in tool call
                if isinstance(args, dict):
                    args["query"] = q
                else:
                    tc["args"] = json.dumps({"query": q})

        return {"messages": [final_response], "query_history": query_history}

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error during query generation: {str(e)}")]}


def print_query_history(state: AgentState):
    print("\n=== Query History ===")
    query_history = state.get('query_history', [])
    if not query_history:
        print("No query history available.")
    else:
        for entry in query_history:
            print(f"Type: {entry.get('type', 'Unknown')}")
            print(f"Query: {entry.get('query', 'No query available')}")
            print(f"Result: {json.dumps(entry.get('result', {}), ensure_ascii=False, indent=2)}")
            print("---")
    return {}


def after_run_query(state: AgentState):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, ToolMessage):
        result = last_msg.content
        query_history = list(state.get("query_history", []))
        last_type = None
        
        # Find the corresponding entry in history and update its result
        for i in range(len(query_history) - 1, -1, -1):
            if query_history[i].get("result") == {}:
                query_history[i]["result"] = result
                last_type = query_history[i].get("type")
                break

        updates = {"query_history": query_history}
        if last_type:
            updates["last_query_type"] = last_type
        if last_type == "exploratory":
            updates["exploratory_results"] = result
        return updates
    return {}


def check_schema_result(state: AgentState) -> Literal["store_schema", "continue_loop"]:
    """Checks if the detailed schema was retrieved successfully (simplified as we use overview.txt)."""
    last_message = state["messages"][-1]
    # If the tool message contains an error string, go back to loop. Otherwise, store the result.
    if isinstance(last_message, ToolMessage) and '"error":' in str(last_message.content):
        return "continue_loop"
    else:
        return "store_schema"


def check_for_remaining_tables(state: AgentState) -> Literal["get_next_schema", END]:
    """Routes based on whether there are tables left to check."""
    if state.get("tables_to_check"):
        return "get_next_schema"
    else:
        return END


def final_answer(state: AgentState):
    user_question = state["messages"][0].content
    last_result = state["messages"][-1].content

    system_prompt = """אתה עוזר חכם שעונה על שאלות משתמשים.
קיבלת את תוצאות השאילתות מהדאטהבייס. 
עכשיו תסכם את המידע וענה על השאלה של המשתמש בצורה ישירה וברורה בשפה שבה נשאלת השאלה.
אם אין נתונים או הייתה שגיאה, הסבר בקצרה שלא נמצאה תשובה והצע מה ייתכן שחסר (למשל סטטוס/תאריך/סכמה).
"""

    response = llm.invoke([
        ("system", system_prompt),
        ("human", f"שאלה: {user_question}\n\nתוצאות:\n{last_result}")
    ])

    return {"messages": [response]}


def trace_node(state: AgentState):
    print("\n--- TRACE NODE ---")
    print("Last message:", state["messages"][-1])
    print("Query history:", json.dumps(state.get("query_history", []), ensure_ascii=False, indent=2))
    print("Tables to check:", state.get("tables_to_check", []))
    print("Current table:", state.get("current_table"))
    return {}


def run_query_node(state: AgentState):
    """Manually runs the query from the tool_calls in the last AIMessage."""
    last_ai = state["messages"][-1]
    if not isinstance(last_ai, AIMessage):
        return {"messages": [AIMessage(content="Error: No AI query found.")]}

    tool_calls = getattr(last_ai, "tool_calls", None)
    if not tool_calls:
        return {"messages": [AIMessage(content="Error: No tool call found in AI message.")]}

    # Use first tool call only
    tc = tool_calls[0]
    if tc.get("name") != "run_query_tool":
        return {"messages": [AIMessage(content=f"Unexpected tool: {tc.get('name')}")]}

    args = tc.get("args")
    query = args.get("query") if isinstance(args, dict) else json.loads(args)["query"]

    query = ensure_public_dataset(query)

    # Actually run the tool
    result = run_query_tool.invoke({"query": query})

    # Return as ToolMessage so after_run_query can pick it up
    return {
        "messages": [
            ToolMessage(content=result, tool_name="run_query_tool", tool_call_id=tc.get("id", "manual"))
        ]
    }

# ==========================
# Routing Logic
# ==========================

def route_after_run(state: AgentState) -> Literal["final_answer", "generate_query_or_loop", "final_failure_next_table"]:
    """Routes based on the result type of the last executed tool."""
    last_msg = state.get("messages", [])[-1]
    
    if not isinstance(last_msg, ToolMessage):
        # Should always be ToolMessage if we just executed a query, but as a safety net:
        return "final_answer"

    # If last execution was exploratory -> go build the final query
    if state.get("last_query_type") == "exploratory":
        return "generate_query_or_loop"
        
    # If final query execution:
    if state.get("last_query_type") == "final":
        is_empty_or_failed = is_query_result_empty_or_failed(last_msg.content)
        has_more_tables = bool(state.get("tables_to_check"))
        
        if is_empty_or_failed and has_more_tables:
            # Failed on final query, but there are more tables to check
            return "final_failure_next_table"
        else:
            # Success or failure on the last available table
            return "final_answer"
            
    return "final_answer" # Default fallback


def on_final_failure_reset_for_next_table(state: AgentState):
    """Resets exploration state before moving to the next table."""
    return {
        "exploration_done": False,
        "exploratory_results": "",
        "exploration_attempts": 0,
        "last_query_type": None,
        "current_table": None
    }

# ==========================
# Build The FINAL ROBUST Graph
# ==========================
builder = StateGraph(AgentState)

# Nodes
builder.add_node("select_candidates", select_candidate_tables)
builder.add_node("loop_entry", loop_entry_point)  
builder.add_node("get_next_schema", get_next_schema_to_check)
builder.add_node("store_schema", store_schema)
builder.add_node("generate_query_node", generate_query)
builder.add_node("after_run_query", after_run_query)
builder.add_node("run_query_node", run_query_node) # Replaced ToolNode
builder.add_node("final_answer", final_answer)
builder.add_node("trace_node", trace_node) # Optional for debug
builder.add_node("on_final_failure", on_final_failure_reset_for_next_table)

# Edges & Conditional Edges
builder.add_edge(START, "select_candidates")
builder.add_edge("select_candidates", "loop_entry")

# Loop condition: Are there tables left to check?
builder.add_conditional_edges(
    "loop_entry",
    check_for_remaining_tables,
    {
        "get_next_schema": "get_next_schema",
        END: END
    })

# Schema loading flow
builder.add_edge("get_next_schema", "generate_query_node") # Simplified: skip schema loading and go straight to generation

# Generation -> Execution
builder.add_edge("generate_query_node", "run_query_node")

# Execution -> Post Processing
builder.add_edge("run_query_node", "after_run_query")

# Post Processing Routing
builder.add_conditional_edges(
    "after_run_query",
    route_after_run,
    {
        "final_answer": "final_answer",
        "generate_query_or_loop": "generate_query_node", # Go back to generate final query
        "final_failure_next_table": "on_final_failure" # Try next table
    }
)

# On failure reset -> go back to loop_entry to try next table
builder.add_edge("on_final_failure", "loop_entry")

# Final step in the graph
builder.add_edge("final_answer", END)

agent = builder.compile()
print("✅ Agent graph compiled successfully!")