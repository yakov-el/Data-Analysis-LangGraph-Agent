# Architecture — LangGraph Data Analysis Agent

This document describes the architecture used for the LangGraph Data Analysis Agent, aligned with the assignment requirements.

## Overview
The agent is a stateful LangGraph StateGraph that orchestrates LLM reasoning and tool usage. The primary flow:
1. User asks a natural-language question (CLI).
2. The agent reads `overview.txt` (schema + sample rows) to understand available tables and columns.
3. The agent runs a selection step to pick candidate tables (3–5).
4. For each candidate table: run a single exploratory query (tool call) to gather sample rows or simple aggregates.
5. The agent constructs a single final query using the exploratory results and executes it via `run_query_tool` (BigQuery).
6. The agent summarizes results via the LLM and returns a human-readable answer.

## Components
- **CLI (User)**: Simple REPL that forwards the user's question into the StateGraph messages.
- **Overview Generator (`overview_generator.py`)**: Produces `overview.txt` containing table schema and up to 5 sample rows per table.
- **StateGraph Agent (`agent_graph.py`)**:
  - `select_candidate_tables` node: uses LLM structured output to choose candidate tables.
  - `generate_query` node: creates exploratory and final SQL queries. Enforces format rules and uses `ensure_public_dataset()`.
  - `run_query_node` / `run_query_tool`: executes the SQL on BigQuery and returns ToolMessage with JSON results.
  - `after_run_query` node: stores results in `query_history` and decides routing.
  - `final_answer` node: LLM summarizes the final result.
- **LLM Backend**: Google Gemini (recommended) via `langchain-google-genai` / `google-generativeai`. Bedrock is listed as an alternative.
- **BigQuery**: `bigquery-public-data.thelook_ecommerce` (orders, order_items, products, users).

## Data flow
User -> CLI -> StateGraph (select candidates) -> LLM -> exploration tool calls -> BigQuery -> results -> StateGraph -> final answer -> CLI (user)

See `docs/langgraph_architecture.png` for a visual diagram (generated and included in this repo).

## Security & Cost Controls
- Enforce least privilege on Service Account (roles/bigquery.dataViewer or custom minimal role).
- Use query filters and `LIMIT` in exploratory queries.
- Consider `dryRun` or `maxBytesBilled` when running expensive queries in production.
- Never store credentials in the repository.

## Error handling & fallback strategies
- If `run_query_tool` returns `[]` or a permission error, the agent logs the issue and attempts the next candidate table.
- If exploratory queries found insufficient information, agent falls back to a conservative final query (returning "insufficient data" if necessary).
- The agent uses `is_query_result_empty_or_failed()` to detect failures and route accordingly.

## Rationale for choices
- **LangGraph**: Structured state + routing simplifies safe multi-step query planning and rerouting when tables fail.
- **Gemini / Bedrock**: Strong instruction-following and JSON/tool-call capabilities; Gemini integrates nicely with Google Cloud ecosystem.
- **BigQuery public dataset**: Avoids private data and simplifies evaluation for reviewers.

