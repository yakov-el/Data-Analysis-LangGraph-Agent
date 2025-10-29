
````markdown
# 🤖 LangGraph Data Analysis Agent - E-commerce Insights

**Assignment:** AI Technical Assignment - Data Analysis LangGraph Agent

This agent is built using the LangGraph framework and the Gemini 2.5 Pro model to analyze the public Google BigQuery E-commerce dataset (`bigquery-public-data.thelook_ecommerce`) and generate business insights based on user queries.

---

## 1. Deliverables: Architecture Documentation

### 1.1 High-Level Architecture Diagram (Link to Image)

[**View Architecture Diagram: architecture.png**](architecture.png) 
*(Please ensure the image file is included in the repository)*

### 1.2 Technical Explanation

#### Reasoning for Chosen Cloud Services and LLM

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **LLM Backend** | **Google Gemini 2.5 Pro** | Chosen for its superior reasoning capabilities, which is crucial for the complex, multi-step task of dynamic SQL generation and context integration (schema, exploratory results). The high-quality structured output capabilities enhance the reliability of tool calling. |
| **Agent Framework** | **LangGraph** | Required by the assignment. It enables complex, multi-stage control flow, specifically the essential looping mechanism to sequentially explore candidate tables and the conditional routing for robust error and success management. |
| **Data Source** | **Google BigQuery** | Required by the assignment to access the specified public dataset. Utilized directly via the `google-cloud-bigquery` library for efficient data retrieval. |
| **Authentication** | **Application Default Credentials (ADC)** | Best practice for authenticating to Google Cloud services (BigQuery) in a development environment, ensuring security without hardcoding sensitive credentials. |

#### Data Flow and Agent Logic

The agent utilizes a stateful LangGraph structure (`AgentState`) to manage conversational history and data analysis context:

1.  **Start:** The user provides a natural language query (`cli_interface.py`).
2.  **Table Selection (`select_candidates`):** Gemini reads the `overview.txt` (containing all table schemas) and selects the 3-5 most relevant tables for the query.
3.  **Iterative Analysis (Loop):** The graph enters a loop (`loop_entry`, `get_next_schema`).
    * **Exploratory Query (`generate_query_node`):** For the first pass on a table, Gemini generates a small exploratory query to identify key values (e.g., date ranges, status fields). This query is executed via the `run_query_tool`.
    * **Final Query Generation (`generate_query_node`):** Once exploratory results are available, Gemini generates the final, complex SQL query using the original user question, the full schema, and the gathered exploratory data.
4.  **Execution (`run_query_node`):** The query is executed against BigQuery.
5.  **Routing & Fallback (`route_after_run`):**
    * **Success (Exploratory):** Loops back to generate the Final Query.
    * **Success (Final):** Proceeds to the `final_answer` node.
    * **Failure/Empty Result (Final):** Routes to the `on_final_failure` node, resets the state for exploration, and returns to the loop entry to check the **next candidate table**. This provides a robust fallback mechanism.
6.  **Summary (`final_answer`):** Gemini summarizes the successful query results into a clear business insight in Hebrew and returns it to the user.

#### Error Handling and Fallback Strategies

1.  **BigQuery Connection:** Relies on `gcloud auth application-default login` for robust initial connection setup.
2.  **Tool Execution Errors:** The `run_query_tool` catches Python and BigQuery exceptions (e.g., invalid SQL syntax) and returns the error message to the LLM for self-correction.
3.  **Empty/Failed Results:** The conditional edge logic (`route_after_run`) explicitly checks for empty or failed final query results. If a failure occurs and there are remaining candidate tables, the agent **automatically tries the next relevant table**, demonstrating resilience.

---

## 2. Deliverables: Working Application (CLI)

### Setup and Running the Agent

This project requires Python 3.8+, the Google Cloud SDK, and a Gemini API key.

#### Step 1: Clone Repository and Install Dependencies

```bash
# Clone your repository here
git clone <your_repo_link>
cd <your_repo_folder>

# Install Python libraries with pinned versions
pip install -r requirements.txt
````

#### Step 2: Configure Authentication

1.  **BigQuery Access (ADC):** You must authenticate your environment to access the public BigQuery dataset:
    ```bash
    gcloud auth application-default login
    ```
2.  **Gemini API Key:** Create a file named **`.env`** in the project root and add your API key:
    ```env
    GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
    ```

#### Step 3: Generate Schema Overview

Run the setup script once to connect to BigQuery and generate the context file (`overview.txt`) used by the agent:

```bash
python setup_overview.py
# Expected output: ✅ Overview generation complete. Saved to overview.txt
```

#### Step 4: Run the CLI Agent

Start the conversational interface:

```bash
python cli_interface.py
```

**Example Query:** `מהם 5 המוצרים המדורגים ביותר (highest rating) שנמכרו בארצות הברית בחודשים האחרונים?`

```

```
