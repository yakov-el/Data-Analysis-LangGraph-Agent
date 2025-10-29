import sys
import os
# מכיוון שהמצב (AgentState) והסוכן המוגמר (agent) מוגדרים ב-agent_graph.py,
# אנחנו פשוט מייבאים אותם.
try:
    from agent_graph import agent, AgentState
    from langchain_core.messages import HumanMessage, AIMessage # לצורך יצירת הודעת משתמש
except ImportError as e:
    print("FATAL ERROR: Failed to import agent_graph.")
    print("Please ensure 'agent_graph.py' is in the same directory and contains the 'agent' and 'AgentState' objects.")
    print(f"Details: {e}")
    sys.exit(1)

def chat_loop():
    """
    מריץ את לולאת הצ'אט ב-CLI, מקבל קלט מהמשתמש ומפעיל את LangGraph.
    """
    print("--- 🤖 Data Analysis Agent Initialized (powered by LangGraph & Gemini) 🤖 ---")
    print("Hello! Ask me anything about the e-commerce data (sales, users, products).")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 60)
    
    # אתחול המצב. ההודעות הראשוניות הן ריקות.
    # ה-invoke הראשון יוסיף את שאלת המשתמש למצב ההתחלתי.
    current_state = {
        "messages": []
    }

    while True:
        try:
            # 1. קבל קלט מהמשתמש
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting agent session. Goodbye! 👋")
                break

            if not user_input.strip():
                continue

            # 2. הכן את ההודעה למצב LangGraph
            # הוספת שאלת המשתמש הנוכחית למצב הקיים.
            current_state["messages"].append(HumanMessage(content=user_input))
            
            # 3. הפעל את הגרף
            # Agent.invoke מקבל מצב (state) ומחזיר מצב מעודכן (כולל התשובה הסופית)
            print("Agent is thinking and querying BigQuery... (This may take a moment)")
            result = agent.invoke(current_state)
            
            # 4. חילוץ התשובה הסופית והצגתה
            final_message = result["messages"][-1]
            
            # ודא שהתשובה הסופית היא הודעת AI
            if isinstance(final_message, AIMessage):
                print("-" * 60)
                print(f"Agent: {final_message.content}")
                print("-" * 60)
            else:
                # מקרה קצה אם הגרף הסתיים בצורה לא צפויה
                print(f"Agent finished but did not return a final AI message. Final state: {result}")
                
            # 5. עדכן את המצב עבור האינטראקציה הבאה (מכיל את כל ההיסטוריה)
            # אם היינו רוצים שהסוכן ישכח את האינטראקציה הקודמת, היינו מאתחלים: current_state = {"messages": []}
            current_state = result
            
        except Exception as e:
            print("\n🚨 ERROR: An unhandled error occurred during agent invocation.")
            print(f"Error details: {e}")
            # במקרה של שגיאה, מאתחלים את השיחה כדי למנוע קריסה בגלל מצב פגום
            current_state = {"messages": []}
            print("Restarting chat state. Please try your question again.")

if __name__ == "__main__":
    chat_loop()