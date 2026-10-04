# this is used during the extraction
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveJsonSplitter
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from src.main import create_model, SMART
from typing import Optional, List
# Import the vector store search function
from src.rag.BQ_db import search_knowledge_base

# This is the high-level context for the premature ai agent
premature_context = """
    Key facts
    - An estimated 13.4 million babies were born preterm in 2020 (before 37 completed weeks of gestation) (1).
    - Preterm birth complications are the leading cause of death among children under 5 years of age, responsible for 
    approximately 900 000 deaths in 2019 (2).
    - Three-quarters of these deaths could be prevented with current, cost-effective interventions.
    - Across countries, the rate of preterm birth ranges from 4–16% of babies born in 2020.
    
    Overview
    - Preterm is defined as babies born alive before 37 weeks of pregnancy are completed. There are sub-categories of 
    preterm birth, based on gestational age:
        a) extremely preterm (less than 28 weeks)
        b) very preterm (28 to less than 32 weeks)
        c) moderate to late preterm (32 to 37 weeks).
    
    Babies may be born preterm because of spontaneous preterm labour or because there is a medical indication to 
    plan an induction of labour or caesarean birth early.

    An estimated 13.4 million babies were born too early in 2020. That is more than 1 in 10 babies. Approximately 
    900 000 children die in 2019 of complications of preterm birth (1). Many survivors face a lifetime of disability, 
    including learning disabilities and visual and hearing problems.

    Globally, prematurity is the leading cause of death in children under the age of 5 years. Inequalities in survival 
    rates around the world are stark. In low-income settings, half of the babies born at or below 32 weeks 
    (2 months early) die due to a lack of feasible, cost-effective care such as warmth, breastfeeding support and 
    basic care for infections and breathing difficulties. In high-income countries, almost all these babies survive. 
    Suboptimal use of technology in middle-income settings is causing an increased burden of disability among preterm 
    babies who survive the neonatal period.
    
    Why does preterm birth happen?
    a) Preterm birth occurs for a variety of reasons. Most preterm births happen spontaneously, but some are due to 
    medical reasons such as infections, or other pregnancy complications that require early 
    induction of labour or caesarean birth.
    
    b) More research is needed to determine the causes and mechanisms of preterm birth. Causes include multiple 
    pregnancies, infections and chronic conditions such as diabetes and high blood pressure; however, 
    often no cause is identified. There could also be a genetic influence.

    Where and when does preterm birth happen?
    The majority of preterm births occur in southern Asia and sub-Saharan Africa, but preterm birth is truly a 
    global problem. There is a dramatic difference in survival of premature babies depending on 
    where they are born. For example, more than 90% of extremely preterm babies (less than 28 weeks) 
    born in low-income countries die within the first few days of life, yet less than 10% of extremely 
    preterm babies die in high-income settings.
    
    The solution
    Preventing deaths and complications from preterm birth starts with a healthy pregnancy. WHO’s antenatal care 
    guidelines include key interventions to help prevent preterm birth, such as counselling on healthy diet, 
    optimal nutrition, and tobacco and substance use; fetal measurements including use of early ultrasound 
    to help determine gestational age and detect multiple pregnancies; and a minimum of 8 contacts with 
    health professionals throughout pregnancy – starting before 12 weeks – to identify and manage risk factors 
    such as infections.
    
    If a woman experiences preterm labour or is at risk of preterm childbirth, treatments are available to 
    help protect the preterm baby from future neurological impairment as well as difficulties 
    with breathing and infection. These include antenatal steroids and tocolytic treatments to 
    delay labour and antibiotics for preterm prolabour rupture of membranes (PPROM).
    
    In 2022, WHO also published new recommendations on the care of the preterm infant. These reflect new evidence 
    that simple interventions such as kangaroo mother care immediately after birth, early initiation of 
    breastfeeding, use of continuous positive airway pressure (CPAP) and medicines such as caffeine for 
    breathing problems can substantially reduce mortality in preterm and low birthweight babies.
    
    WHO guidance stresses the need to ensure the mother and family take the pivotal role in their baby’s care. 
    Mothers and newborns should remain together from birth and not be separated unless the baby is critically ill. 
    The recommendations further call for improvements in family support including education and counselling, 
    peer support and home visits by trained health-care providers.

"""

# System prompt
system_prompt = """
You are PrematureAI, an expert assistant specializing in all aspects of prematurity. Your purpose is to help clinicians, 
researchers, and families understand, analyze, and answer questions about premature birth, neonatal care, outcomes, 
and related research.

=====Your Role:=====

- Only answer questions related to prematurity, neonatal care, preterm infants, and associated research or clinical topics.
- Use facts from peer-reviewed literature, clinical guidelines, reputable medical sources, and established best practices.
- Search the knowledge base for clinical protocols, research findings, and neonatal care documentation when relevant.
- Never guess or make unsupported assumptions.
- Use tools and external resources only when needed.
- If a question is unclear or lacks necessary details, ask for clarification (e.g., gestational age, clinical context).
- If unsure or out of scope, reply: "I'm not confident I can provide an accurate answer to that. Please try rephrasing or narrowing your question."

- Respond quickly with progress updates if a query may take longer than 5 seconds. The progress update should be concise.

=====Thinking Process:=====

- Think step-by-step before using tools or providing answers.
- Be concise, clear, and accurate.
- Use bullet points, field-value pairs, and brief explanations.
- Use conversation history as memory, but do not carry memory across unrelated topics.

=====Output Format (WhatsApp):=====
- Use \*asterisks\* for all emphasis—\*no\* single-asterisk mistakes.
- **Always** prefix all titles with your chosen emoji.
  e.g. `🍼 *Prematurity Risk Factors*…`
- **Always** bold and color key terms, values, and important findings.
- **Do not** use any Markdown heading syntax (no `#`, `##`, or `###`).
- For section titles or “headers,” simply write them in bold.
  e.g.
  *Neonatal Outcomes Summary*
- Wrap single identifiers (like a study ID or code blocks) in inline backticks:
  e.g.
  `` `NCT01234567` ``
- When you need a fixed-width table or block, wrap it in triple backticks (```…```) so Slack renders it properly.
- Use bullet points (\-) or numbered lists for structure.
- Leave a blank line before and after any triple-backtick block.
- Keep replies under 3,000 characters; split into threaded follow-ups if needed.

Your performance is measured by accuracy, usefulness, and clarity in helping users understand and analyze prematurity-related topics.
"""


# Initialize the LLM model for medical queries
llm = create_model("my-openai-api-key", SMART)

# Create a prompt template for medical information extraction
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are PrematureAI, an expert in prematurity and neonatal care. Provide accurate, evidence-based information."
        ),
        ("human", "Question: {question}"),
    ]
)

# Chunk size for processing medical documents
JSON_CHUNK_SIZE = 50000
splitter = RecursiveJsonSplitter(max_chunk_size=JSON_CHUNK_SIZE)

class MedicalInsight(BaseModel):
    """Medical information extracted from documents"""
    insight: str = Field(
        ..., description="The key medical information or answer to the user's question."
    )
    source: str = Field(
        ..., description="The source of this medical information."
    )

class MedicalResponse(BaseModel):
    """Structured medical response"""

    insights: List[MedicalInsight]

extractor = prompt | llm.with_structured_output(
    schema=MedicalResponse,
    include_raw=False,
)


# Function to define Tools for the Agent

@tool
def get_context() -> str:
    """To best answer questions about prematurity and neonatal care, this function provides context information."""
    return premature_context

@tool
def search_preterm_knowledge(query: str) -> str:
    """Search the Preterm knowledge base for relevant information.

    Args:
        query (str): The search query to find relevant information or context

    Returns:
        str: Relevant context from Preterm related documentation
    """
    try:
        results = search_preterm_knowledge(query, k=3)  # Get top 3 most relevant documents

        if not results:
            return "No relevant information found in the knowledge base."

        # This formats the context parts for better readability
        # It extracts source information from the metadata of each document and formats it nicely.
        context_parts = []
        for doc in results:
            source_type = doc.metadata.get('source_type', 'Unknown')
            source_file = doc.metadata.get('source_file', 'Unknown source')

            # This formats source information
            if source_type == 'pdf':
                source_info = f"Source: {source_file.split('/')[-1]}"  # Just filename
            else:
                source_info = f"Source: {source_file}"

            context_parts.append(f"{source_info}\n{doc.page_content}")

        # Join all context parts with a separator for better readability
        return "\n\n---\n\n".join(context_parts)

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"
"""
This code defines the tools for the PrematureAI agent:
The first tool, get_context, provides context information about prematurity and neonatal care.
The second tool, search_preterm_knowledge, searches the medical knowledge base for relevant information.
These tools enable the agent to answer medical queries by accessing context and searching medical documentation.

"""
tools = [get_context, search_preterm_knowledge]

checkpointer = InMemorySaver()

# Function to initialize the Agent
agent = create_react_agent(
    tools =tools,
    model =llm,
    checkpointer=checkpointer,
    prompt=system_prompt
)

config = {
    "configurable": {
        "thread_id": "1"

    },

}
# This is langgraph's stream function to print updates and messages from the agent
def print_stream(stream, output_messages_key="llm_input_messages"):
    for chunk in stream:
        for node, update in chunk.items():
            print(f"Update from node: {node}")
            messages_key = (
                output_messages_key if node == "pre_model_hook" else "messages"
            )
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

        print("\n\n")

# This is the main function that initializes the agent and starts a loop to accept user input
# This can be removed if you want to use the agent in a different way, such as in a web application or another interface.
if __name__ == "__main__":
    print("Agent initialized. You can now use it to ask preterm related knowledge.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ('exit', 'quit'):
            print("Exiting the agent.")
        inputs = {"messages": [{"role": "user", "content": user_input}]}
        print_stream(agent.stream(inputs, stream_mode="updates", config=config))