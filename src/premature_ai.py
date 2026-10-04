import os
from functools import lru_cache

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.main import create_model, SMART, API_KEY, BASE_URL
# Import the vector store search function
from src.rag.BQ_db import search_knowledge_base, is_configured as knowledge_base_configured

# Where premature_context comes from, so answers can link to it
WHO_FACT_SHEET = {
    "title": "WHO: Preterm birth fact sheet",
    "url": "https://www.who.int/news-room/fact-sheets/detail/preterm-birth",
    "kind": "who",
}

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
You are Njiti, an expert assistant specializing in all aspects of prematurity ("njiti" is Swahili for a baby
born too early). Your purpose is to help families, health workers, clinicians and researchers understand and
answer questions about premature birth, neonatal care, outcomes, and related research.

=====Your Role:=====

- Only answer questions related to prematurity, pregnancy and preterm birth, neonatal care, preterm infants, and
  associated research or clinical topics.
- Use facts from peer-reviewed literature, clinical guidelines, reputable medical sources, and established best practices.
- Search the knowledge base for clinical protocols, research findings, and neonatal care documentation when relevant.
- Never guess or make unsupported assumptions.
- If a question is unclear or lacks necessary details, ask for clarification (e.g., gestational age, clinical context).
- If unsure or out of scope, reply: "I'm not confident I can provide an accurate answer to that. Please try rephrasing or narrowing your question."

=====Safety:=====

- If the question describes danger signs in a baby (not breathing or struggling to breathe, blue lips or skin,
  convulsions/fits, not feeding or unable to suck, very cold or hot to the touch, very sleepy or floppy, yellow
  palms or soles, bleeding, a red or pus-filled umbilical cord), start your answer by telling them to take the
  baby to the nearest health facility now. Then give brief safe actions for the way there (e.g. keep the baby
  warm skin-to-skin).
- Do the same for danger signs in pregnancy (bleeding, leaking fluid, regular contractions before 37 weeks,
  severe headache or blurred vision, fever, the baby moving less).
- You give general information, not a diagnosis. Never give medicine doses for a specific baby; refer to their
  health worker.

=====Thinking Process:=====

- Think step-by-step before using tools or providing answers.
- Be concise, clear, and accurate.
- Use conversation history as memory, but do not carry memory across unrelated topics.

=====Language:=====
- Reply in the language the user writes in (for example Swahili or English).
- Use plain words a parent can follow; explain medical terms the first time you use them.

=====Output Format (web chat, Markdown):=====
- Start with a direct one or two sentence answer, then details.
- Use short **bold** section titles (no `#` headings), bullet points and numbered steps.
- Bold the key terms, values and important findings.
- Cite where facts come from with inline Markdown links when a source has a URL,
  e.g. [WHO](https://www.who.int/news-room/fact-sheets/detail/preterm-birth). Name the document for sources without a URL.
- Keep replies under about 300 words unless the user asks for more.

Your performance is measured by accuracy, usefulness, and clarity in helping users understand and analyze prematurity-related topics.
"""


# Function to define Tools for the Agent
# Each tool returns (text for the model, list of sources for the website to show)

@tool(response_format="content_and_artifact")
def get_context() -> tuple[str, list[dict]]:
    """To best answer questions about prematurity and neonatal care, this function provides context information
    from the WHO preterm birth fact sheet."""
    return f"Source: {WHO_FACT_SHEET['title']} ({WHO_FACT_SHEET['url']})\n{premature_context}", [WHO_FACT_SHEET]

@tool(response_format="content_and_artifact")
def search_preterm_knowledge(query: str) -> tuple[str, list[dict]]:
    """Search the Preterm knowledge base (clinical guidelines, protocols and research documents) for relevant information.

    Args:
        query (str): The search query to find relevant information or context

    Returns:
        str: Relevant context from Preterm related documentation
    """
    if not knowledge_base_configured():
        return "The knowledge base is not available right now. Use get_context instead.", []

    try:
        results = search_knowledge_base(query, k=3)  # Get top 3 most relevant documents

        if not results:
            return "No relevant information found in the knowledge base.", []

        # This formats the context parts for better readability
        # It extracts source information from the metadata of each document and formats it nicely.
        context_parts = []
        sources = []
        for doc in results:
            source_file = doc.metadata.get('source_file') or doc.metadata.get('source', 'Unknown source')
            title = os.path.basename(source_file)
            page = doc.metadata.get('page')
            if page is not None:
                title = f"{title}, page {int(page) + 1}"  # PyPDFLoader pages start at 0

            url = doc.metadata.get('url')
            sources.append({"title": title, "url": url, "kind": "document"})
            context_parts.append(f"Source: {title}" + (f" ({url})" if url else "") + f"\n{doc.page_content}")

        # Join all context parts with a separator for better readability
        return "\n\n---\n\n".join(context_parts), sources

    except Exception as e:
        return f"Error searching knowledge base: {str(e)}", []
"""
This code defines the tools for the PrematureAI agent:
The first tool, get_context, provides context information about prematurity and neonatal care.
The second tool, search_preterm_knowledge, searches the medical knowledge base for relevant information.
These tools enable the agent to answer medical queries by accessing context and searching medical documentation.

"""
tools = [get_context, search_preterm_knowledge]

checkpointer = InMemorySaver()

# Function to initialize the Agent. Built on first use so the app can start before credentials are set.
@lru_cache(maxsize=1)
def get_agent():
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    llm = create_model(API_KEY, SMART, BASE_URL)
    return create_react_agent(
        tools=tools,
        model=llm,
        checkpointer=checkpointer,
        prompt=system_prompt
    )

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

# Console version of the agent. The website (src/server.py) uses the same agent.
if __name__ == "__main__":
    agent = get_agent()
    config = {"configurable": {"thread_id": "cli"}}
    print("Agent initialized. You can now use it to ask preterm related knowledge.")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ('exit', 'quit'):
            print("Exiting the agent.")
            break
        inputs = {"messages": [{"role": "user", "content": user_input}]}
        print_stream(agent.stream(inputs, stream_mode="updates", config=config))
