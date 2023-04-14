from langchain.chains import RetrievalQA
from langchain.llms import Anthropic
from langchain.retrievers import TFIDFRetriever
from langchain_use import ClaudeCombineDocumentsChain, ClaudeRetriever

search_backend = TFIDFRetriever.from_texts(['doc1', 'doc2 contents'])
WIKIPEDIA_DESCRIPTION = 'The search engine will exclusively search over Wikipedia for pages with keywords similar to your query. It returns for each page its title and full page content. Use this tool if you want to get up-to-date and comprehensive information on a topic to help answer queries.'
from anthropic import Client
API_KEY = "sk-kQJhew0Yl3FBA9s7h4QTT_MjTaK7IWSYuWIk5S7mbiaQfz6gSpTL3OpAXp66W4kqwFF4VwBfKGzpK8ocT9ro6w"

# TODO: Make this a langchain client
anthropic_client = Client(api_key=API_KEY)
claude_retriever = ClaudeRetriever(
    base_retriever=search_backend,
    # TODO: This probably should be some description that properly captures the nature of the base retriever
    search_tool_description=WIKIPEDIA_DESCRIPTION,
    client=anthropic_client
)



results = claude_retriever.get_relevant_documents('What are good document names?')
print(results)
formatted_string = ClaudeCombineDocumentsChain().combine_docs(results)
print(formatted_string)



# TODO: This doesn't work. The RetrievalQA has a few too many layers, and expects The CombineDocuments to actually hit the LLM vs just formatting text.
# And so I haven't bene able to figure out how to hook things in properly here
retrievalQA = RetrievalQA.from_llm(llm=Anthropic(anthropic_api_key=API_KEY), retriever=claude_retriever)
retrievalQA.combine_documents_chain = ClaudeCombineDocumentsChain()
retrievalQA.combine_documents_chain.llm_chain.prompt.template = """\
\n\nHuman: Here are some documents:

{context}

Now please answer the following question:
{question}

Assistant:"""
#result = retrievalQA.run('What is the capital of France?')
#print(result)
result = retrievalQA.run(prompt='What are good document names?')
print(result)
