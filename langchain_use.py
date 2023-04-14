
# To make this work on langchain:
# Define a BaseRetriever subclass, giving it an Indexer (which supports Wikipedia and VectorStores)
# ClaudeRetriever does a "Claude thing" in front of the underlying index. May do multiple rounds, etc.
# ClaudeRetriever supports modification of the starting prompt to guide it in a certain direction
#...gives a list of documents

from typing import Any, List, Tuple
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import Document
from langchain.schema import BaseRetriever
from pydantic import BaseModel
import logging
from anthropic import Client, count_tokens, HUMAN_PROMPT, AI_PROMPT, ApiException
from langchain_util import get_search_starting_prompt, extract_search_query
logger = logging.getLogger(__name__)

class ClaudeCombineDocumentsChain(BaseCombineDocumentsChain):
    def combine_docs(self, docs: List[Document], **kwargs: Any) -> Tuple[str, dict]:
        """Combine documents into a single string."""
        # TODO: Include more than just page content, like titles, etc
        # But not sure what the conventions are in that domain...
        formatted_docs = [x.page_content for x in docs]
        result = "\n".join(
            [
                f'<item index="{i+1}">\n<page_content>\n{r}\n</page_content>\n</item>'
                for i, r in enumerate(formatted_docs)
            ]
        )
        return result, {}

    async def acombine_docs(
        self, docs: List[Document], **kwargs: Any
    ) -> Tuple[str, dict]:
        """Combine documents into a single string asynchronously."""
        return self.combine_docs(docs, **kwargs)


class ClaudeRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    search_tool_description: str
    client: Client
    model: str = "claude-v1.4-tango"
    n_search_results_to_use: int = 3
    max_searches_to_try: int = 5
    max_tokens_to_sample: int = 1000
    combine_documents_chain: BaseCombineDocumentsChain = ClaudeCombineDocumentsChain()

    class Config:
        arbitrary_types_allowed = True

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        prompt = f'{HUMAN_PROMPT} {query}{AI_PROMPT}'
        prompt = get_search_starting_prompt(raw_prompt = prompt, search_tool = self.search_tool_description)
        starting_prompt = prompt
        token_budget = self.max_tokens_to_sample
        tries = 0
        all_raw_search_results: list[Document] = []
        while True:
            if tries >= self.max_searches_to_try:
                logger.warning(f'max_searches_to_try ({self.max_searches_to_try}) exceeded.')
                break
            result = self.client.completion(prompt = prompt,
                                                stop_sequences=[HUMAN_PROMPT, '</search_query>'],
                                                model=self.model,
                                                disable_checks=True,
                                                max_tokens_to_sample = self.max_tokens_to_sample)
            completion = result['completion']
            tries += 1
            partial_completion_tokens = count_tokens(completion)
            token_budget -= partial_completion_tokens
            prompt += completion
            if result['stop_reason'] == 'stop_sequence' and result['stop'] == '</search_query>':
                logger.info(f'Attempting search number {tries}.')

                search_query = extract_search_query(completion + '</search_query>')
                if search_query is None:
                    raise ApiException(f'Completion with retrieval failed as partial completion returned mismatched <search_query> tags.')
                logger.info(f'Running search query: {search_query}')
                #TODO: self.n_search_results_to_use??
                # maybe construct the retriever inline passing it a k value? No standardization here..
                search_results: List[Document] = await self.base_retriever.aget_relevant_documents(search_query)
                formatted_search_results = self.combine_documents_chain.combine_docs(search_results)[0]
                prompt += '</search_query>' + formatted_search_results
                all_raw_search_results += search_results
            else:
                break
        final_model_response = prompt[len(starting_prompt):]
        return all_raw_search_results
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        prompt = f'{HUMAN_PROMPT} {query}{AI_PROMPT}'
        prompt = get_search_starting_prompt(raw_prompt = prompt, search_tool = self.search_tool_description)
        starting_prompt = prompt
        token_budget = self.max_tokens_to_sample
        tries = 0
        all_raw_search_results: list[Document] = []
        while True:
            if tries >= self.max_searches_to_try:
                logger.warning(f'max_searches_to_try ({self.max_searches_to_try}) exceeded.')
                break
            print(prompt)
            result = self.client.completion(prompt = prompt,
                                                stop_sequences=[HUMAN_PROMPT, '</search_query>'],
                                                model=self.model,
                                                disable_checks=True,
                                                max_tokens_to_sample = self.max_tokens_to_sample)
            completion = result['completion']
            tries += 1
            partial_completion_tokens = count_tokens(completion)
            token_budget -= partial_completion_tokens
            prompt += completion
            if result['stop_reason'] == 'stop_sequence' and result['stop'] == '</search_query>':
                logger.info(f'Attempting search number {tries}.')

                search_query = extract_search_query(completion + '</search_query>')
                if search_query is None:
                    raise ApiException(f'Completion with retrieval failed as partial completion returned mismatched <search_query> tags.')
                logger.info(f'Running search query: {search_query}')
                #TODO: self.n_search_results_to_use??
                # maybe construct the retriever inline passing it a k value? No standardization here..
                search_results: List[Document] = self.base_retriever.get_relevant_documents(search_query)
                formatted_search_results = self.combine_documents_chain.combine_docs(search_results)[0]
                prompt += '</search_query>' + formatted_search_results
                all_raw_search_results += search_results
            else:
                break
        final_model_response = prompt[len(starting_prompt):]
        return all_raw_search_results
