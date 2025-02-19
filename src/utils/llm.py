from openai import OpenAI
import pdb
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    AnyMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,
    RunInfo,
)
from langchain_ollama import ChatOllama
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

class DeepSeekR1ChatOpenAI(ChatOpenAI):
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.client = OpenAI(
            base_url=kwargs.get("base_url"),
            api_key=kwargs.get("api_key")
        ) 
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        message_history = []
        for input_ in input:
            if isinstance(input_, SystemMessage):
                message_history.append({"role": "system", "content": input_.content})
            elif isinstance(input_, AIMessage):
                message_history.append({"role": "assistant", "content": input_.content})
            else:
                message_history.append({"role": "user", "content": input_.content})
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=message_history
        )

        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
class DeepSeekR1ChatOllama(ChatOllama):
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = await super().ainvoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        org_ai_message = super().invoke(input=input)
        org_content = org_ai_message.content
        reasoning_content = org_content.split("</think>")[0].replace("<think>", "")
        content = org_content.split("</think>")[1]
        if "**JSON Response:**" in content:
            content = content.split("**JSON Response:**")[-1]
        return AIMessage(content=content, reasoning_content=reasoning_content)

class CustomAzureOpenAI(AzureChatOpenAI):
    """Custom Azure OpenAI wrapper with additional headers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a custom client with the required headers
        self.client = OpenAI(
            base_url=kwargs.get('azure_endpoint', ''),
            api_key=kwargs.get("api_key", ""),
            default_headers={
                'Accept': 'application/json',
                'x-subscription-key': kwargs.get("api_key", ""),
                'x-correlation-id': kwargs.get("x_correlation_id", ""),
                'Content-Type': 'application/json',
            }
        )
        self.model_name = "gpt-35-turbo"
        self.temperature = kwargs.get("temperature", 0.7)
        
    def _sanitize_text(self, text: Any) -> str:
        """Sanitize text to ensure it's properly encoded"""
        if text is None:
            return ""
        try:
            return str(text).encode('utf-8', errors='ignore').decode('utf-8')
        except Exception:
            return str(text)
        
    def _convert_message_to_dict(self, message: BaseMessage) -> dict:
        """Convert a message to a dictionary format that the API expects"""
        role = "user"
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, AIMessage):
            role = "assistant"
            
        return {
            "role": role,
            "content": self._sanitize_text(message.content)
        }
        
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Override invoke to use our custom client"""
        try:
            messages = self._convert_input_to_messages(input)
            message_dicts = [self._convert_message_to_dict(m) for m in messages]
            
            print(f"Sending request to: {self.client.base_url}")
            print(f"Headers: {self.client.default_headers}")
            print(f"Messages: {message_dicts}")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=100,
                n=1
            )
            response_content = self._sanitize_text(completion.choices[0].message.content)
            return AIMessage(content=response_content)
        except Exception as e:
            print(f"Error in invoke: {str(e)}")
            raise
        
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Override ainvoke to use our custom client"""
        try:
            messages = self._convert_input_to_messages(input)
            message_dicts = [self._convert_message_to_dict(m) for m in messages]
            
            completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=100,
                n=1
            )
            response_content = self._sanitize_text(completion.choices[0].message.content)
            return AIMessage(content=response_content)
        except Exception as e:
            print(f"Error in ainvoke: {str(e)}")
            raise

    def _convert_input_to_messages(self, input: LanguageModelInput) -> list[BaseMessage]:
        """Helper method to convert input to messages"""
        if isinstance(input, list):
            return input
        elif isinstance(input, BaseMessage):
            return [input]
        else:
            return [HumanMessage(content=self._sanitize_text(input))]

    def _construct_endpoint(self, deployment_name: str) -> str:
        """Override to return the exact URL with api-version"""
        return self.client.base_url