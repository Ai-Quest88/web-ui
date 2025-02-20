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
import logging
import json
from datetime import datetime
import os
import tempfile
import uuid
from httpx import Client

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

# Create tmp directory in current project directory
log_dir = os.path.join(os.getcwd(), 'tmp', 'logs')
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, 'custom_llm_debug.log')

# Set up logging at the top of the file
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True  # Force configuration to ensure it takes effect
)

# Print the log file location to console
print(f"Log file location: {log_file}")

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
        """Process async invocation"""
        try:
            messages = self._convert_input_to_messages(input)
            message_dicts = []
            
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    message_dicts.append({"role": "system", "content": str(msg.content)})
                elif isinstance(msg, AIMessage):
                    message_dicts.append({"role": "assistant", "content": str(msg.content)})
                else:
                    message_dicts.append({"role": "user", "content": str(msg.content)})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts
            )
            
            return AIMessage(content=str(response.choices[0].message.content))
        except Exception as e:
            logging.error(f"Error in ainvoke: {str(e)}", exc_info=True)
            raise
    
    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Process synchronous invocation"""
        try:
            messages = self._convert_input_to_messages(input)
            message_dicts = []
            
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    message_dicts.append({"role": "system", "content": str(msg.content)})
                elif isinstance(msg, AIMessage):
                    message_dicts.append({"role": "assistant", "content": str(msg.content)})
                else:
                    message_dicts.append({"role": "user", "content": str(msg.content)})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts
            )
            
            return AIMessage(content=str(response.choices[0].message.content))
        except Exception as e:
            logging.error(f"Error in invoke: {str(e)}", exc_info=True)
            raise

    def _convert_input_to_messages(self, input: LanguageModelInput) -> list[BaseMessage]:
        """Convert input to messages"""
        if isinstance(input, list):
            return [m if isinstance(m, BaseMessage) else HumanMessage(content=str(m)) for m in input]
        elif isinstance(input, BaseMessage):
            return [input]
        else:
            return [HumanMessage(content=str(input))]

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
        base_url = ""
        api_version = kwargs.get("api_version", "2024-10-21")
        
        logging.debug(f"Initializing CustomAzureOpenAI with base URL: {base_url}")
        
        # Create headers dict first
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'x-subscription-key': kwargs.get("api_key", ""),
            'x-correlation-id': str(uuid.uuid4())
        }
        
        logging.debug(f"Using headers: {headers}")
        
        # Create client with minimal configuration like in the image
        self.client = OpenAI(
            base_url=base_url,
            api_key=kwargs.get("api_key", ""),
            default_headers=headers
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
            
            # Add extra headers for each request
            extra_headers = {
                'x-correlation-id': str(uuid.uuid4())
            }
            
            # Get headers without Omit objects
            all_headers = dict(self.client.default_headers)
            headers_to_log = {k: v for k, v in all_headers.items() 
                            if not (hasattr(v, '__class__') and v.__class__.__name__ == 'Omit')}
            headers_to_log.update(extra_headers)
            
            # Log request details
            request_details = {
                'url': str(self.client.base_url),
                'headers': headers_to_log,
                'request': {
                    'model': self.model_name,
                    'messages': message_dicts,
                    'temperature': self.temperature,
                    'max_tokens': 100,
                    'n': 1
                }
            }
            logging.debug(f"Request Details:\n{json.dumps(request_details, indent=2)}")
            
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=message_dicts,
                temperature=self.temperature,
                max_tokens=100,
                n=1,
                extra_headers=extra_headers
            )
            
            # Log response details
            response_details = {
                'status': 'success',
                'response': {
                    'content': completion.choices[0].message.content,
                    'role': completion.choices[0].message.role
                }
            }
            logging.debug(f"Response Details:\n{json.dumps(response_details, indent=2)}")
            
            response_content = self._sanitize_text(completion.choices[0].message.content)
            return AIMessage(content=response_content)
        except Exception as e:
            # Log error details
            error_details = {
                'status': 'error',
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            logging.error(f"Error Details:\n{json.dumps(error_details, indent=2)}", exc_info=True)
            raise

    def _convert_input_to_messages(self, input: LanguageModelInput) -> list[BaseMessage]:
        """Helper method to convert input to messages"""
        if isinstance(input, list):
            return [m if isinstance(m, BaseMessage) else HumanMessage(content=str(m)) for m in input]
        elif isinstance(input, BaseMessage):
            return [input]
        else:
            return [HumanMessage(content=self._sanitize_text(input))]

    def _construct_endpoint(self, deployment_name: str) -> str:
        """Override to return the exact URL"""
        return self.client.base_url

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle Omit type
        if hasattr(obj, '__class__') and obj.__class__.__name__ == 'Omit':
            return str(obj)  # or return a dict representation if available
        return super().default(obj)