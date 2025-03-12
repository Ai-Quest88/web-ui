from openai import OpenAI
import pdb
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.globals import get_llm_cache
from langchain_core.language_models.chat_models import BaseChatModel
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
from langchain_mistralai import ChatMistralAI
from langchain.callbacks.manager import CallbackManagerForLLMRun
import logging
import json
from datetime import datetime
import os
import tempfile
import uuid
from httpx import Client
import asyncio
import time
import random
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    ClassVar,
)
import traceback

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

# Set up logging
logger = logging.getLogger(__name__)

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
        reasoning_content, content = self._extract_reasoning_and_content(org_content)
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
        reasoning_content, content = self._extract_reasoning_and_content(org_content)
        return AIMessage(content=content, reasoning_content=reasoning_content)

    def _extract_reasoning_and_content(self, content: Union[str, list, dict]) -> Tuple[Optional[str], str]:
        """Extract reasoning and content from response."""
        try:
            # Convert content to string if it's not already
            if not isinstance(content, str):
                content = str(content)
            
            # Split by think tags if present
            if "<think>" in content and "</think>" in content:
                parts = content.split("</think>", 1)
                if len(parts) == 2:
                    reasoning = parts[0].replace("<think>", "").strip()
                    content = parts[1].strip()
                else:
                    reasoning = None
                    content = content.strip()
            else:
                reasoning = None
                content = content.strip()
            
            # Extract JSON response if present
            if "**JSON Response:**" in content:
                content = content.split("**JSON Response:**", 1)[1].strip()
            
            return reasoning, content
            
        except Exception as e:
            logger.error(f"Error extracting reasoning and content: {str(e)}")
            return None, str(content)

class CustomAzureOpenAI(AzureChatOpenAI):
    """Custom Azure OpenAI wrapper with additional headers"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        api_version = kwargs.get("api_version", "2024-10-21")
        
        # Get base URL from azure_endpoint
        base_url = kwargs.get("azure_endpoint", "").rstrip('/')
        if not base_url:
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip('/')
            
        # Add API version to base URL
        self._base_url = f"{base_url}?api-version={api_version}"
        
        logging.debug(f"Initializing CustomAzureOpenAI with base URL: {self._base_url}")
        
        # Create headers dict
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'x-subscription-key': kwargs.get("api_key", ""),
            'x-correlation-id': str(uuid.uuid4())
        }
        
        logging.debug(f"Using headers: {headers}")
        
        # Create client with minimal configuration
        self.client = OpenAI(
            base_url=self._base_url,
            api_key=kwargs.get("api_key", ""),
            default_headers=headers
        )
        self.model_name = kwargs.get("model_name", "gpt-35-turbo")
        self.temperature = kwargs.get("temperature", 0.7)
        
    def _get_clean_url(self) -> str:
        """Get base URL without trailing slash"""
        return self._base_url.rstrip('/')

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
            
            # Add correlation ID for this request
            extra_headers = {
                'x-correlation-id': str(uuid.uuid4())
            }
            
            # Get actual headers being sent, handling Omit objects
            actual_headers = {}
            for k, v in self.client.default_headers.items():
                if hasattr(v, '__class__') and v.__class__.__name__ == 'Omit':
                    continue  # Skip Omit objects
                actual_headers[k] = v
            actual_headers.update(extra_headers)
            
            # Create completion parameters
            completion_params = {
                'model': self.model_name,
                'messages': message_dicts,
                'temperature': self.temperature,
                'max_tokens': 100,
                'n': 1,
                'extra_headers': extra_headers
            }
            
            # Log request details with all information
            request_details = {
                'request_url': self._get_clean_url(),  # Use our clean URL getter
                'request_headers': {
                    k: '****' if k.lower() in ['x-subscription-key', 'authorization'] else v 
                    for k, v in actual_headers.items()
                },
                'completion_params': completion_params
            }
            logging.debug(f"Request Details:\n{json.dumps(request_details, indent=2)}")
            
            # Make the API call with same parameters we logged
            completion = self.client.chat.completions.create(**completion_params)
            
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

class CustomMistralAI(BaseChatModel):
    def __init__(self, model_name="mistral-large-latest"):
        """Initialize the Mistral AI client."""
        super().__init__()
        self.model_name = model_name
        self.llm = ChatMistralAI(model_name=model_name)
        self.logger = logging.getLogger(__name__)
        self._last_request_time = 0
        self._min_request_interval = 1.0  # Minimum time between requests in seconds

    @property
    def _llm_type(self) -> str:
        """Return identifier of llm."""
        return "custom_mistral"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response."""
        message_dicts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                message_dicts.append({"role": "system", "content": str(message.content)})
            elif isinstance(message, AIMessage):
                message_dicts.append({"role": "assistant", "content": str(message.content)})
            else:
                message_dicts.append({"role": "user", "content": str(message.content)})

        try:
            self._wait_for_rate_limit()
            response = self.llm.invoke(messages)
            if isinstance(response, AIMessage):
                content = response.content
            else:
                content = str(response)
            
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])
            
        except Exception as e:
            error_msg = f"Error in _generate: {str(e)}"
            logging.error(error_msg)
            raise

    def _wait_for_rate_limit(self):
        """Wait if needed to respect rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    async def _async_wait_for_rate_limit(self):
        """Asynchronously wait if needed to respect rate limits."""
        current_time = time.time()
        time_since_last_request = current_time - self._last_request_time
        if time_since_last_request < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last_request
            await asyncio.sleep(sleep_time)
        self._last_request_time = time.time()

    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> AIMessage:
        """
        Asynchronously invoke the model with retry logic and improved error handling.
        
        Args:
            input: The input to send to the model
            config: Optional configuration
            **kwargs: Additional arguments
            
        Returns:
            AIMessage containing the model's response
        """
        max_retries = kwargs.get('max_retries', 3)
        base_delay = kwargs.get('base_delay', 1.0)
        last_error = None

        messages = convert_to_messages(input)

        for attempt in range(max_retries):
            try:
                await self._async_wait_for_rate_limit()
                logging.debug("Making API call to Mistral...")
                raw_response = await self.llm.ainvoke(messages)
                
                # Log the raw response type and content for debugging
                logging.debug(f"Raw response type: {type(raw_response)}")
                logging.debug(f"Raw response: {raw_response}")
                
                # If the response is already an AIMessage, return it
                if isinstance(raw_response, AIMessage):
                    return raw_response
                
                # Otherwise, convert the response to an AIMessage
                if isinstance(raw_response, str):
                    return AIMessage(content=raw_response)
                else:
                    return AIMessage(content=str(raw_response))
                
            except Exception as e:
                last_error = e
                if "rate limit exceeded" in str(e).lower():
                    delay = base_delay * (2 ** attempt)  # Exponential backoff for rate limits
                else:
                    delay = base_delay * (attempt + 1)  # Linear backoff for other errors
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
        
        # If all retries failed, raise a RuntimeError with the error message
        error_msg = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any
    ) -> AIMessage:
        """
        Invoke the model with retry logic and improved error handling.
        
        Args:
            input: The input to send to the model
            config: Optional configuration
            **kwargs: Additional arguments
            
        Returns:
            AIMessage containing the model's response
        """
        max_retries = kwargs.get('max_retries', 3)
        base_delay = kwargs.get('base_delay', 1.0)
        last_error = None

        messages = convert_to_messages(input)

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                logging.debug("Making API call to Mistral...")
                raw_response = self.llm.invoke(messages)
                
                # Log the raw response type and content for debugging
                logging.debug(f"Raw response type: {type(raw_response)}")
                logging.debug(f"Raw response: {raw_response}")
                
                # If the response is already an AIMessage, return it
                if isinstance(raw_response, AIMessage):
                    return raw_response
                
                # Otherwise, convert the response to an AIMessage
                if isinstance(raw_response, str):
                    return AIMessage(content=raw_response)
                else:
                    return AIMessage(content=str(raw_response))
                
            except Exception as e:
                last_error = e
                if "rate limit exceeded" in str(e).lower():
                    delay = base_delay * (2 ** attempt)  # Exponential backoff for rate limits
                else:
                    delay = base_delay * (attempt + 1)  # Linear backoff for other errors
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
        
        # If all retries failed, raise a RuntimeError with the error message
        error_msg = f"Failed after {max_retries} attempts. Last error: {str(last_error)}"
        logging.error(error_msg)
        raise RuntimeError(error_msg)

    def _process_string_response(self, response: str) -> Dict[str, Any]:
        """Process a string response into the required format."""
        return {
            "current_state": {
                "summary": response,
                "thought": response,
                "task_progress": "Processed string response",
                "future_plans": "Continue with next action"
            }
        }

    def _process_response(self, response) -> Dict[str, Any]:
        """Process a structured response into the required format."""
        try:
            if hasattr(response, 'content'):
                content = response.content
            else:
                content = str(response)
            
            # Try to parse as JSON if it looks like JSON
            if isinstance(content, str) and content.strip().startswith('{'):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and 'current_state' in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                
            # If not JSON or parsing failed, create a structured response
            return {
                "current_state": {
                    "summary": content,
                    "thought": content,
                    "task_progress": "Processed structured response",
                    "future_plans": "Continue with next action"
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing response: {str(e)}")
            return {
                "current_state": {
                    "summary": f"Error processing response: {str(e)}",
                    "thought": "Failed to process response",
                    "task_progress": "Error occurred",
                    "future_plans": "Retry with different parameters"
                }
            }

    def _log_response_details(self, response):
        """Log details about the response for debugging."""
        try:
            # Handle both string and object responses
            content = response.content if hasattr(response, 'content') else str(response)
            truncated_content = content[:500] + '...' if len(content) > 500 else content
            
            details = {
                'type': type(response).__name__,
                'content': truncated_content
            }
            
            if hasattr(response, 'additional_kwargs'):
                details['additional_kwargs'] = response.additional_kwargs
                
            if hasattr(response, 'response_metadata'):
                details['metadata'] = response.response_metadata
                
            logging.debug("📥 Response Details:\n%s", json.dumps(details, indent=2))
        except Exception as e:
            logging.error("Error logging request details: %s", str(e))