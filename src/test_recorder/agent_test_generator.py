#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional, Tuple, cast
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage
from pydantic import SecretStr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mistral API key from environment
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "3LwHm6LjdjaooZwnEAHPjtpaJfjIHesg")

class AITestAgent:
    """Agent that generates and executes Playwright tests using best practices"""
    
    def __init__(self):
        self.task_description = ""
        self.llm = ChatMistralAI(
            model_name="mistral-large-latest",
            temperature=0.0,
            api_key=SecretStr(MISTRAL_API_KEY)
        )
    
    def set_task(self, task_description: str):
        """Set the task to be tested"""
        self.task_description = task_description
        logger.info(f"Task set: {task_description}")
    
    async def generate_test_code(self, error_message: Optional[str] = None) -> str:
        """Generate Playwright test code directly based on task description"""
        logger.info("Generating Playwright test code...")
        
        page_context = """
        Best Practices for Web UI Testing:
        1. Locator Priority (from most to least stable):
           - test-id/data-testid attributes (e.g. [data-testid="submit-button"])
           - ARIA roles and labels (e.g. getByRole('button', { name: 'Submit' }))
           - Semantic HTML (e.g. getByRole('navigation'), getByRole('main'))
           - Form labels and text content (e.g. getByLabel('Username'), getByText('Submit'))
           - IDs and unique attributes (only if they are stable)
           - CSS selectors (last resort, as they are fragile)

        2. Important Waiting Strategies:
           - Wait for elements to be visible before interaction
           - Wait for network requests to complete after actions
           - Wait for dynamic content to load
           - Handle loading states and transitions
           - Consider animation completion

        3. Common UI Patterns:
           - Navigation menus and breadcrumbs
           - Search functionality (input, suggestions, results)
           - Forms and validation
           - Modals and overlays
           - Infinite scroll and pagination
           - Dynamic content loading
           - Error states and messages
        """
        
        retry_context = ""
        if error_message:
            retry_context = f"""
            The previous test failed with the following error:
            {error_message}
            
            Please fix the test code to address this error.
            """
        
        prompt = f"""
        Generate a Playwright test in Python that accomplishes this task:
        "{self.task_description}"
        
        Requirements:
        1. Use the Page Object Model (POM) pattern for maintainability
        2. Include proper assertions to verify the task was completed
        3. Include proper error handling and waiting strategies
        4. Include all necessary imports
        5. The test should be standalone and runnable with Playwright
        6. Follow these best practices for web testing:
        {page_context}
        {retry_context}
        
        Return format (ONLY Python code, no documentation or markdown):
        # Page Object Model
        [Python code for page object]

        # Test File
        [Python code for test]
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        code = cast(str, response.content)
        
        # Extract code from response
        return self._extract_code_from_response(code)
    
    async def execute_test(self, test_code: str) -> Tuple[bool, Optional[str]]:
        """Execute the generated test code"""
        logger.info("Executing generated Playwright test...")
        
        # Create a temporary file for the test code
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(test_code.encode('utf-8'))
        
        try:
            # Ensure dependencies are installed
            logger.info("Ensuring dependencies are installed...")
            try:
                subprocess.run(
                    ["pip", "install", "playwright"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                subprocess.run(
                    ["playwright", "install", "chromium"],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError:
                logger.warning("Dependency installation may have issues, but continuing...")
            
            # Execute the test
            logger.info(f"Running test from {temp_filename}...")
            result = subprocess.run(
                ["python", temp_filename],
                capture_output=True,
                text=True,
                timeout=120  # Increase timeout for more complex tasks
            )
            
            # Check if the test passed
            if result.returncode == 0:
                logger.info("Test executed successfully!")
                return True, None
            else:
                error_message = result.stderr or result.stdout
                logger.error(f"Test failed with error: {error_message}")
                return False, error_message
                
        except Exception as e:
            logger.error(f"Exception during test execution: {e}")
            return False, str(e)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
    
    async def analyze_error(self, test_code: str, error_message: str) -> str:
        """Have the agent analyze the error and suggest improvements"""
        logger.info("Agent analyzing test failure...")
        
        prompt = f"""
        The following Playwright test failed with this error:
        {error_message}
        
        Here is the test code:
        ```python
        {test_code}
        ```
        
        Please analyze the error and explain what went wrong and how to fix it.
        Focus on:
        1. Element visibility and timing issues
        2. Selector reliability and uniqueness
        3. Page load and network states
        4. Dynamic content loading
        5. Error handling improvements
        
        Be specific about what needs to change in the code.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        analysis = cast(str, response.content)
        logger.info(f"Analysis: {analysis}")
        return analysis
    
    async def run_with_retry(self, max_attempts: int = 3) -> bool:
        """Generate and run test with retry logic"""
        error_message: Optional[str] = None
        test_code: Optional[str] = None
        
        for attempt in range(max_attempts):
            logger.info(f"Test attempt {attempt + 1}/{max_attempts}")
            
            # Generate test code
            test_code = await self.generate_test_code(error_message)
            logger.info(f"Generated test code:\n{test_code}\n")
            
            # Execute the test
            success, new_error_message = await self.execute_test(test_code)
            
            if success:
                logger.info(f"Test passed on attempt {attempt + 1}")
                return True
            
            error_message = new_error_message
            
            if attempt < max_attempts - 1:
                logger.info(f"Retrying... ({attempt + 2}/{max_attempts})")
                # Let the agent analyze the error to inform the next generation
                if error_message and test_code:
                    analysis = await self.analyze_error(test_code, error_message)
                    error_message = f"{error_message}\n\nAnalysis: {analysis}"
                time.sleep(1)  # Small delay before retry
            else:
                logger.error(f"All {max_attempts} attempts failed")
        
        return False
    
    def _extract_code_from_response(self, content: str) -> str:
        """Extract code from API response"""
        try:
            content = content.strip()
            
            # If the response contains markdown code blocks, extract the code
            if "```python" in content:
                code_blocks = content.split("```python")
                code = code_blocks[1].split("```")[0].strip()
                return code
            elif "```" in content:
                code_blocks = content.split("```")
                if len(code_blocks) > 1:
                    return code_blocks[1].strip()
            
            # Split by "# Page Object Model" and "# Test File" if present
            if "# Page Object Model" in content and "# Test File" in content:
                parts = content.split("# Page Object Model", 1)
                if len(parts) == 2:
                    page_model = parts[1].split("# Test File", 1)
                    if len(page_model) == 2:
                        page_code = page_model[0].strip()
                        test_code = page_model[1].strip()
                        return f"# Page Object Model\n{page_code}\n\n# Test File\n{test_code}"
            
            # If no code blocks or sections, return the whole content
            return content
            
        except Exception as e:
            logger.error(f"Failed to extract code from response: {e}")
            logger.error(f"Response: {content}")
            raise Exception("Failed to extract code from API response")


async def main():
    """Main function to run the AI test agent"""
    task_description = """Navigate to https://demo.playwright.dev/todomvc/, then:
    1. Add three todo items: 'Buy groceries', 'Clean house', 'Walk dog'
    2. Mark 'Clean house' as completed
    3. Verify the remaining active items count is 2
    4. Verify 'Clean house' is shown as completed"""
    
    logger.info(f"Starting AI test agent with task: {task_description}")
    
    agent = AITestAgent()
    agent.set_task(task_description)
    
    success = await agent.run_with_retry(max_attempts=3)
    
    if success:
        logger.info("Test generated and executed successfully!")
        return 0
    else:
        logger.error("Failed to generate and execute a passing test")
        return 1


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 