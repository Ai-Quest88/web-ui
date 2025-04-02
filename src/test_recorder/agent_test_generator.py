#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
import tempfile
import time
from typing import Dict, Any, Optional, Tuple, cast, Union, List, Mapping
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
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY environment variable is required")

class AITestAgent:
    """Agent that generates and executes Playwright tests using best practices"""
    
    def __init__(self):
        self.task_description = ""
        self.llm = ChatMistralAI(
            model_name="mistral-large-latest",
            temperature=0.0,
            api_key=SecretStr(MISTRAL_API_KEY)
        )
        self.logs = []
    
    def log(self, message: str):
        """Add a log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        log_msg = f"[{timestamp}] {message}"
        print(log_msg, flush=True)  # Print directly to console
        logger.info(log_msg)
        self.logs.append(log_msg)
        return "\n".join(self.logs)
    
    def set_task(self, task_description: str):
        """Set the task to be tested"""
        self.task_description = task_description
        return self.log(f"Task set: {task_description}")
    
    async def verify_test_code(self, test_code: str) -> Tuple[bool, str, Mapping[str, Union[str, List[str]]]]:
        """Verify the test code through static and dynamic checks"""
        self.log("\n🔍 Verifying generated test code...")
        
        # First do static verification
        verification_steps = [
            ("Imports check", ["import pytest", "from playwright"]),
            ("Page Object Model", ["class", "def __init__"]),
            ("Test class", ["class Test", "def test_"]),
            ("Browser config", ["launch(headless=False", "viewport", "context.new_page"]),
            ("Assertions", ["assert", "expect"]),
            ("Error handling", ["try:", "except", "finally"]),
            ("Waiting strategies", ["wait_for", "expect", "timeout"])
        ]
        
        issues = []
        for step_name, required_items in verification_steps:
            self.log(f"  ⚡ Checking {step_name}...")
            missing = [item for item in required_items if item not in test_code]
            if missing:
                issues.append(f"Missing {step_name}: {', '.join(missing)}")
            else:
                self.log(f"  ✅ {step_name} verified")
        
        if issues:
            self.log("\n❌ Static verification failed:")
            for issue in issues:
                self.log(f"  - {issue}")
            return False, "\n".join(issues), {}
            
        self.log("\n✅ Static verification passed!")
        
        # Now do dynamic verification by actually running the test
        self.log("\n🚀 Running test for dynamic verification...")
        success, error_msg, media_paths = await self.execute_test(test_code)
        
        if success:
            self.log("✅ Dynamic verification passed - test runs successfully!")
            return True, "Test code meets all requirements and runs successfully", media_paths
        else:
            self.log("❌ Dynamic verification failed - test execution error")
            self.log(f"Error: {error_msg}")
            return False, error_msg, media_paths
    
    async def generate_test_code(self, error_message: Optional[str] = None) -> Union[str, Tuple[str, Mapping[str, Union[str, List[str]]]]]:
        """Generate test code using LLM"""
        self.log("\n🎯 Starting test generation process...")
        self.log(f"📝 Task: {self.task_description}")
        
        media_paths: Dict[str, Union[str, List[str]]] = {
            "screenshots": [],
            "videos": [],
            "generation_video": "",
            "execution_video": ""
        }
        
        if error_message:
            self.log("\n🔄 Previous error detected - adjusting generation strategy")
            self.log(f"Previous error: {error_message}")
        
        self.log("\n1️⃣ Generating test code...")
        page_context = """
        Generate a robust Playwright test following these patterns and requirements:

        1. Page Object Model Structure:
           - Create a Page class that encapsulates all page interactions
           - Use meaningful method names that describe the action being performed
           - Each method should handle one specific action
           - Include proper validation after each action
           - Use strong typing with Page parameter

        2. Locator Patterns:
           - Use data-testid attributes when available
           - Use role-based selectors (getByRole) when appropriate
           - Use text-based selectors as fallback
           - Use class selectors only when necessary
           - Use meaningful locator names

        3. Action Patterns:
           - Before any action: expect(element).to_be_visible(timeout=5000)
           - After input: expect(element).to_have_value(expected_value, timeout=5000)
           - After state changes: expect(element).to_have_class(expected_class, timeout=5000)
           - After count changes: expect(count_element).to_contain_text(str(expected_count), timeout=5000)
           - After any change: expect(element).to_be_visible(timeout=5000)

        4. Test Structure:
           - Use pytest fixtures for setup
           - Implement proper error handling with try/except
           - Take screenshots on failure
           - Use meaningful test names
           - Group related actions together
           - Add comments explaining test flow

        5. Required Validations:
           - Verify element state after interactions
           - Verify text content after changes
           - Verify element attributes after updates
           - Verify counts after list changes
           - Verify form field states after submission

        6. Error Handling:
           - Catch and log exceptions
           - Take screenshots on failure
           - Use proper cleanup in finally blocks
           - Add timeouts to all expect calls
           - Handle page load states

        The test must implement all these patterns while completing the task.
        Focus on reliability and proper validation of each step.
        """
        
        max_attempts = 3
        for attempt in range(max_attempts):
            self.log(f"\n📝 Generating test code (attempt {attempt + 1}/{max_attempts})...")
            response = await self.llm.ainvoke([HumanMessage(content=f"""
            Task: Create a Playwright test that will:
            {self.task_description}

            Requirements and Patterns:
            {page_context}

            Previous error (if any): {error_message if error_message else 'None'}

            Important Notes:
            1. Return ONLY the complete Python test code without any explanations or markdown
            2. The code must be immediately runnable with pytest
            3. Include ALL necessary imports
            4. Follow ALL the patterns exactly as specified
            5. Implement proper validation after EVERY action
            6. Use the EXACT selectors provided in the patterns
            7. Add appropriate timeouts to ALL expect calls
            8. Handle ALL possible errors
            9. Take screenshots at key points
            10. Use meaningful names and comments

            Generate the complete test code now.
            """)])
            
            test_code = self._extract_code_from_response(cast(str, response.content))
            
            # Log the generated code
            self.log("\n📋 Generated test code:")
            self.log("```python")
            self.log(test_code)
            self.log("```")
            
            # Run dynamic verification
            self.log("\n🚀 Running test for verification...")
            success, error_msg, execution_media = await self.execute_test(test_code)
            
            # Merge media paths
            media_paths = {
                **media_paths,
                **execution_media
            }
            
            if success:
                self.log("✅ Test passed verification!")
                return test_code, media_paths
            else:
                self.log("❌ Test failed verification")
                self.log(f"Error: {error_msg}")
                if attempt < max_attempts - 1:
                    error_message = error_msg
                    continue
                else:
                    self.log("\n💥 Failed to generate working test after max attempts")
                    raise Exception(f"Failed to generate working test after {max_attempts} attempts. Last error: {error_msg}")
        
        raise Exception("Failed to generate valid test code after max attempts")
    
    async def execute_test(self, test_code: str) -> Tuple[bool, str, Dict[str, Union[str, List[str]]]]:
        """Execute the test code and return success status, error message, and media paths"""
        self.log("\n🚀 Starting test execution process...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.getcwd(), "test_artifacts")
        run_dir = os.path.join(base_dir, timestamp)
        screenshots_dir = os.path.join(run_dir, "screenshots")
        videos_dir = os.path.join(run_dir, "videos")
        
        # Create all directories
        for directory in [base_dir, run_dir, screenshots_dir, videos_dir]:
            os.makedirs(directory, exist_ok=True)
            
        media_paths: Dict[str, Union[str, List[str]]] = {
            "screenshots": [],
            "videos": [],
            "generation_video": "",
            "execution_video": ""
        }
        
        try:
            # Create temporary test directory for pytest
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create conftest.py with video recording configuration
                conftest_path = os.path.join(temp_dir, "conftest.py")
                with open(conftest_path, "w") as f:
                    f.write(f"""
import pytest
import os
import time
from typing import Generator
from playwright.sync_api import Page, Browser, BrowserContext, expect

def take_screenshot(page: Page, name: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    screenshots_dir = "{screenshots_dir}"
    os.makedirs(screenshots_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshots_dir, f"{{name}}_{{timestamp}}.png")
    try:
        page.screenshot(path=screenshot_path)
        print(f"\\n📸 Screenshot saved to: {{screenshot_path}}")
    except Exception as e:
        print(f"\\n⚠️ Failed to take screenshot: {{e}}")

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {{
        **browser_context_args,
        "viewport": {{"width": 1280, "height": 720}},
        "record_video_dir": "{videos_dir}"
    }}

@pytest.fixture(scope="function")
def context(browser: Browser) -> Generator[BrowserContext, None, None]:
    context = browser.new_context(
        record_video_size={{"width": 1280, "height": 720}},
        viewport={{"width": 1280, "height": 720}}
    )
    yield context
    context.close()

@pytest.fixture(scope="function")
def page(context: BrowserContext, request) -> Generator[Page, None, None]:
    page = context.new_page()
    test_name = request.node.name
    
    # Take screenshot at start
    take_screenshot(page, f"{{test_name}}_start")
    
    yield page
    
    try:
        if page:
            # Take screenshot at end
            take_screenshot(page, f"{{test_name}}_end")
            
            # Take screenshot based on test result
            if request.node.rep_call.failed:
                take_screenshot(page, f"{{test_name}}_failure")
            else:
                take_screenshot(page, f"{{test_name}}_success")
    finally:
        page.close()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, "rep_" + rep.when, rep)
""")
                
                # Create test file
                test_path = os.path.join(temp_dir, "test_generated.py")
                enhanced_test_code = test_code.replace(
                    "def test_", 
                    "def test_auto_"  # Ensure unique test name with generic prefix
                )
                with open(test_path, "w") as f:
                    f.write(enhanced_test_code)
                
                self.log("📄 Created test files with enhanced recording configuration")
                
                try:
                    # Install dependencies
                    subprocess.run(
                        ["pip", "install", "-q", "pytest", "pytest-playwright", "playwright"],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    self.log("  ✅ Packages installed")
                    
                    subprocess.run(
                        ["playwright", "install", "--with-deps", "chromium"],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    self.log("  ✅ Browser installed")
                    
                    # Run test with video recording
                    self.log("\n2️⃣ Executing test with video recording...")
                    start_time = time.time()
                    
                    result = subprocess.run(
                        [
                            "pytest", "-v",
                            "--headed",  # Run in headed mode
                            "--browser", "chromium",
                            "--video=on",  # Enable video recording
                            "--screenshot=on",  # Enable screenshots
                            "--tracing=on",  # Enable tracing
                            test_path
                        ],
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=temp_dir
                    )
                    
                    duration = time.time() - start_time
                    
                    # Collect media files
                    if os.path.exists(screenshots_dir):
                        screenshots = [
                            os.path.join(screenshots_dir, f) 
                            for f in os.listdir(screenshots_dir) 
                            if f.endswith('.png')
                        ]
                        media_paths["screenshots"] = screenshots
                        
                    if os.path.exists(videos_dir):
                        videos = [
                            os.path.join(videos_dir, f)
                            for f in os.listdir(videos_dir)
                            if f.endswith('.webm')
                        ]
                        media_paths["videos"] = videos
                        if videos:
                            media_paths["execution_video"] = videos[-1]  # Latest video
                    
                    # Log captured media
                    if media_paths["screenshots"]:
                        self.log("\n📸 Screenshots captured:")
                        for screenshot in media_paths["screenshots"]:
                            self.log(f"  - {screenshot}")
                    
                    if media_paths["videos"]:
                        self.log("\n🎥 Videos captured:")
                        for video in media_paths["videos"]:
                            self.log(f"  - {video}")
                    
                    if result.returncode == 0:
                        self.log(f"\n✅ Test passed! ({duration:.1f}s)")
                        return True, "Test passed successfully", media_paths
                    else:
                        error_output = result.stdout + "\n" + result.stderr
                        self.log(f"\n❌ Test failed! ({duration:.1f}s)")
                        self.log(f"Error output:\n{error_output}")
                        return False, error_output, media_paths
                        
                except subprocess.TimeoutExpired:
                    self.log("\n⏰ Test execution timed out after 120s")
                    return False, "Test execution timed out", media_paths
                except Exception as e:
                    self.log(f"\n💥 Execution error: {str(e)}")
                    return False, str(e), media_paths
                
        except Exception as e:
            error_msg = f"Setup error: {str(e)}"
            self.log(f"\n💥 {error_msg}")
            return False, error_msg, media_paths
            
    def get_media_paths(self) -> Dict[str, str]:
        """Get paths to the latest media files"""
        base_dir = os.path.join(os.getcwd(), "test_artifacts")
        
        # Get latest timestamp directory
        screenshots_dir = os.path.join(base_dir, "screenshots")
        videos_dir = os.path.join(base_dir, "videos")
        
        latest_media = {
            "screenshots": [],
            "videos": [],
            "generation_video": "",
            "execution_video": ""
        }
        
        if os.path.exists(screenshots_dir):
            timestamps = sorted(os.listdir(screenshots_dir), reverse=True)
            if timestamps:
                latest_dir = os.path.join(screenshots_dir, timestamps[0])
                latest_media["screenshots"] = [
                    os.path.join(latest_dir, f) 
                    for f in os.listdir(latest_dir) 
                    if f.endswith('.png')
                ]
        
        if os.path.exists(videos_dir):
            timestamps = sorted(os.listdir(videos_dir), reverse=True)
            if timestamps:
                latest_dir = os.path.join(videos_dir, timestamps[0])
                videos = [
                    os.path.join(latest_dir, f)
                    for f in os.listdir(latest_dir)
                    if f.endswith('.webm')
                ]
                latest_media["videos"] = videos
                if videos:
                    latest_media["execution_video"] = videos[-1]
        
        return latest_media
    
    async def analyze_error(self, test_code: str, error_message: str) -> str:
        """Analyze test failure and suggest improvements"""
        self.log("\n🔍 Starting error analysis...")
        
        prompt = f"""
        The following Playwright test failed with this error:
        {error_message}
        
        Test code:
        ```python
        {test_code}
        ```
        
        Please analyze the error and provide:
        1. Root cause of the failure
        2. Specific code issues that need to be fixed
        3. Recommended solutions
        4. Best practices to prevent similar issues
        
        Focus on:
        - Element visibility and timing issues
        - Selector reliability
        - Page load states
        - Network conditions
        - Error handling
        """
        
        self.log("🤖 Analyzing error with LLM...")
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        analysis = cast(str, response.content)
        
        self.log("\n📋 Analysis complete")
        return analysis
    
    async def run_with_retry(self, max_attempts: int = 3) -> Tuple[bool, Mapping[str, Union[str, List[str]]]]:
        """Generate and run test with retry logic"""
        error_message: Optional[str] = None
        media_paths: Dict[str, Union[str, List[str]]] = {
            "screenshots": [],
            "videos": [],
            "generation_video": "",
            "execution_video": ""
        }
        
        for attempt in range(max_attempts):
            self.log(f"\n🔄 Test attempt {attempt + 1}/{max_attempts}")
            
            try:
                # Generate test code
                result = await self.generate_test_code(error_message)
                if isinstance(result, tuple):
                    test_code, gen_media_paths = result
                    media_paths.update(gen_media_paths)
                else:
                    test_code = result
                
                self.log(f"📝 Generated test code:\n{test_code}\n")
                
                # Execute test
                success, error_msg, execution_media = await self.execute_test(test_code)
                if success:
                    self.log("🎉 Test passed successfully!")
                    # Merge media paths from generation and execution
                    media_paths.update(execution_media)
                    return True, media_paths
                
                # If test failed and we have more attempts
                if attempt < max_attempts - 1:
                    self.log(f"❌ Test failed, analyzing error...")
                    if error_msg:
                        error_message = await self.analyze_error(test_code, error_msg)
                    else:
                        self.log("⚠️ No specific error message available")
                        error_message = "Test failed without a specific error message. Will try to improve test reliability."
                else:
                    self.log("❌ All retry attempts exhausted")
                    return False, media_paths
                    
            except Exception as e:
                self.log(f"❌ Error during attempt {attempt + 1}: {str(e)}")
                if attempt < max_attempts - 1:
                    error_message = str(e)
                    continue
                else:
                    self.log("❌ All retry attempts exhausted")
                    return False, media_paths
        
        return False, media_paths
    
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
    
    success, media_paths = await agent.run_with_retry(max_attempts=3)
    
    if success:
        logger.info("Test generated and executed successfully!")
        return 0
    else:
        logger.error("Failed to generate and execute a passing test")
        return 1


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())