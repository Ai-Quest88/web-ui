#!/usr/bin/env python3
import os
import sys
import logging
import subprocess
import tempfile
import time
import json
from typing import Dict, Any, Optional, Tuple, cast, Union, List, Mapping
from langchain.schema import HumanMessage
from pydantic import SecretStr
from src.utils import utils
from playwright.async_api import Page
from playwright.sync_api import expect

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AITestAgent:
    """Agent that generates and executes Playwright tests using best practices"""
    
    def __init__(self, 
                 llm_provider: str = "mistral", 
                 llm_model_name: str = "", 
                 llm_api_key: str = "", 
                 llm_base_url: str = "",
                 # Browser settings
                 use_own_browser: bool = True,
                 keep_browser_open: bool = False,
                 headless: bool = False,
                 disable_security: bool = True,
                 window_w: int = 1920,
                 window_h: int = 1080,
                 save_recording_path: str = "recordings",
                 save_agent_history_path: str = "agent_history",
                 save_trace_path: str = "traces",
                 enable_recording: bool = True,
                 max_steps: int = 50,
                 use_vision: bool = False,
                 max_actions_per_step: int = 5,
                 tool_calling_method: str = "function_calling"):
        self.task_description = ""
        self.page_analysis = None
        self.llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name or "mistral-large-latest",  # Default model
            api_key=llm_api_key or os.getenv("MISTRAL_API_KEY", ""),  # Get from env if not provided
            base_url=llm_base_url or "",  # Empty string if not provided
            temperature=0.0
        )
        self.logs = []
        
        # Store browser settings
        self.use_own_browser = use_own_browser
        self.keep_browser_open = keep_browser_open
        self.headless = headless
        self.disable_security = disable_security
        self.window_w = window_w
        self.window_h = window_h
        self.save_recording_path = save_recording_path
        self.save_agent_history_path = save_agent_history_path
        self.save_trace_path = save_trace_path
        self.enable_recording = enable_recording
        self.max_steps = max_steps
        self.use_vision = use_vision
        self.max_actions_per_step = max_actions_per_step
        self.tool_calling_method = tool_calling_method
    
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
    
    async def analyze_page(self, url: str) -> Dict[str, Any]:
        """Analyze the page structure using LLM."""
        self.log("\n🔍 Analyzing page structure...")
        
        from playwright.async_api import async_playwright
        
        try:
            async with async_playwright() as p:
                # Configure browser with settings
                extra_args = [f"--window-size={self.window_w},{self.window_h}"]
                if self.use_own_browser:
                    chrome_path = os.getenv("CHROME_PATH", None)
                    if chrome_path == "":
                        chrome_path = None
                    chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                    if chrome_user_data:
                        extra_args += [f"--user-data-dir={chrome_user_data}"]
                else:
                    chrome_path = None

                browser = await p.chromium.launch(
                    headless=self.headless,
                    channel="chrome",
                    args=extra_args,
                    executable_path=chrome_path if self.use_own_browser else None
                )

                # Configure browser context
                context = await browser.new_context(
                    viewport={"width": self.window_w, "height": self.window_h},
                    record_video_dir=self.save_recording_path if self.enable_recording else None
                )
                page = await context.new_page()
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                
                # Get page content and metadata
                title = await page.title()
                content = await page.content()
                
                # Ask LLM to analyze the page structure
                prompt = f"""
                Analyze this HTML page structure and identify key interactive elements and their properties.
                Focus on elements that would be important for testing.
                
                Page Title: {title}
                URL: {url}
                
                HTML Content:
                {content}
                
                Please analyze and return a JSON structure with:
                1. Page metadata (title, h1 headings)
                2. Interactive elements (buttons, inputs, links, etc.)
                3. Important structural elements
                4. Best selectors to use for each element
                5. Suggested test interactions
                
                Return ONLY valid JSON without any other text.
                The JSON should follow this exact structure:
                {{
                    "title": "page title",
                    "elements": [
                        {{
                            "type": "element type",
                            "selector": "best selector",
                            "text": "element text or value",
                            "interactions": ["list", "of", "possible", "interactions"]
                        }}
                    ],
                    "structure": [
                        {{
                            "type": "structural element",
                            "selector": "selector path"
                        }}
                    ],
                    "suggested_tests": [
                        "list of suggested test cases"
                    ]
                }}
                """
                
                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                try:
                    analysis = json.loads(cast(str, response.content))
                except json.JSONDecodeError:
                    # If JSON parsing fails, try to extract JSON from the response
                    content = cast(str, response.content)
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        try:
                            analysis = json.loads(content[json_start:json_end])
                        except json.JSONDecodeError:
                            # If still fails, return a basic structure
                            analysis = {
                                "title": title,
                                "elements": [],
                                "structure": [],
                                "suggested_tests": []
                            }
                    else:
                        # If no JSON found, return basic structure
                        analysis = {
                            "title": title,
                            "elements": [],
                            "structure": [],
                            "suggested_tests": []
                        }
                
                self.page_analysis = analysis
                
                # Log summary
                self.log("\n📋 Page Analysis Summary:")
                self.log(f"Title: {analysis.get('title', 'N/A')}")
                self.log(f"Elements found: {len(analysis.get('elements', []))}")
                
                await browser.close()
                return analysis
                
        except Exception as e:
            error = f"Error analyzing page: {str(e)}"
            self.log(f"\n❌ {error}")
            # Return basic structure even on error
            return {
                "title": "Error",
                "elements": [],
                "structure": [],
                "suggested_tests": []
            }
            
    async def generate_test_code(self, error_message: Optional[str] = None) -> Union[str, Tuple[str, Mapping[str, Union[str, List[str]]]]]:
        """Generate test code using LLM with page analysis"""
        self.log("\n🎯 Starting test generation process...")
        self.log(f"📝 Task: {self.task_description}")
        
        if error_message:
            self.log(f"⚠️ Previous attempt failed: {error_message}")
        
        # Get page content for analysis
        from playwright.async_api import async_playwright
        
        try:
            async with async_playwright() as p:
                # Configure browser with settings
                extra_args = [f"--window-size={self.window_w},{self.window_h}"]
                if self.use_own_browser:
                    chrome_path = os.getenv("CHROME_PATH", None)
                    if chrome_path == "":
                        chrome_path = None
                    chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                    if chrome_user_data:
                        extra_args += [f"--user-data-dir={chrome_user_data}"]
                else:
                    chrome_path = None

                browser = await p.chromium.launch(
                    headless=self.headless,
                    channel="chrome",
                    args=extra_args,
                    executable_path=chrome_path
                )

                # Configure browser context
                context = await browser.new_context(
                    viewport={"width": self.window_w, "height": self.window_h},
                    record_video_dir=self.save_recording_path if self.enable_recording else None
                )
                page = await context.new_page()
                
                # Extract URL from task description
                import re
                url_match = re.search(r'Navigate to (https?://[^\s,]+)', self.task_description)
                if not url_match:
                    raise ValueError("No URL found in task description. Task must start with 'Navigate to <url>'")
                    
                url = url_match.group(1)
                await page.goto(url)
                await page.wait_for_load_state("networkidle")
                
                title = await page.title()
                content = await page.content()
                await browser.close()
                
                # Build comprehensive prompt with page analysis and error feedback
                prompt = f"""
                Task: Create a Playwright test that will:
                {self.task_description}

                Page Information:
                URL: {url}
                Title: {title}
                
                HTML Content for Analysis:
                {content}
                
                {"IMPORTANT - Previous test failed with error:" + error_message if error_message else ""}
                {"Please ensure the test handles these issues and uses correct selectors." if error_message else ""}

                Requirements:
                1. Analyze the HTML content and identify:
                   - Key interactive elements and their best selectors
                   - Important structural elements
                   - Best practices for element selection
                   - Potential stability issues to handle

                2. Generate a complete test using:
                   - Page Object Model pattern
                   - Reliable selector strategies (data-testid > role > text > class)
                   - Proper waiting after each action
                   - Clear assertions with stable selectors
                   - Error handling for element state

                3. Follow these patterns:
                   - Use pytest fixtures
                   - Wait for elements to be ready before interacting
                   - Validate element state after each action
                   - Handle loading and transition states
                   - Use semantic method names
                   - Add appropriate timeouts to assertions

                Return ONLY the complete Python test code without any explanations or markdown.
                The code must be immediately runnable with pytest.
                """

                response = await self.llm.ainvoke([HumanMessage(content=prompt)])
                test_code = self._extract_code_from_response(cast(str, response.content))
                
                # Log the generated code
                self.log("\n📋 Generated test code:")
                self.log("```python")
                self.log(test_code)
                self.log("```")
                
                # Run verification
                success, error_msg, execution_media = await self.execute_test(test_code)
                
                if success:
                    self.log("✅ Test passed verification!")
                    return test_code, execution_media
                else:
                    self.log("❌ Test failed verification")
                    self.log(f"Error: {error_msg}")
                    raise Exception(error_msg)
                    
        except Exception as e:
            error = f"Error generating test: {str(e)}"
            self.log(f"\n❌ {error}")
            raise Exception(error)
    
    async def execute_test(self, test_code: str) -> Tuple[bool, str, Dict[str, Union[str, List[str]]]]:
        """Execute the test code and return success status, error message, and media paths"""
        self.log("\n🚀 Starting test execution process...")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_dir = os.path.join(os.getcwd(), "test_artifacts")
        run_dir = os.path.join(base_dir, timestamp)
        screenshots_dir = os.path.join(run_dir, "screenshots")
        
        # Create directories
        for directory in [base_dir, run_dir, screenshots_dir]:
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
                # Create conftest.py with browser settings
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
        print(f"\\n📸 Screenshot saved: {{screenshot_path}}")
    except Exception as e:
        print(f"\\n⚠️ Failed to take screenshot: {{e}}")

@pytest.fixture(scope="function")
def page(browser: Browser, request) -> Generator[Page, None, None]:
    # Configure browser context with settings
    context = browser.new_context(
        viewport={{"width": {self.window_w}, "height": {self.window_h}}},
        record_video_dir="{self.save_recording_path}" if {self.enable_recording} else None
    )
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
        context.close()

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
                
                self.log("📄 Created test files with basic configuration")
                
                try:
                    # Install dependencies
                    subprocess.run(
                        ["pip", "install", "-q", "pytest", "pytest-playwright", "playwright"],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    self.log("  ✅ Packages installed")
                    
                    # Install Chrome browser if not using system Chrome
                    if not self.use_own_browser:
                        subprocess.run(
                            ["playwright", "install", "--with-deps", "chrome"],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        self.log("  ✅ Browser installed")
                    
                    # Run test with browser settings
                    self.log("\n2️⃣ Executing test...")
                    start_time = time.time()
                    
                    # Build pytest command with browser settings
                    pytest_cmd = [
                        "pytest", "-v",
                        "--headed" if not self.headless else "",
                        "--browser", "chromium",
                        "--browser-channel", "chrome",
                        test_path
                    ]
                    
                    # Set environment variables for browser settings
                    env = os.environ.copy()
                    if self.use_own_browser:
                        chrome_path = os.getenv("CHROME_PATH", None)
                        if chrome_path:
                            env["PLAYWRIGHT_CHROMIUM_PATH"] = chrome_path
                    
                    result = subprocess.run(
                        [cmd for cmd in pytest_cmd if cmd],  # Remove empty strings
                        capture_output=True,
                        text=True,
                        timeout=120,
                        cwd=temp_dir,
                        env=env
                    )
                    
                    duration = time.time() - start_time
                    
                    # Collect screenshots
                    if os.path.exists(screenshots_dir):
                        screenshots = [
                            os.path.join(screenshots_dir, f) 
                            for f in os.listdir(screenshots_dir) 
                            if f.endswith('.png')
                        ]
                        media_paths["screenshots"] = screenshots
                    
                    # Log captured screenshots
                    if media_paths["screenshots"]:
                        self.log("\n📸 Screenshots captured:")
                        for screenshot in media_paths["screenshots"]:
                            self.log(f"  - {screenshot}")
                    
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
    
    async def run_with_retry(self, max_attempts: int = 3) -> Tuple[bool, str, Mapping[str, Union[str, List[str]]]]:
        """Generate and run test with retry logic"""
        error_message: Optional[str] = None
        media_paths: Dict[str, Union[str, List[str]]] = {
            "screenshots": [],
            "videos": [],
            "generation_video": "",
            "execution_video": ""
        }
        test_code = ""
        
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
                    return True, test_code, media_paths
                
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
                    return False, test_code, media_paths
                    
            except Exception as e:
                self.log(f"❌ Error during attempt {attempt + 1}: {str(e)}")
                if attempt < max_attempts - 1:
                    error_message = str(e)
                    continue
                else:
                    self.log("❌ All retry attempts exhausted")
                    return False, test_code, media_paths
        
        return False, test_code, media_paths
    
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
    
    success, test_code, media_paths = await agent.run_with_retry(max_attempts=3)
    
    if success:
        logger.info("Test generated and executed successfully!")
        return 0
    else:
        logger.error("Failed to generate and execute a passing test")
        return 1


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())