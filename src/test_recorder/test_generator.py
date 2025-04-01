import json
import os
import logging
import re
import subprocess
from pathlib import Path
from typing import cast, Optional, Dict, List, Tuple, Any
from urllib.parse import urlparse
from langchain_mistralai import ChatMistralAI
from langchain.schema import HumanMessage
from pydantic import SecretStr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RECORDED_TESTS_DIR = Path(__file__).parent / 'recorded_tests'
PAGES_DIR = RECORDED_TESTS_DIR / 'pages'
TEST_ARTIFACTS_DIR = RECORDED_TESTS_DIR / 'test_artifacts'

# Ensure directories exist
RECORDED_TESTS_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
TEST_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
(RECORDED_TESTS_DIR / '__init__.py').touch()
(PAGES_DIR / '__init__.py').touch()

def get_page_name(url: str) -> str:
    """Extract page name from URL."""
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p]
    if not path_parts:
        return 'home'
    return path_parts[-1].replace('-', '_')

def get_class_name(page_name: str) -> str:
    """Convert page name to class name."""
    return ''.join(word.capitalize() for word in page_name.split('_')) + 'Page'


class CodeHelper:
    """Helper class for code manipulation and analysis."""
    
    @staticmethod
    def strip_markdown(code: str) -> str:
        """Strip markdown formatting from code."""
        # Remove code block markers
        code = re.sub(r'```(?:python)?\n', '', code)
        code = code.replace('```', '')
        
        # Remove any leading/trailing whitespace
        code = code.strip()
        
        # If code starts with Python filename comment, remove it
        code = re.sub(r'^#\s*\w+\.py\s*\n', '', code)
        
        return code
        
    @staticmethod
    def log_code_differences(original_code: str, new_code: str) -> None:
        """Log the key differences between original and new code."""
        original_lines = original_code.split('\n')
        new_lines = new_code.split('\n')
        
        # Simple diff to highlight added/modified lines
        added_or_modified = []
        for i, line in enumerate(new_lines):
            if i >= len(original_lines) or line != original_lines[i]:
                added_or_modified.append((i, line))
        
        # Log a reasonable number of changes
        for i, (line_num, line) in enumerate(added_or_modified[:10]):
            logger.info(f"  Line {line_num+1}: {line[:100]}")
        
        if len(added_or_modified) > 10:
            logger.info(f"  ... plus {len(added_or_modified) - 10} more changes")


class DiagnosticRunner:
    """Class for running diagnostic tests and capturing page state."""
    
    @staticmethod
    def analyze_error(error_output: str) -> Dict[str, Optional[str]]:
        """Extract key information from test error output."""
        error_info: Dict[str, Optional[str]] = {
            'type': None,
            'element': None,
            'details': None,
            'state': None
        }
        
        if 'TimeoutError' in error_output:
            error_info['type'] = 'timeout'
            match = re.search(r'waiting for (.+?) to', error_output)
            if match:
                error_info['element'] = match.group(1)
                state_match = re.search(r'to be (\w+)', error_output)
                if state_match:
                    error_info['state'] = state_match.group(1)
            
            if 'locator resolved to' in error_output:
                state_match = re.search(r'locator resolved to (\w+)', error_output)
                if state_match and not error_info['state']:
                    error_info['state'] = state_match.group(1)
                    
            if 'Call log:' in error_output:
                error_info['details'] = error_output.split('Call log:')[1].strip()
                
        elif 'Error: Element is not' in error_output:
            error_info['type'] = 'invalid_element'
            match = re.search(r'Error: Element is not (.+)', error_output)
            if match:
                error_info['details'] = match.group(1)
        
        return error_info
    
    @staticmethod
    def capture_page_state(test_file_path: str, page_name: str, error_info: Dict[str, Optional[str]]) -> Tuple[str, str, str]:
        """Run a diagnostic test to capture page state at failure point."""
        screenshot_path = str(TEST_ARTIFACTS_DIR / f"{page_name}_error_screenshot.png")
        html_path = str(TEST_ARTIFACTS_DIR / f"{page_name}_error_dom.html")
        
        # Read original test
        with open(test_file_path, "r") as f:
            original_test = f.read()
            
        # Add diagnostic code to capture state at failure
        diagnostic_code = original_test.replace(
            "def test_", 
            """import os
import time
from pathlib import Path

def _save_failure_artifacts(page, name):
    # Capture page state for test failure analysis
    artifacts_dir = Path(__file__).parent / 'test_artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(artifacts_dir / f"{name}_error_screenshot.png"))
    with open(str(artifacts_dir / f"{name}_error_dom.html"), "w", encoding="utf-8") as f:
        f.write(page.content())
    print(f"Captured failure artifacts to test_artifacts/{name}_*")

def test_"""
        )
        
        # Add diagnostic calls at key points
        if error_info['element']:
            # Try to add diagnostics right before the failing action
            action_pattern = rf"(\s+)({page_name}_page\..*{error_info['element'].split('(')[0]}.*\(.*\))"
            if re.search(action_pattern, diagnostic_code):
                diagnostic_code = re.sub(
                    action_pattern,
                    r'\1_save_failure_artifacts(page, "' + page_name + r'")\n\1\2',
                    diagnostic_code
                )
        
        # Write the instrumented test
        with open(test_file_path, "w") as f:
            f.write(diagnostic_code)
            
        try:
            # Run the test with diagnostics enabled
            diagnostic_command = [
                'python', '-m', 'pytest', 
                test_file_path, 
                '--headed',
                '-v',
                '--no-summary',
                f'--screenshot=on',
                f'--browser=chromium'
            ]
            subprocess.run(diagnostic_command, capture_output=True, text=True)
            
        finally:
            # Always restore original test
            with open(test_file_path, "w") as f:
                f.write(original_test)
        
        # Check if screenshot was captured
        screenshot_context = ""
        if os.path.exists(screenshot_path):
            screenshot_context = f"A screenshot was captured at the failure point and saved to: {screenshot_path}"
            
        # Extract DOM context
        dom_context = "DOM analysis not available"
        if os.path.exists(html_path):
            try:
                dom_context = DiagnosticRunner.extract_dom_context(html_path, error_info)
            except Exception as e:
                logger.error(f"Error analyzing DOM: {str(e)}")
                dom_context = f"DOM analysis failed: {str(e)}"
                
        return screenshot_path, html_path, dom_context
    
    @staticmethod
    def extract_dom_context(html_path: str, error_info: Dict[str, Optional[str]]) -> str:
        """Extract relevant sections of the DOM based on error information."""
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        if not error_info['element']:
            return f"Portion of DOM (no element info):\n{html_content[:1000]}"
            
        # Find most relevant section based on error text
        search_text = str(error_info['element']).replace('locator(', '').replace(')', '')
        search_text = re.sub(r'["\'()]', '', search_text)
        
        # Look for elements matching the error description
        dom_sections = []
        for term in search_text.split():
            if len(term) > 3:  # Skip very short terms
                pattern = rf'<[^>]*{term}[^>]*>.*?</.*?>'
                matches = re.findall(pattern, html_content, re.DOTALL)
                dom_sections.extend(matches[:2])  # Get up to 2 matches per term
                
        if dom_sections:
            return "Relevant DOM fragments:\n" + "\n---\n".join(
                [s[:500] for s in dom_sections[:3]]  # Limit to 3 total sections
            )
        else:
            # Fallback: just get a portion of the DOM
            return f"Portion of DOM (page state at error point):\n{html_content[:2000]}"


class TestGenerator:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.test_file_path = RECORDED_TESTS_DIR / f"test_{test_name}.py"
        self.page_file_path = RECORDED_TESTS_DIR / "pages" / f"{test_name}_page.py"
        self.init_file_path = RECORDED_TESTS_DIR / "pages" / "__init__.py"
        
        # Create directories if they don't exist
        self.test_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.page_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.llm = ChatMistralAI(
            model_name="mistral-large-latest",
            temperature=0.0,
            api_key=SecretStr("3LwHm6LjdjaooZwnEAHPjtpaJfjIHesg")
        )
        self.code_helper = CodeHelper()

    async def generate_and_verify_test(self, actions: List[Dict[str, Any]], url: str, page_name: str, 
                                     class_name: str, page_file: str, test_file: str) -> Tuple[bool, str, str, str]:
        """Generate test files and verify them."""
        prompt = f"""Generate a Page Object Model and Playwright test for the following recorded actions. Return ONLY valid Python code without any documentation or markdown.

Actions:
{json.dumps(actions, indent=2)}

Requirements:
1. Page Object ({page_file}):
- Class name: {class_name}
- Import Page and expect from playwright.sync_api
- Use semantic locators based on element roles and attributes:
  * Buttons: get_by_role("button") with name from aria-label or text content
  * Links: get_by_role("link") with name from text or href
  * Inputs: get_by_role("textbox") or get_by_placeholder() for search inputs
  * Comboboxes: get_by_role("combobox") with name from label or aria-label
  * Options: get_by_role("option") with name from text content
  * Search fields: get_by_placeholder() or data-target attributes for modern web apps
  * Prefer data-testid, data-target or other unique attributes when available
- Define locators in __init__ using attributes from interacted_element
- Add methods for each recorded action:
  * go_to() -> None: Navigate to initial URL
  * Methods for each click_element action
  * Methods for each input_text action
  * Verification methods based on done actions
- Handle dynamic elements:
  * After clicking elements that trigger popups/modals:
    - Wait for dialog/modal to be attached: dialog.wait_for(state="attached")
    - Then wait for dialog to be visible: dialog.wait_for(state="visible")
    - Finally wait for interactive elements inside dialog
  * After clicking elements that load content:
    - Wait for new elements to be visible
    - Use appropriate role-based selectors with wait_for(state="visible")
  * After form submissions:
    - Wait for success/error messages
    - Wait for URL changes if redirecting
- Use type hints
- Return None for action methods

2. Test ({test_file}):
- Import pytest, Page and expect from playwright.sync_api
- Import {class_name} using relative import (.pages.{page_name}_page)
- Add pytest fixture named {page_name}_page that returns {class_name} instance
- Test function should be named test_{page_name}
- Test should:
  * Use both page and {page_name}_page fixtures
  * Call methods in the same order as recorded actions
  * Include verification steps from done actions
- Follow AAA pattern with comments
- Use semantic expect assertions based on the interactions

Return format (ONLY Python code, no documentation or markdown):
# Page Object Model
[Python code for page object with proper imports]

# Test File
[Python code for test with proper imports]"""

        logger.info("Sending request to LLM for test generation...")
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        code = cast(str, response.content)
        
        # Strip markdown formatting
        code = self.code_helper.strip_markdown(code)

        if "# Page Object Model" not in code or "# Test File" not in code:
            logger.error("Invalid code format received from LLM")
            return False, "", "", "Invalid code format from LLM"

        parts = code.split("# Page Object Model")
        if len(parts) != 2:
            logger.error("Invalid code structure received from LLM")
            return False, "", "", "Invalid code format"

        page_code = self.code_helper.strip_markdown(parts[1].split("# Test File")[0].strip())
        test_code = self.code_helper.strip_markdown(parts[1].split("# Test File")[1].strip())

        # Log generated code (first few lines)
        logger.info(f"Generated Page Object Model ({page_file}) - first 10 lines")
        line_count = len(page_code.split("\n")) - 10
        if line_count > 0:
            logger.info(f"  ... plus {line_count} more lines")
            
        logger.info(f"Generated Test File ({test_file}) - first 10 lines")
        line_count = len(test_code.split("\n")) - 10
        if line_count > 0:
            logger.info(f"  ... plus {line_count} more lines")

        # Save and verify
        page_file_path = os.path.join(PAGES_DIR, page_file)
        test_file_path = os.path.join(RECORDED_TESTS_DIR, test_file)

        with open(page_file_path, "w") as f:
            f.write(page_code)
        with open(test_file_path, "w") as f:
            f.write(test_code)

        # Execute the test
        result = subprocess.run(['pytest', test_file_path, '-v'], 
                             capture_output=True, 
                             text=True)
        
        success = result.returncode == 0
        if success:
            logger.info("Test execution passed! ✅")
        else:
            logger.info("Test execution failed! ❌")
            
        return success, page_code, test_code, result.stdout + "\n" + result.stderr

    async def fix_test(self, page_code: str, test_code: str, error_output: str, 
                      class_name: str, page_name: str, actions: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Fix failing tests using LLM with enhanced error and element analysis."""
        logger.info("Starting test fix process with enriched context...")
        
        # Create directory for diagnostic artifacts
        os.makedirs("test-results", exist_ok=True)
        
        # Extract test file path for diagnostics
        test_file = f"test_{page_name}_page.py"
        test_file_path = os.path.join(RECORDED_TESTS_DIR, test_file)
        
        # Analyze the error
        error_info = DiagnosticRunner.analyze_error(error_output)
        
        # Run diagnostic test to capture page state at failure point
        screenshot_path, html_path, dom_context = DiagnosticRunner.capture_page_state(
            test_file_path, page_name, error_info
        )
        
        # Extract failing action details from recorded actions if possible
        failing_action_context = ""
        if error_info['element']:
            element_name = str(error_info['element'])
            # Find actions that might match the failing element
            relevant_actions = []
            for action in actions:
                if not isinstance(action, dict):
                    continue
                    
                # Check action data for matches to failing element
                action_str = json.dumps(action).lower()
                if any(term.lower() in action_str for term in element_name.split() if len(term) > 3):
                    relevant_actions.append(action)
                    
            if relevant_actions:
                failing_action_context = "Potentially failing actions:\n" + json.dumps(relevant_actions, indent=2)
        
        # Construct prompt for fixing the test, letting the LLM figure out element details
        prompt = f"""Fix the failing Playwright test based on detailed error analysis and page context.

Error Details:
- Type: {error_info['type'] or 'Unknown error'}
- Failed Element: {error_info['element'] or 'Unknown element'}
- Expected State: {error_info['state'] or 'Unknown state'}
- Raw Error: {error_output[:500]}...

Page State at Failure:
{dom_context}

{failing_action_context}

The test is trying to interact with a {error_info['element']} element but is failing.
Carefully analyze the DOM structure from the page capture to understand what's actually on the page.

Original Recorded Actions:
{json.dumps(actions, indent=2)}

Current Page Object Code:
{page_code}

Current Test Code:
{test_code}

Fix Requirements:
1. Analyze the HTML structure to find the most reliable way to target elements
2. For timeout errors:
   - Consider waiting for different elements that appear more reliably
   - For modals/dialogs, try waiting for input fields inside them instead of the container
   - Try completely different approaches if the same strategy has failed before
3. For DOM structure/selector issues:
   - Look at the provided DOM fragment to determine better selectors
   - Check if elements are in iframes or shadow DOM
   - Consider adding explicit waits after page loads and clicks
4. For input/interaction issues:
   - Check if the element is actually the right type for the interaction
   - For search boxes, pay attention if they're two-step processes (click, then input)
   - For GitHub-style sites, remember search is often a button that opens a dialog first

Return format (ONLY valid Python code, no explanations or comments):
# Page Object Model
[fixed page code]

# Test File
[fixed test code]"""

        logger.info("Sending enhanced fix request to LLM...")
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        code = cast(str, response.content)

        if "# Page Object Model" not in code or "# Test File" not in code:
            logger.error("Invalid code format received from LLM")
            raise ValueError("Invalid code format from LLM")

        parts = code.split("# Page Object Model")
        if len(parts) != 2:
            logger.error("Invalid code structure received from LLM")
            raise ValueError("Invalid code format")

        fixed_page_code = self.code_helper.strip_markdown(parts[1].split("# Test File")[0].strip())
        fixed_test_code = self.code_helper.strip_markdown(parts[1].split("# Test File")[1].strip())

        # Log differences between original and fixed code
        logger.info("Key changes in page object code:")
        CodeHelper.log_code_differences(page_code, fixed_page_code)
        
        logger.info("Key changes in test code:")
        CodeHelper.log_code_differences(test_code, fixed_test_code)

        return fixed_page_code, fixed_test_code

    async def verify_test(self, test_file: str) -> Tuple[bool, str]:
        """Verify a test file by running it."""
        logger.info(f"Verifying test: {test_file}")
        result = subprocess.run(['pytest', test_file, '-v'], 
                             capture_output=True, 
                             text=True)
                             
        success = result.returncode == 0
        if success:
            logger.info("Test verification passed! ✅")
        else:
            logger.info("Test verification failed! ❌")
                    
        return success, result.stdout + "\n" + result.stderr

    async def generate_test(self, json_path: str) -> None:
        """Generate and verify test files with retries."""
        max_fix_attempts = 3
        
        try:
            logger.info(f"📋 Starting test generation for {json_path}")
            logger.info("=" * 80)
            
            with open(json_path, 'r') as f:
                actions = json.load(f)
            
            if not isinstance(actions, list):
                logger.error("Invalid JSON format: actions must be a list")
                raise ValueError("Invalid actions format")

            url = next((a['go_to_url']['url'] for a in actions if 'go_to_url' in a), None)
            if not url:
                logger.error("No starting URL found in actions")
                raise ValueError("No URL found in actions")

            # Use JSON filename without extension for test name
            json_filename = os.path.splitext(os.path.basename(json_path))[0]
            page_name = json_filename.replace('_test', '')  # Remove _test suffix if present
            class_name = get_class_name(page_name)
            page_file = f"{page_name}_page.py"
            test_file = f"test_{page_name}_page.py"
            test_file_path = os.path.join(RECORDED_TESTS_DIR, test_file)
            
            logger.info(f"Test details:")
            logger.info(f"  Page name: {page_name}")
            logger.info(f"  Class name: {class_name}")
            logger.info(f"  Page file: {page_file}")
            logger.info(f"  Test file: {test_file}")
            logger.info(f"  Actions count: {len(actions)}")
            logger.info("=" * 80)

            # Step 1: Generate the test from recorded actions
            logger.info("🔄 STEP 1: Generating test from recorded browser interactions...")
            logger.info("-" * 80)
            success, page_code, test_code, output = await self.generate_and_verify_test(
                actions=actions,
                url=url,
                page_name=page_name,
                class_name=class_name,
                page_file=page_file,
                test_file=test_file
            )

            # Step 2: Check if the test passed on first attempt
            if success:
                logger.info("=" * 80)
                logger.info("✅ FINAL RESULT: Test passed on the first attempt.")
                logger.info("=" * 80)
                return

            logger.info("=" * 80)
            logger.info("❌ STEP 1 RESULT: Initial test failed. Starting fix attempts...")
            logger.info("=" * 80)

            # Step 3: If test failed, try to fix it up to max_fix_attempts times
            for attempt in range(1, max_fix_attempts + 1):
                logger.info(f"🔄 STEP 2.{attempt}: Fix attempt {attempt}/{max_fix_attempts}...")
                logger.info("-" * 80)
                
                try:
                    # Try to fix the test by sending code and error to LLM
                    fixed_page_code, fixed_test_code = await self.fix_test(
                        page_code=page_code,
                        test_code=test_code,
                        error_output=output,
                        class_name=class_name,
                        page_name=page_name,
                        actions=actions
                    )

                    # Save fixed code
                    with open(os.path.join(PAGES_DIR, page_file), "w") as f:
                        f.write(fixed_page_code.strip())
                    with open(test_file_path, "w") as f:
                        f.write(fixed_test_code.strip())

                    # Execute the fixed test
                    logger.info(f"🧪 Running fix attempt {attempt}...")
                    success, new_output = await self.verify_test(test_file_path)
                    
                    if success:
                        logger.info("=" * 80)
                        logger.info(f"✅ FINAL RESULT: Fix attempt {attempt} passed.")
                        logger.info("=" * 80)
                        return

                    # Update for next iteration if we still need to retry
                    output = new_output
                    page_code = fixed_page_code
                    test_code = fixed_test_code
                    logger.info("=" * 80)
                    logger.info(f"❌ STEP 2.{attempt} RESULT: Fix attempt {attempt} failed.")
                    if attempt < max_fix_attempts:
                        logger.info(f"Will continue with fix attempt {attempt+1}.")
                    logger.info("=" * 80)

                except Exception as e:
                    logger.error(f"⚠️ Error during fix attempt {attempt}: {str(e)}")
                    logger.info("=" * 80)
                    logger.info(f"❌ STEP 2.{attempt} RESULT: Fix attempt {attempt} failed with exception.")
                    if attempt < max_fix_attempts:
                        logger.info(f"Will continue with fix attempt {attempt+1}.")
                    logger.info("=" * 80)
                    # Continue with next attempt if any remain
            
            # If we get here, all fix attempts failed
            logger.info("=" * 80)
            logger.info(f"❌ FINAL RESULT: All {max_fix_attempts} fix attempts failed.")
            logger.info("=" * 80)
            raise ValueError(f"❌ All {max_fix_attempts} fix attempts failed. Last error:\n{output}")

        except Exception as e:
            logger.error("=" * 80)
            logger.error("❌ Test generation process failed", exc_info=True)
            logger.error("=" * 80)
            raise


async def main():
    try:
        generator = TestGenerator(sys.argv[1] if len(sys.argv) > 1 else "github_search_test")
        json_file = sys.argv[2] if len(sys.argv) > 2 else "test_recorder/recorded_tests/github_search_test.json"
        await generator.generate_test(json_file)
    except Exception as e:
        logger.error(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    import sys
    import asyncio
    asyncio.run(main())