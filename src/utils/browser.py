import logging
import time
from playwright.sync_api import Page, expect

class Browser:
    def __init__(self, page: Page):
        """Initialize the Browser with a Playwright Page.
        
        Args:
            page: Playwright Page instance
        """
        self.page = page
        
    def navigate_to_url(self, url: str, max_retries: int = 3, retry_delay: float = 1.0) -> bool:
        """Navigate to a URL with retry logic and redirect handling.
        
        Args:
            url: The URL to navigate to
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will be exponentially increased)
            
        Returns:
            bool: True if navigation was successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                self.page.goto(url)
                
                # Wait for page load
                self.page.wait_for_load_state('networkidle')
                
                # Check if we landed on the intended URL
                current_url = self.page.url
                if url in current_url or current_url == url:
                    return True
                    
                # If redirected, try to navigate back and retry
                self.page.go_back()
                time.sleep(retry_delay * (2 ** attempt))
                
            except Exception as e:
                logging.error(f"Navigation error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))
                
        return False

    def input_text(self, text: str, index: int = 0, max_retries: int = 3) -> bool:
        """Input text into an element with retry logic.
        
        Args:
            text: Text to input
            index: Index of the input element
            max_retries: Maximum number of retry attempts
            
        Returns:
            bool: True if text input was successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Try different selectors for the search input
                selectors = [
                    'textarea[name="q"]',  # Google's main search textarea
                    'input[name="q"]',     # Google's main search input
                    'input[type="search"]', # Generic search input
                    'input[type="text"]',   # Generic text input
                    f'input >> nth={index}' # Fallback to index-based selector
                ]
                
                # Try each selector until we find a visible input
                input_element = None
                for selector in selectors:
                    try:
                        logging.info(f"Trying selector: {selector}")
                        element = self.page.wait_for_selector(selector, state="visible", timeout=5000)
                        if element:
                            input_element = self.page.locator(selector).first
                            break
                    except Exception as e:
                        logging.debug(f"Selector {selector} failed: {str(e)}")
                        continue
                
                if not input_element:
                    raise Exception("Could not find a visible input element")
                
                # Clear existing text and input new text
                input_element.click()
                input_element.fill(text)
                
                # Press Enter to submit
                input_element.press('Enter')
                return True
                
            except Exception as e:
                logging.error(f"Text input error on attempt {attempt + 1}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    
        return False 