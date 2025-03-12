from playwright.sync_api import sync_playwright
from src.utils.browser import Browser
import logging
import time
from concurrent.futures import ThreadPoolExecutor

def setup_browser(position: int):
    """Setup and return a browser instance with specific window position
    
    Args:
        position: Integer to determine window position (0, 1, or 2)
    """
    playwright = sync_playwright().start()
    # Launch browser with specific window position
    browser = playwright.chromium.launch(headless=False, args=[
        f'--window-position={position * 800},0',
        '--window-size=800,900'
    ])
    page = browser.new_page()
    return Browser(page), browser, playwright

def agent_task(search_term: str):
    """Simulates an agent performing a search task"""
    # Get position based on search term
    position = {
        "OpenAI ChatGPT": 0,
        "Google Bard": 1,
        "Anthropic Claude": 2
    }.get(search_term, 0)
    
    browser_util, browser, playwright = setup_browser(position)
    logger = logging.getLogger(f"Agent-{search_term}")
    
    try:
        # Add initial delay based on position to stagger launches
        time.sleep(position * 2)
        
        # Navigate to Google
        logger.info(f"Agent searching for '{search_term}'...")
        success = browser_util.navigate_to_url('https://www.google.com')
        
        if success:
            # Input search term
            logger.info("Entering search term...")
            if browser_util.input_text(search_term):
                logger.info("Search term entered successfully")
                # Keep browser open longer to observe results
                time.sleep(10)  # Increased from 2 to 10 seconds
            else:
                logger.error("Failed to input search term")
        else:
            logger.error("Failed to navigate to Google")
            
    except Exception as e:
        logger.error(f"Task failed: {str(e)}")
    finally:
        browser.close()
        playwright.stop()

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Define search tasks for different agents
    search_terms = [
        "OpenAI ChatGPT",
        "Google Bard",
        "Anthropic Claude"
    ]
    
    logger.info("Starting multi-agent test...")
    
    # Run agents in parallel
    with ThreadPoolExecutor(max_workers=len(search_terms)) as executor:
        executor.map(agent_task, search_terms)
    
    logger.info("Multi-agent test completed")

if __name__ == "__main__":
    main() 