import pytest
from playwright.sync_api import Page, expect
from .pages.github_search_page import GithubSearchPage

@pytest.fixture
def github_search_page(page: Page) -> GithubSearchPage:
    return GithubSearchPage(page)

def test_github_search(page: Page, github_search_page: GithubSearchPage) -> None:
    # Arrange
    github_search_page.go_to()

    # Act
    github_search_page.click_search_button()
    github_search_page.input_search_text("langchain")
    github_search_page.click_search_result_option()
    github_search_page.click_repository_link()

    # Assert
    github_search_page.verify_navigation()