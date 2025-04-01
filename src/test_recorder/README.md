# Test Recorder

This module records browser interactions and saves them as structured JSON files that can be used to generate Playwright tests.

## Current Milestone

We have achieved a clean implementation that:
1. Records browser interactions using the browser-use agent
2. Captures full DOM information for each interacted element
3. Saves the complete model actions in a structured JSON format

### Directory Structure
```
test_recorder/
├── README.md
├── test_generator.py         # Records browser interactions
└── recorded_tests/          # Stores the recorded test JSONs
    └── github_login_test.json
```

### JSON Format
Each recorded test is saved as a JSON file containing an array of actions. Each action has:
1. Action type and parameters (go_to_url, input_text, click_element)
2. Full DOM information for interacted elements:
   - tag_name
   - xpath
   - attributes (all HTML attributes)
   - css_selector
   - parent branch path
   - and more

Example action:
```json
{
  "input_text": {
    "index": 1,
    "text": "test@example.com"
  },
  "interacted_element": {
    "tag_name": "input",
    "attributes": {
      "type": "text",
      "name": "login",
      "id": "login_field",
      ...
    },
    "css_selector": "...",
    ...
  }
}
```

## Next Steps
1. Create a test generator that:
   - Takes the recorded JSON
   - Uses LLM to analyze DOM elements
   - Generates reliable Playwright tests 