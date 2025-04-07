import os
import pickle
import uuid
import gradio as gr


def default_config():
    """Prepare the default configuration"""
    return {
        "agent_type": "custom",
        "max_steps": 100,
        "max_actions_per_step": 10,
        "use_vision": True,
        "tool_calling_method": "auto",
        "llm_provider": "mistral",
        "llm_model_name": "mistral-large-latest",
        "llm_num_ctx": 32000,
        "llm_temperature": 1.0,
        "llm_base_url": "",
        "llm_api_key": "",
        "use_own_browser": os.getenv("CHROME_PERSISTENT_SESSION", "false").lower() == "true",
        "keep_browser_open": False,
        "headless": False,
        "disable_security": True,
        "enable_recording": True,
        "window_w": 1280,
        "window_h": 1100,
        "save_recording_path": "./tmp/record_videos",
        "save_trace_path": "./tmp/traces",
        "save_agent_history_path": "./tmp/agent_history",
        "task": "go to google.com and type 'OpenAI' click search and give me the first url",
        # Default tasks for multi-agent tab
        "task1": "Go to google.com, search for 'Latest AI developments', click on the most recent news article, and summarize the key points.",
        "task2": "Visit github.com, search for 'AI agents', find a popular repository, and extract its main features and star count.",
        "task3": "Navigate to arxiv.org, search for 'multi-agent systems', find the most recent paper, and provide its title and abstract."
    }


def load_config_from_file(provider=None):
    """Load settings from a UUID.pkl file."""
    try:
        # Use default config if no provider is specified
        config = default_config()
        if provider:
            config["llm_provider"] = provider
        return (
            config["agent_type"],
            config["max_steps"],
            config["max_actions_per_step"],
            config["use_vision"],
            config["tool_calling_method"],
            config["llm_provider"],
            config["llm_model_name"],
            config["llm_num_ctx"],
            config["llm_temperature"],
            config["llm_base_url"],
            config["llm_api_key"],
            config["use_own_browser"],
            config["keep_browser_open"],
            config["headless"],
            config["disable_security"],
            config["enable_recording"],
            config["window_w"],
            config["window_h"],
            config["save_recording_path"],
            config["save_trace_path"],
            config["save_agent_history_path"],
            config["task"]
        )
    except Exception as e:
        return f"Error loading configuration: {str(e)}"


def save_config_to_file(settings, save_dir="./tmp/webui_settings"):
    """Save the current settings to a UUID.pkl file with a UUID name."""
    os.makedirs(save_dir, exist_ok=True)
    config_file = os.path.join(save_dir, f"{uuid.uuid4()}.pkl")
    with open(config_file, 'wb') as f:
        pickle.dump(settings, f)
    return f"Configuration saved to {config_file}"


def save_current_config(*args):
    current_config = {
        "agent_type": args[0],
        "max_steps": args[1],
        "max_actions_per_step": args[2],
        "use_vision": args[3],
        "tool_calling_method": args[4],
        "llm_provider": args[5],
        "llm_model_name": args[6],
        "llm_num_ctx": args[7],
        "llm_temperature": args[8],
        "llm_base_url": args[9],
        "llm_api_key": args[10],
        "use_own_browser": args[11],
        "keep_browser_open": args[12],
        "headless": args[13],
        "disable_security": args[14],
        "enable_recording": args[15],
        "window_w": args[16],
        "window_h": args[17],
        "save_recording_path": args[18],
        "save_trace_path": args[19],
        "save_agent_history_path": args[20],
        "task": args[21],
    }
    return save_config_to_file(current_config)


def update_ui_from_config(provider=None):
    """Update UI with default config values."""
    config = default_config()
    if provider:
        config["llm_provider"] = provider
    return (
        gr.update(value=config["agent_type"]),
        gr.update(value=config["max_steps"]),
        gr.update(value=config["max_actions_per_step"]),
        gr.update(value=config["use_vision"]),
        gr.update(value=config["tool_calling_method"]),
        gr.update(value=config["llm_provider"]),
        gr.update(value=config["llm_model_name"]),
        gr.update(value=config["llm_num_ctx"]),
        gr.update(value=config["llm_temperature"]),
        gr.update(value=config["llm_base_url"]),
        gr.update(value=config["llm_api_key"]),
        gr.update(value=config["use_own_browser"]),
        gr.update(value=config["keep_browser_open"]),
        gr.update(value=config["headless"]),
        gr.update(value=config["disable_security"]),
        gr.update(value=config["enable_recording"]),
        gr.update(value=config["window_w"]),
        gr.update(value=config["window_h"]),
        gr.update(value=config["save_recording_path"]),
        gr.update(value=config["save_trace_path"]),
        gr.update(value=config["save_agent_history_path"]),
        gr.update(value=config["task"])
    )
