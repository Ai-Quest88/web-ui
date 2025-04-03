import pdb
import logging
import json
import time
import sys
import subprocess

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState
from src.test_recorder.agent_test_generator import AITestAgent

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

async def generate_test(task_description: str, llm_provider: str, llm_model_name: str, llm_api_key: str, llm_base_url: str):
    """Generate test code based on task description"""
    try:
        agent = AITestAgent(
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url
        )
        
        # Initial state
        yield "", gr.update(visible=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False), "Starting test generation..."
        
        # Set task
        agent.log("\n🎯 Starting test generation for task:")
        agent.log(f"📝 {task_description}")
        agent.set_task(task_description)
        yield "", gr.update(visible=False), gr.update(interactive=False), gr.update(visible=False), gr.update(visible=False), "\n".join(agent.logs)
        
        # Generate and verify test code with retries
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                agent.log(f"\n📝 Attempt {attempt + 1}/{max_attempts} to generate working test...")
                result = await agent.generate_test_code()
                
                # Handle tuple return value
                if isinstance(result, tuple):
                    test_code, gen_media_paths = result
                else:
                    test_code = result
                    gen_media_paths = {}
                
                # Run test for verification
                agent.log("\n🔍 Verifying generated test...")
                success, error_msg, exec_media_paths = await agent.execute_test(test_code)
                
                # Get video if available
                video_path = None
                if exec_media_paths and "videos" in exec_media_paths and exec_media_paths["videos"]:
                    video_path = exec_media_paths["videos"][-1]  # Get latest video
                
                if success:
                    agent.log("\n✅ Test verification passed!")
                    if video_path:
                        agent.log(f"\n🎥 Test recording saved: {video_path}")
                    yield (
                        test_code,
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                        gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                        "\n".join(agent.logs)
                    )
                    return
                else:
                    if attempt < max_attempts - 1:
                        agent.log("\n🔄 Test failed verification - retrying with error feedback...")
                        agent.log(f"Error: {error_msg}")
                        continue
                    else:
                        agent.log("\n❌ Failed to generate working test after max attempts")
                        error_msg = f"Failed to generate working test after {max_attempts} attempts"
                        yield (
                            "",
                            gr.update(visible=True, value=error_msg),
                            gr.update(interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "\n".join(agent.logs)
                        )
                        return
                        
            except Exception as e:
                if attempt < max_attempts - 1:
                    agent.log(f"\n⚠️ Error in attempt {attempt + 1}: {str(e)}")
                    agent.log("🔄 Retrying...")
                    continue
                else:
                    error_msg = f"Error during test generation: {str(e)}"
                    agent.log(f"\n❌ {error_msg}")
                    yield (
                        "",
                        gr.update(visible=True, value=error_msg),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        "\n".join(agent.logs)
                    )
                    return
        
    except Exception as e:
        error_msg = f"Error during test generation: {str(e)}"
        yield (
            "",
            gr.update(visible=True, value=error_msg),
            gr.update(interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            error_msg
        )

async def execute_test(test_code: str, llm_provider: str, llm_model_name: str, llm_api_key: str, llm_base_url: str):
    """Execute generated test code"""
    try:
        agent = AITestAgent(
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url
        )
        
        # Initial state
        yield (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            "Starting test execution..."
        )
        
        # Execute test
        agent.log("\n🚀 Executing test...")
        success, error_msg, media_paths = await agent.execute_test(test_code)
        
        # Get screenshot and video if available
        screenshot_path = None
        video_path = None
        if media_paths:
            if "screenshots" in media_paths and media_paths["screenshots"]:
                screenshot_path = media_paths["screenshots"][-1]  # Get latest screenshot
            if "videos" in media_paths and media_paths["videos"]:
                video_path = media_paths["videos"][-1]  # Get latest video
        
        if success:
            agent.log("\n✅ Test executed successfully!")
            if video_path:
                agent.log(f"\n🎥 Test recording saved: {video_path}")
            if screenshot_path:
                agent.log(f"\n📸 Screenshot saved: {screenshot_path}")
            yield (
                gr.update(visible=False),
                gr.update(visible=True, value=screenshot_path) if screenshot_path else gr.update(visible=False),
                gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                "\n".join(agent.logs)
            )
        else:
            agent.log("\n❌ Test execution failed!")
            if video_path:
                agent.log(f"\n🎥 Failure recording saved: {video_path}")
            if screenshot_path:
                agent.log(f"\n📸 Failure screenshot saved: {screenshot_path}")
            yield (
                gr.update(visible=True, value=error_msg),
                gr.update(visible=True, value=screenshot_path) if screenshot_path else gr.update(visible=False),
                gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                gr.update(visible=True, value=video_path) if video_path else gr.update(visible=False),
                "\n".join(agent.logs)
            )
                
    except Exception as e:
        error_msg = f"Error during test execution: {str(e)}"
        yield (
            gr.update(visible=True, value=error_msg),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            error_msg
        )

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()

async def stop_agent():
    """Stop the currently running agent"""
    try:
        if '_global_agent' in globals() and _global_agent is not None and hasattr(_global_agent, 'stop'):
            if asyncio.iscoroutinefunction(_global_agent.stop):
                await _global_agent.stop()
            else:
                _global_agent.stop()
        return '', gr.update(value="Stop", interactive=True), gr.update(interactive=True)
    except Exception as e:
        return f"Error stopping agent: {str(e)}", gr.update(value="Stop", interactive=True), gr.update(interactive=True)

async def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state, _global_browser_context, _global_browser

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (                                   # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),                      # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    global _global_agent_state
    _global_agent_state.clear_stop()  # Clear any previous stop requests

    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',                                         # final_result
            errors,                                     # errors
            '',                                         # model_actions
            '',                                         # model_thoughts
            None,                                       # latest_video
            None,                                       # history_file
            None,                                       # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)    # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent
        
        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None
            
        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method
):
    try:
        global _global_browser, _global_browser_context, _global_agent_state, _global_agent

        # Clear any previous stop request
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )
            
        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)        

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None


            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stop", interactive=True),  # Re-enable stop button
                        gr.update(interactive=True)  # Re-enable run button
                    ]
                await asyncio.sleep(0.05)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                latest_videos = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)    # Re-enable run button
            ]

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}

async def close_global_browser():
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None
        
async def run_deep_search(research_task, max_search_iteration_input, max_query_per_iter_input, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser, headless):
    from src.utils.deep_research import deep_research
    global _global_agent_state

    # Clear any previous stop request
    _global_agent_state.clear_stop()
    
    llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
    markdown_content, file_path = await deep_research(research_task, llm, _global_agent_state,
                                                        max_search_iterations=max_search_iteration_input,
                                                        max_query_num=max_query_per_iter_input,
                                                        use_vision=use_vision,
                                                        headless=headless,
                                                        use_own_browser=use_own_browser
                                                        )
    
    return markdown_content, file_path, gr.update(value="Stop", interactive=True),  gr.update(interactive=True) 
    

async def run_multiple_agents(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task1,
    task2,
    task3,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    # Create browsers list
    browsers = []
    browser_contexts = []
    
    # Initialize task_results with empty dictionaries
    task_results = [
        {
            'final_result': '',
            'errors': '',
            'model_actions': '',
            'model_thoughts': '',
            'trace_file': None,
            'history_file': None
        } for _ in range(3)
    ]
    
    try:
        # Create API rate limiter semaphore
        api_rate_limiter = asyncio.Semaphore(2)  # Allow 2 concurrent API calls
        
        # Create browsers for each task
        for _ in range(3):
            extra_chromium_args = [f"--window-size={window_w},{window_h}"]
            if use_own_browser:
                chrome_path = os.getenv("CHROME_PATH", None)
                if chrome_path == "":
                    chrome_path = None
                chrome_user_data = os.getenv("CHROME_USER_DATA", None)
                if chrome_user_data:
                    extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
            else:
                chrome_path = None
                
            browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )
            browsers.append(browser)
        
        async def run_agent_with_rate_limit(task_idx, task_text):
            if not task_text.strip():  # Skip empty tasks
                return None
                
            try:
                # Create browser context for this task
                browser_context = await browsers[task_idx].new_context(
                    config=BrowserContextConfig(
                        trace_path=os.path.join(save_trace_path, f"task_{task_idx+1}") if save_trace_path else None,
                        save_recording_path=os.path.join(save_recording_path, f"task_{task_idx+1}") if save_recording_path else None,
                        no_viewport=False,
                        browser_window_size=BrowserContextWindowSize(
                            width=window_w, height=window_h
                        ),
                    )
                )
                browser_contexts.append(browser_context)
                
                async with api_rate_limiter:  # Only rate limit the API calls
                    # Create LLM instance for this task
                    llm = utils.get_llm_model(
                        provider=llm_provider,
                        model_name=llm_model_name,
                        num_ctx=llm_num_ctx,
                        temperature=llm_temperature,
                        base_url=llm_base_url,
                        api_key=llm_api_key,
                    )
                
                    # Create controller for this task
                    controller = CustomController()
                    
                    # Create agent for this task
                    agent = CustomAgent(
                        task=task_text,
                        add_infos=add_infos,
                        use_vision=use_vision,
                        llm=llm,
                        browser=browsers[task_idx],
                        browser_context=browser_context,
                        controller=controller,
                        system_prompt_class=CustomSystemPrompt,
                        agent_prompt_class=CustomAgentMessagePrompt,
                        max_actions_per_step=max_actions_per_step,
                        tool_calling_method=tool_calling_method
                    )
                    
                    # Run the agent
                    history = await agent.run(max_steps=max_steps)
                    
                    # Save history file
                    history_file = os.path.join(
                        os.path.join(save_agent_history_path, f"task_{task_idx+1}"), 
                        f"agent_{task_idx+1}.json"
                    )
                    os.makedirs(os.path.dirname(history_file), exist_ok=True)
                    agent.save_history(history_file)
                    
                    # Get final result from history
                    final_result = history.final_result()
                    if not final_result:  # If final_result is empty, try to get it from the last thought
                        thoughts = history.model_thoughts()
                        if thoughts and len(thoughts) > 0:
                            final_result = str(thoughts[-1])  # Get the last thought as result
                    
                    # Return task results with guaranteed non-empty final result
                    return {
                        'final_result': final_result if final_result else f"Task {task_idx+1} completed",
                        'errors': history.errors(),
                        'model_actions': history.model_actions(),
                        'model_thoughts': history.model_thoughts(),
                        'trace_file': get_latest_files(os.path.join(save_trace_path, f"task_{task_idx+1}")).get('.zip'),
                        'history_file': history_file
                    }
            except Exception as e:
                import traceback
                error_msg = f"Task {task_idx+1} error: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                # Return the error result with a clear final result message
                return {
                    'final_result': f"Task {task_idx+1} failed: {str(e)}",
                    'errors': error_msg,
                    'model_actions': '',
                    'model_thoughts': '',
                    'trace_file': None,
                    'history_file': None
                }

        # Create tasks list with non-empty tasks
        tasks = [(idx, task) for idx, task in enumerate([task1, task2, task3]) if task.strip()]
        
        # Create a queue for UI updates
        ui_update_queue = asyncio.Queue()
        
        async def run_task_and_queue_update(task_idx, task_text):
            result = await run_agent_with_rate_limit(task_idx, task_text)
            if result is not None:
                task_results[task_idx] = result
                await ui_update_queue.put((task_idx, result))  # Queue the task index and result
        
        # Start all tasks concurrently
        running_tasks = [asyncio.create_task(run_task_and_queue_update(idx, text)) for idx, text in tasks]
        
        # Process UI updates as they come in
        while running_tasks:
            done, pending = await asyncio.wait(running_tasks, timeout=0.1, return_when=asyncio.FIRST_COMPLETED)
            running_tasks = list(pending)
            
            # Process any completed tasks
            for task in done:
                try:
                    await task  # Ensure any exceptions are raised
                except Exception as e:
                    logger.error(f"Task error: {str(e)}")
            
            # Update UI with current results - matching all UI components
            yield (
                task_results[0]['final_result'],     # Task 1 result
                task_results[1]['final_result'],     # Task 2 result
                task_results[2]['final_result'],     # Task 3 result
                task_results[0]['errors'],           # Task 1 errors
                task_results[1]['errors'],           # Task 2 errors
                task_results[2]['errors'],           # Task 3 errors
                task_results[0]['model_actions'],    # Task 1 actions
                task_results[1]['model_actions'],    # Task 2 actions
                task_results[2]['model_actions'],    # Task 3 actions
                task_results[0]['model_thoughts'],   # Task 1 thoughts
                task_results[1]['model_thoughts'],   # Task 2 thoughts
                task_results[2]['model_thoughts'],   # Task 3 thoughts
                task_results[0]['trace_file'],       # Task 1 trace
                task_results[1]['trace_file'],       # Task 2 trace
                task_results[2]['trace_file'],       # Task 3 trace
                task_results[0]['history_file'],     # Task 1 history
                task_results[1]['history_file'],     # Task 2 history
                task_results[2]['history_file'],     # Task 3 history
                gr.update(value="Stop", interactive=True),  # Stop button
                gr.update(interactive=True)                 # Run button
            )
            
        # Final UI update with the same structure
        yield (
            task_results[0]['final_result'],     # Task 1 result
            task_results[1]['final_result'],     # Task 2 result
            task_results[2]['final_result'],     # Task 3 result
            task_results[0]['errors'],           # Task 1 errors
            task_results[1]['errors'],           # Task 2 errors
            task_results[2]['errors'],           # Task 3 errors
            task_results[0]['model_actions'],    # Task 1 actions
            task_results[1]['model_actions'],    # Task 2 actions
            task_results[2]['model_actions'],    # Task 3 actions
            task_results[0]['model_thoughts'],   # Task 1 thoughts
            task_results[1]['model_thoughts'],   # Task 2 thoughts
            task_results[2]['model_thoughts'],   # Task 3 thoughts
            task_results[0]['trace_file'],       # Task 1 trace
            task_results[1]['trace_file'],       # Task 2 trace
            task_results[2]['trace_file'],       # Task 3 trace
            task_results[0]['history_file'],     # Task 1 history
            task_results[1]['history_file'],     # Task 2 history
            task_results[2]['history_file'],     # Task 3 history
            gr.update(value="Stop", interactive=True),  # Stop button
            gr.update(interactive=True)                 # Run button
        )
    except Exception as e:
        import traceback
        error_msg = f"Error running multiple agents: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        # Return empty/error values for all 20 expected outputs
        yield (
            "", "", "",                          # Final results
            error_msg, error_msg, error_msg,     # Errors
            "", "", "",                          # Model actions
            "", "", "",                          # Model thoughts
            None, None, None,                    # Trace files
            None, None, None,                    # History files
            gr.update(value="Stop", interactive=True),  # Stop button
            gr.update(interactive=True)                 # Run button
        )
    finally:
        # Clean up resources
        for context in browser_contexts:
            try:
                await context.close()
            except Exception as e:
                logger.error(f"Error closing browser context: {str(e)}")
                
        if not keep_browser_open:
            for browser in browsers:
                try:
                    await browser.close()
                except Exception as e:
                    logger.error(f"Error closing browser: {str(e)}")

def create_ui():
    """Create the Gradio UI"""
    
    # Create theme
    theme = Soft()

    # Create blocks
    with gr.Blocks(theme=theme, title="Browser Use") as demo:
        gr.Markdown("# Browser Use")
        
        with gr.Tabs() as tabs:
            # Single Agent Tab
            with gr.Tab("Single Agent", id="single_agent"):
                with gr.Row():
                    with gr.Column():
                        agent_type = gr.Radio(
                            choices=["custom", "org"],
                            value="custom",
                            label="Agent Type",
                            interactive=True
                        )
                        task = gr.Textbox(
                            label="Task",
                            placeholder="Enter task description",
                            lines=3,
                            interactive=True
                        )
                        add_infos = gr.Textbox(
                            label="Additional Info",
                            placeholder="Enter additional information",
                            lines=3,
                            interactive=True
                        )
                        with gr.Row():
                            run_button = gr.Button("Run", variant="primary")
                            stop_button = gr.Button("Stop")

                    with gr.Column():
                        stream_output = gr.HTML(
                            label="Browser View"
                        )
                        final_result = gr.Textbox(
                            label="Final Result",
                            lines=3,
                            interactive=False
                        )
                        errors_output = gr.Textbox(
                            label="Errors",
                            lines=3,
                            interactive=False
                        )
                        model_actions = gr.Textbox(
                            label="Model Actions",
                            lines=3,
                            interactive=False
                        )
                        model_thoughts = gr.Textbox(
                            label="Model Thoughts",
                            lines=3,
                            interactive=False
                        )
                        with gr.Row():
                            trace_output = gr.File(
                                label="Trace",
                                interactive=False
                            )
                            history_output = gr.File(
                                label="History",
                                interactive=False
                            )

            # Multiple Agents Tab
            with gr.Tab("Multiple Agents", id="multiple_agents"):
                with gr.Row():
                    with gr.Column():
                        agent_type_multi = gr.Radio(
                            choices=["custom", "org"],
                            value="custom",
                            label="Agent Type",
                            interactive=True
                        )
                        task1 = gr.Textbox(
                            label="Task 1",
                            placeholder="Enter task description",
                            lines=3,
                            interactive=True
                        )
                        task2 = gr.Textbox(
                            label="Task 2",
                            placeholder="Enter task description",
                            lines=3,
                            interactive=True
                        )
                        task3 = gr.Textbox(
                            label="Task 3",
                            placeholder="Enter task description",
                            lines=3,
                            interactive=True
                        )
                        add_infos_multi = gr.Textbox(
                            label="Additional Info",
                            placeholder="Enter additional information",
                            lines=3,
                            interactive=True
                        )
                        with gr.Row():
                            run_multi_button = gr.Button("Run", variant="primary")
                            stop_multi_button = gr.Button("Stop")

                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                result1 = gr.Textbox(
                                    label="Task 1 Result",
                                    lines=3,
                                    interactive=False
                                )
                                errors1 = gr.Textbox(
                                    label="Task 1 Errors",
                                    lines=3,
                                    interactive=False
                                )
                                actions1 = gr.Textbox(
                                    label="Task 1 Actions",
                                    lines=3,
                                    interactive=False
                                )
                                thoughts1 = gr.Textbox(
                                    label="Task 1 Thoughts",
                                    lines=3,
                                    interactive=False
                                )
                                trace1 = gr.File(
                                    label="Task 1 Trace",
                                    interactive=False
                                )
                                history1 = gr.File(
                                    label="Task 1 History",
                                    interactive=False
                                )

                            with gr.Column():
                                result2 = gr.Textbox(
                                    label="Task 2 Result",
                                    lines=3,
                                    interactive=False
                                )
                                errors2 = gr.Textbox(
                                    label="Task 2 Errors",
                                    lines=3,
                                    interactive=False
                                )
                                actions2 = gr.Textbox(
                                    label="Task 2 Actions",
                                    lines=3,
                                    interactive=False
                                )
                                thoughts2 = gr.Textbox(
                                    label="Task 2 Thoughts",
                                    lines=3,
                                    interactive=False
                                )
                                trace2 = gr.File(
                                    label="Task 2 Trace",
                                    interactive=False
                                )
                                history2 = gr.File(
                                    label="Task 2 History",
                                    interactive=False
                                )

                            with gr.Column():
                                result3 = gr.Textbox(
                                    label="Task 3 Result",
                                    lines=3,
                                    interactive=False
                                )
                                errors3 = gr.Textbox(
                                    label="Task 3 Errors",
                                    lines=3,
                                    interactive=False
                                )
                                actions3 = gr.Textbox(
                                    label="Task 3 Actions",
                                    lines=3,
                                    interactive=False
                                )
                                thoughts3 = gr.Textbox(
                                    label="Task 3 Thoughts",
                                    lines=3,
                                    interactive=False
                                )
                                trace3 = gr.File(
                                    label="Task 3 Trace",
                                    interactive=False
                                )
                                history3 = gr.File(
                                    label="Task 3 History",
                                    interactive=False
                                )

            # Deep Research Tab
            with gr.Tab("Deep Research", id="deep_research"):
                with gr.Row():
                    with gr.Column():
                        research_task = gr.Textbox(
                            label="Research Task",
                            placeholder="Enter research task description",
                            lines=3,
                            interactive=True
                        )
                        max_search_iteration_input = gr.Number(
                            value=3,
                            label="Max Search Iterations",
                            interactive=True
                        )
                        max_query_per_iter_input = gr.Number(
                            value=3,
                            label="Max Queries per Iteration",
                            interactive=True
                        )
                        with gr.Row():
                            run_research_button = gr.Button("Run", variant="primary")
                            stop_research_button = gr.Button("Stop")

                    with gr.Column():
                        research_output = gr.Markdown(
                            value="Research results will appear here..."
                        )
                        research_file = gr.File(
                            label="Research File",
                            interactive=False
                        )

            # Settings Tab
            with gr.Tab("Settings", id="settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### LLM Settings")
                        llm_provider = gr.Dropdown(
                            choices=[provider for provider, model in utils.model_names.items()],
                            value="openai",
                            label="LLM Provider",
                            info="Select your preferred language model provider",
                            interactive=True
                        )
                        llm_model_name = gr.Dropdown(
                            choices=utils.model_names[llm_provider.value],
                            value=utils.model_names[llm_provider.value][0] if utils.model_names[llm_provider.value] else "",
                            label="Model Name",
                            info="Select a model from the dropdown or type a custom model name",
                            interactive=True,
                            allow_custom_value=True
                        )
                        llm_num_ctx = gr.Slider(
                            minimum=1000,
                            maximum=128000,
                            value=4000,
                            step=1000,
                            label="Context Length",
                            info="Maximum context length in tokens",
                            interactive=True
                        )
                        llm_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness in the output (0.0 = deterministic, 2.0 = very random)",
                            interactive=True
                        )
                        llm_base_url = gr.Textbox(
                            value="",
                            label="Base URL (optional)",
                            info="Custom API endpoint URL (leave blank to use default)",
                            interactive=True
                        )
                        llm_api_key = gr.Textbox(
                            value="",
                            label="API Key",
                            type="password",
                            info="Your API key (leave blank to use .env)",
                            interactive=True
                        )

                    with gr.Column():
                        gr.Markdown("### Browser Settings")
                        use_own_browser = gr.Checkbox(
                            value=False,
                            label="Use Own Browser",
                            interactive=True
                        )
                        keep_browser_open = gr.Checkbox(
                            value=False,
                            label="Keep Browser Open",
                            interactive=True
                        )
                        headless = gr.Checkbox(
                            value=False,
                            label="Headless Mode",
                            interactive=True
                        )
                        disable_security = gr.Checkbox(
                            value=False,
                            label="Disable Security",
                            interactive=True
                        )
                        window_w = gr.Number(
                            value=1280,
                            label="Window Width",
                            interactive=True
                        )
                        window_h = gr.Number(
                            value=720,
                            label="Window Height",
                            interactive=True
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Recording Settings")
                        save_recording_path = gr.Textbox(
                            value="recordings",
                            label="Recording Path",
                            interactive=True
                        )
                        save_agent_history_path = gr.Textbox(
                            value="agent_history",
                            label="Agent History Path",
                            interactive=True
                        )
                        save_trace_path = gr.Textbox(
                            value="traces",
                            label="Trace Path",
                            interactive=True
                        )
                        enable_recording = gr.Checkbox(
                            value=True,
                            label="Enable Recording",
                            interactive=True
                        )

                    with gr.Column():
                        gr.Markdown("### Agent Settings")
                        max_steps = gr.Number(
                            value=50,
                            label="Max Steps",
                            interactive=True
                        )
                        use_vision = gr.Checkbox(
                            value=False,
                            label="Use Vision",
                            interactive=True
                        )
                        max_actions_per_step = gr.Number(
                            value=5,
                            label="Max Actions Per Step",
                            interactive=True
                        )
                        tool_calling_method = gr.Radio(
                            choices=["function_calling", "json_mode"],
                            value="function_calling",
                            label="Tool Calling Method",
                            interactive=True
                        )

                with gr.Row():
                    save_config_btn = gr.Button("Save Config")
                    load_config_btn = gr.Button("Load Config")
                    reset_config_btn = gr.Button("Reset to Default")

            # Test Automator Tab
            with gr.Tab("Test Automator", id="test_automator"):
                with gr.Row():
                    with gr.Column():
                        task_input = gr.Textbox(
                            label="Task Description",
                            placeholder="Describe the test scenario",
                            lines=3,
                            interactive=True
                        )
                        generate_button = gr.Button("Generate Test", variant="primary")
                        execute_button = gr.Button("Execute Test", interactive=False)

                    with gr.Column():
                        test_code = gr.Code(
                            label="Generated Test Code",
                            language="python",
                            interactive=False
                        )
                        error_output = gr.Textbox(
                            label="Error",
                            visible=False,
                            interactive=False
                        )
                        screenshot_output = gr.Image(
                            label="Screenshot",
                            visible=False,
                            interactive=False
                        )
                        video_output = gr.Video(
                            label="Recording",
                            visible=False,
                            interactive=False
                        )
                        video_file = gr.File(
                            label="Download Recording",
                            visible=False,
                            interactive=False
                        )
                        logs_output = gr.Textbox(
                            label="Logs",
                            lines=10,
                            interactive=False
                        )

        # Event handlers
        llm_provider.change(
            fn=update_model_dropdown,
            inputs=[llm_provider],
            outputs=[llm_model_name]
        )

        save_config_btn.click(
            fn=save_current_config,
            inputs=[
                llm_provider, llm_model_name, llm_num_ctx, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, keep_browser_open,
                headless, disable_security, window_w, window_h,
                save_recording_path, save_agent_history_path, save_trace_path,
                enable_recording, max_steps, use_vision, max_actions_per_step,
                tool_calling_method
            ],
            outputs=[]
        )

        load_config_btn.click(
            fn=load_config_from_file,
            inputs=[],
            outputs=[
                llm_provider, llm_model_name, llm_num_ctx, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, keep_browser_open,
                headless, disable_security, window_w, window_h,
                save_recording_path, save_agent_history_path, save_trace_path,
                enable_recording, max_steps, use_vision, max_actions_per_step,
                tool_calling_method
            ]
        )

        reset_config_btn.click(
            fn=update_ui_from_config,
            inputs=[],
            outputs=[
                llm_provider, llm_model_name, llm_num_ctx, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, keep_browser_open,
                headless, disable_security, window_w, window_h,
                save_recording_path, save_agent_history_path, save_trace_path,
                enable_recording, max_steps, use_vision, max_actions_per_step,
                tool_calling_method
            ]
        )

        run_button.click(
            fn=run_with_stream,
            inputs=[
                agent_type,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                window_w,
                window_h,
                save_recording_path,
                save_agent_history_path,
                save_trace_path,
                enable_recording,
                task,
                add_infos,
                max_steps,
                use_vision,
                max_actions_per_step,
                tool_calling_method
            ],
            outputs=[
                stream_output,
                final_result,
                errors_output,
                model_actions,
                model_thoughts,
                trace_output,
                history_output,
                stop_button,
                run_button
            ]
        )

        stop_button.click(
            fn=stop_research_agent,
            inputs=[],
            outputs=[
                stop_button,
                run_button
            ]
        )

        run_multi_button.click(
            fn=run_multiple_agents,
            inputs=[
                agent_type_multi,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                window_w,
                window_h,
                save_recording_path,
                save_agent_history_path,
                save_trace_path,
                enable_recording,
                task1,
                task2,
                task3,
                add_infos_multi,
                max_steps,
                use_vision,
                max_actions_per_step,
                tool_calling_method
            ],
            outputs=[
                result1,
                result2,
                result3,
                errors1,
                errors2,
                errors3,
                actions1,
                actions2,
                actions3,
                thoughts1,
                thoughts2,
                thoughts3,
                trace1,
                trace2,
                trace3,
                history1,
                history2,
                history3,
                stop_multi_button,
                run_multi_button
            ]
        )

        stop_multi_button.click(
            fn=stop_research_agent,
            inputs=[],
            outputs=[
                stop_multi_button,
                run_multi_button
            ]
        )

        run_research_button.click(
            fn=run_deep_search,
            inputs=[
                research_task,
                max_search_iteration_input,
                max_query_per_iter_input,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_vision,
                use_own_browser,
                headless
            ],
            outputs=[
                research_output,
                research_file,
                stop_research_button,
                run_research_button
            ]
        )

        stop_research_button.click(
            fn=stop_research_agent,
            inputs=[],
            outputs=[
                stop_research_button,
                run_research_button
            ]
        )

        generate_button.click(
            fn=generate_test,
            inputs=[task_input, llm_provider, llm_model_name, llm_api_key, llm_base_url],
            outputs=[test_code, error_output, execute_button, video_output, video_file, logs_output]
        )

        execute_button.click(
            fn=execute_test,
            inputs=[test_code, llm_provider, llm_model_name, llm_api_key, llm_base_url],
            outputs=[error_output, screenshot_output, video_output, video_file, logs_output]
        )

        return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()

    # Create test artifacts directory if it doesn't exist
    artifacts_dir = os.path.join(os.getcwd(), "test_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    demo = create_ui()
    demo.launch(
        server_name=args.ip,
        server_port=args.port,
        share=False,
        inbrowser=False,
        inline=True
    )

if __name__ == '__main__':
    main()
