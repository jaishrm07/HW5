import json
import os
from typing import Any, Callable, Dict, List

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None


DEFAULT_BASE_URL = "https://llm-api.arc.vt.edu/api/v1"
DEFAULT_MODEL = "gpt-oss-120b"

DEFAULT_SYSTEM_PROMPT = """
You are the chat interface for a tabletop assistive robot arm in PyBullet.

Your job is to interpret the user's instruction and either:
1. call one or more robot tools, or
2. answer briefly in plain language if no robot action is needed.

Rules:
- Only use the tools that are provided to you.
- Never invent Python code, filenames, helper functions, or capabilities.
- If the request is unsupported or the user explicitly wants manual control, call `switch_to_teleop`.
- If the user says "pick cube" without a number, use `cube1`.
- You may call multiple tools in sequence for multi-step requests.
- Keep the final user-facing response short and concrete.
""".strip()


DEFAULT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "pick_cube",
            "description": "Pick up one of the tabletop cubes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cube_name": {
                        "type": "string",
                        "enum": ["cube1", "cube2", "cube3"],
                        "description": "Which cube to pick.",
                    }
                },
                "required": ["cube_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_microwave",
            "description": "Grasp the microwave handle and pull the door open.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_cabinet",
            "description": "Grasp the cabinet handle and pull the drawer open.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_cabinet",
            "description": "Grasp the cabinet handle and push the drawer closed.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "go_home",
            "description": "Move the robot to its home pose and open the gripper.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "switch_to_teleop",
            "description": "Hand control back to the user for manual teleoperation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short reason for the handoff.",
                    }
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
        },
    },
]


class LLMChatRouter:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tools: List[Dict[str, Any]] | None = None,
    ):
        if OpenAI is None:
            raise RuntimeError("Missing dependency: install `openai` to enable the LLM router.")

        resolved_api_key = (
            api_key
            or os.getenv("VT_LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not resolved_api_key:
            raise RuntimeError(
                "Set `VT_LLM_API_KEY` or `OPENAI_API_KEY` before running the LLM chat interface."
            )

        resolved_base_url = (
            base_url
            or os.getenv("VT_LLM_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or DEFAULT_BASE_URL
        )
        resolved_model = (
            model
            or os.getenv("VT_LLM_MODEL")
            or os.getenv("OPENAI_MODEL")
            or DEFAULT_MODEL
        )

        self.client = OpenAI(api_key=resolved_api_key, base_url=resolved_base_url)
        self.model = resolved_model
        self.tools = tools or DEFAULT_TOOLS
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    def run_turn(
        self,
        user_text: str,
        tool_executor: Callable[[str, Dict[str, Any]], Dict[str, Any]],
    ) -> Dict[str, Any]:
        self.messages.append({"role": "user", "content": user_text})

        first_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="auto",
            temperature=0,
        )
        assistant_message = first_response.choices[0].message
        tool_calls = list(assistant_message.tool_calls or [])

        if not tool_calls:
            reply = assistant_message.content or ""
            self.messages.append({"role": "assistant", "content": reply})
            return {"assistant_message": reply, "executed_steps": []}

        self.messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "{}",
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        executed_steps = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            result = tool_executor(tool_name, arguments)
            executed_steps.append(
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.tools,
            tool_choice="none",
            temperature=0,
        )
        final_message = final_response.choices[0].message
        reply = final_message.content or "Done."
        self.messages.append({"role": "assistant", "content": reply})
        return {"assistant_message": reply, "executed_steps": executed_steps}
