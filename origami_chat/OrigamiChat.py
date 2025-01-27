from __future__ import annotations

from typing import Any, Dict, Type, cast

from maubot.handlers import command
from maubot.matrix import MaubotMessageEvent, parse_formatted
from maubot.plugin_base import Plugin
from mautrix.types import Format
from mautrix.types.event import message
from mautrix.types.event.type import EventType
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper):
        helper.copy("openai")

    @property
    def openai(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.get("openai", {}))


class OrigamiChat(Plugin):
    config: Config

    async def start(self):
        self.log.info(f"Starting Origami Chat")
        await super().start()

        if not self.config:
            raise Exception("Config is not initialized")

        self.config.load_and_update()

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    @command.new(name="gpt", help="Usage: !gpt <prompt>")
    @command.argument("prompt", pass_raw=True, required=True)
    async def gpt(self, event: MaubotMessageEvent, prompt: str) -> None:
        await self.client.send_receipt(
            room_id=event.room_id, event_id=event.event_id, receipt_type="m.read"
        )
        await self.client.set_typing(room_id=event.room_id, timeout=30000)
        payload = {
            "model": self.config.openai["model"],
            "messages": [
                {"role": "system", "content": self.config.openai["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.openai["temperature"],
            "max_completion_tokens": self.config.openai["max_completion_tokens"],
        }
        headers = {
            "Authorization": f"Bearer {self.config.openai['api_key']}",
            "Content-Type": "application/json",
        }
        try:
            async with self.http.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.ok:
                    data = await response.json()
                    message_ = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                    )
                    if message_:
                        content = message.TextMessageEventContent(
                            msgtype=message.MessageType.TEXT, body=message_
                        )
                        content.format = Format.HTML
                        content.body, content.formatted_body = await parse_formatted(
                            content.body, render_markdown=True, allow_html=False
                        )
                        if self.config.openai["reply"]:
                            content.set_reply(event, disable_fallback=True)
                        await self.client.send_message_event(
                            room_id=event.room_id,
                            event_type=EventType.ROOM_MESSAGE,
                            content=content,
                        )
                else:
                    self.log.warning(
                        f"OpenAI API request failed. Status: {response.status}, "
                        f"Body: {await response.text()}"
                    )
        except Exception as e:
            self.log.exception(f"Exception while calling OpenAI API: {e}")
        finally:
            # Tell the server we've stopped typing (timeout=0)
            await self.client.set_typing(room_id=event.room_id, timeout=0)
