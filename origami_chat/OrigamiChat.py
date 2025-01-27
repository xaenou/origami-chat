from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Type, cast

from maubot.handlers import command
from maubot.matrix import MaubotMessageEvent, parse_formatted
from maubot.plugin_base import Plugin
from mautrix.types import Format
from mautrix.types.event import message
from mautrix.types.event.type import EventType
from mautrix.util.async_db import UpgradeTable
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

from .migrations import upgrade_table


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper):
        helper.copy("openai")

    @property
    def openai(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.get("openai", {}))


class OrigamiChat(Plugin):
    config: Config

    @classmethod
    def get_db_upgrade_table(cls) -> UpgradeTable:
        return upgrade_table

    async def start(self):
        self.log.info(f"Starting Origami Chat")
        await super().start()

        if not self.config:
            raise Exception("Config is not initialized")

        self.config.load_and_update()

        await self._cleanup_old_rate_instances()

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    @command.new(name="gpt", help="Usage: !gpt <prompt>")
    @command.argument("prompt", pass_raw=True, required=True)
    async def gpt(self, event: MaubotMessageEvent, prompt: str) -> None:
        if not prompt or prompt.strip() == "":
            await self.send_message(
                event=event,
                content_str="No input received. Please provide a question or topic.",
                reply=self.config.openai["reply"],
            )
            return

        if (
            len(prompt) > self.config.openai["input_character_limit"]
            and self.config.openai["enable_input_character_limit"]
        ):
            await self.send_message(
                event=event,
                content_str=f"Character limit exceeded. Please keep your prompt under {self.config.openai['input_character_limit']} characters",
                reply=self.config.openai["reply"],
            )
            return

        await self.client.send_receipt(
            room_id=event.room_id, event_id=event.event_id, receipt_type="m.read"
        )

        user_id = event.sender
        now_utc = datetime.now(timezone.utc)
        day_ago = (now_utc - timedelta(hours=24)).replace(tzinfo=None)

        if self.config.openai["enable_user_rate_limit"]:
            allowed = await self._check_user_rate_limit(
                user_id=user_id,
                since=day_ago,
                limit=self.config.openai["user_rate_limit"],
            )
            if not allowed:
                await self.send_message(
                    event=event,
                    content_str=f"You have reached your daily usage limit of {self.config.openai["user_rate_limit"]} prompts.",
                    reply=self.config.openai["reply"],
                )
                return

        if self.config.openai["enable_global_rate_limit"]:
            allowed = await self._check_global_rate_limit(
                since=day_ago,
                limit=self.config.openai["global_rate_limit"],
            )
            if not allowed:
                await self.send_message(
                    event=event,
                    content_str=f"Daily usage limit of {self.config.openai["global_rate_limit"]} prompts reached.",
                    reply=self.config.openai["reply"],
                )
                return

        await self.client.set_typing(room_id=event.room_id, timeout=1500)

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
                        await self._log_inference(user_id)

                    await self.send_message(
                        event=event,
                        content_str=message_,
                        reply=self.config.openai["reply"],
                    )
                else:
                    self.log.warning(
                        f"OpenAI API request failed. Status: {response.status}, "
                        f"Body: {await response.text()}"
                    )
        except Exception as e:
            self.log.exception(f"Exception while calling OpenAI API: {e}")
        finally:
            await self.client.set_typing(room_id=event.room_id, timeout=0)

    async def _check_user_rate_limit(
        self, user_id: str, since: datetime, limit: int
    ) -> bool:
        """Return True if user is under limit, False if they exceeded."""
        # SELECT COUNT(*) FROM usage_log
        #  WHERE user_id = $1 AND event_ts > $2
        count_query = """
            SELECT COUNT(*) 
            FROM usage_log
            WHERE user_id = $1
            AND event_ts > $2
        """
        usage_count = await self.database.fetchval(count_query, user_id, since)  # type: ignore
        return usage_count < limit

    async def _check_global_rate_limit(self, since: datetime, limit: int) -> bool:
        """Return True if total usage in last 24h is under limit."""
        count_query = """
            SELECT COUNT(*)
            FROM usage_log
            WHERE event_ts > $1
        """
        usage_count = await self.database.fetchval(count_query, since)  # type: ignore
        return usage_count < limit

    async def _log_inference(self, user_id: str) -> None:
        """Insert a new log record indicating a usage by a specific user."""
        insert_query = """
            INSERT INTO usage_log (user_id) 
            VALUES ($1)
        """
        await self.database.execute(insert_query, user_id)  # type: ignore

    async def _cleanup_old_rate_instances(self) -> None:
        """Remove rate instances older than 24 hours."""
        try:
            now_utc = datetime.now(timezone.utc)
            day_ago = now_utc - timedelta(hours=24)

            delete_query = """
                DELETE FROM usage_log
                WHERE event_ts <= $1
            """
            self.log.info("Cleaning up rate instances older than 24 hours...")
            deleted_count = await self.database.execute(delete_query, day_ago)  # type: ignore
            self.log.info(f"Deleted {deleted_count} old rate instances.")
        except Exception as e:
            self.log.exception("Failed to clean up old rate instances")

    async def send_message(
        self, event: MaubotMessageEvent, content_str: str, reply: bool
    ) -> None:
        try:
            content = message.TextMessageEventContent(
                msgtype=message.MessageType.TEXT, body=content_str
            )
            content.format = Format.HTML
            content.body, content.formatted_body = await parse_formatted(
                content.body, render_markdown=True, allow_html=False
            )
            if reply:
                content.set_reply(event, disable_fallback=True)
            await self.client.send_message_event(
                room_id=event.room_id,
                event_type=EventType.ROOM_MESSAGE,
                content=content,
            )
        except Exception as e:
            self.log.exception(f"Failed to send message: {e}")
