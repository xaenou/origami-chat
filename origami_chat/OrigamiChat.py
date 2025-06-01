from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Type, cast

from maubot.handlers import event
from maubot.matrix import MaubotMessageEvent, parse_formatted
from maubot.plugin_base import Plugin
from mautrix.types import Format
from mautrix.types.event import message
from mautrix.types.event.type import EventType
from mautrix.util.async_db import Database, UpgradeTable
from mautrix.util.config import BaseProxyConfig, ConfigUpdateHelper

from .migrations import upgrade_table


class Config(BaseProxyConfig):
    def do_update(self, helper: ConfigUpdateHelper):
        helper.copy("settings")

    @property
    def settings(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], self.get("settings", {}))


class OrigamiChat(Plugin):
    config: Config
    database: Database
    name: str

    @classmethod
    def get_db_upgrade_table(cls) -> UpgradeTable:
        return upgrade_table

    @classmethod
    def get_config_class(cls) -> Type[BaseProxyConfig]:
        return Config

    async def start(self):
        self.log.info(f"Starting Origami Chat")
        await super().start()

        if not self.config:
            raise Exception("Config is not initialized")

        self.config.load_and_update()

        await self._cleanup_old_rate_instances()

        self.name = await self.client.get_displayname(self.client.mxid)

        self.command_name = str(self.config.settings["command_name"])

    @event.on(EventType.ROOM_MESSAGE)
    async def on_message(self, event: MaubotMessageEvent) -> None:
        if not event.content.msgtype.is_text or event.sender == self.client.mxid:
            return

        if self.name != self.config.settings["bot_name"]:
            return

        body = str(event.content.body)

        self.log.warning(
            f"body:{body}, command:{self.command_name}, prompt:{body[len(self.command_name) :].lstrip()}, startswith:{body.startswith(self.command_name)}"
        )

        if not body.startswith(self.command_name):
            return

        prompt = body[len(self.command_name) :].lstrip()

        # Check if the message is a reply to another message
        if event.content.relates_to and event.content.relates_to.in_reply_to:
            # Try to fetch the parent event
            try:
                parent_event = await self.client.get_event(
                    room_id=event.room_id,
                    event_id=event.content.get_reply_to(),
                )
                if parent_event and parent_event.content.msgtype.is_text:
                    # Append the parent message in quotes
                    quoted_text = f'\n"{parent_event.content.body}"'
                    prompt = f"{prompt}{quoted_text}"
            except Exception as e:
                self.log.warning(f"Could not fetch parent event: {e}")

        # Validate prompt presence
        if not prompt.strip():
            await self.send_message(
                event=event,
                content_str=self.config.settings["no_input_message"],
                reply=self.config.settings["reply"],
            )
            return

        # Enforce character limit
        if (
            len(prompt) > self.config.settings["input_character_limit"]
            and self.config.settings["enable_input_character_limit"]
        ):
            await self.send_message(
                event=event,
                content_str=self.config.settings["character_limit_message"],
                reply=self.config.settings["reply"],
            )
            return

        await self.client.send_receipt(
            room_id=event.room_id,
            event_id=event.event_id,
            receipt_type="m.read",
        )

        user_id = event.sender
        now_utc = datetime.now(timezone.utc)
        day_ago = (now_utc - timedelta(hours=24)).replace(tzinfo=None)

        # Rate limit checks
        if self.config.settings["enable_user_rate_limit"]:
            user_allowed = await self._check_user_rate_limit(
                user_id=user_id,
                interval_hours=self.config.settings["user_rate_limit_interval"],
                limit=self.config.settings["user_rate_limit"],
            )
            if not user_allowed:
                await self.send_message(
                    event=event,
                    content_str=self.config.settings["user_rate_limit_message"],
                    reply=self.config.settings["reply"],
                )
                return

        if self.config.settings["enable_global_rate_limit"]:
            global_allowed = await self._check_global_rate_limit(
                interval_hours=self.config.settings["global_rate_limit_interval"],
                limit=self.config.settings["global_rate_limit"],
            )
            if not global_allowed:
                await self.send_message(
                    event=event,
                    content_str=self.config.settings["global_rate_limit_message"],
                    reply=self.config.settings["reply"],
                )
                return

        await self.client.set_typing(
            room_id=event.room_id,
            timeout=300000,
        )

        payload = {
            "model": self.config.settings["model"],
            "messages": [
                {"role": "system", "content": self.config.settings["system_prompt"]},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.settings["temperature"],
            "max_tokens": self.config.settings["max_tokens"],
        }
        headers = {
            "Authorization": f"Bearer {self.config.settings['api_key']}",
            "Content-Type": "application/json",
        }
        try:
            start_time = datetime.now(timezone.utc)
            async with self.http.post(
                self.config.settings["endpoint"],
                headers=headers,
                json=payload,
            ) as response:
                end_time = datetime.now(timezone.utc)
                elapsed_time = (end_time - start_time).total_seconds()
                self.log.info(
                    f"Request completed in {elapsed_time:.2f} seconds for user {user_id}"
                )
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
                        reply=self.config.settings["reply"],
                    )
                else:
                    self.log.warning(
                        f"API request failed. Status: {response.status}, "
                        f"Body: {await response.text()} - Time taken: {elapsed_time:.2f} seconds"
                    )
                    await self.send_message(
                        event=event,
                        content_str=self.config.settings["error_message"],
                        reply=self.config.settings["reply"],
                    )
        except Exception as e:
            self.log.exception(f"Exception while calling API: {e}")

        finally:
            await self.client.set_typing(room_id=event.room_id, timeout=0)

    async def _check_user_rate_limit(
        self, user_id: str, interval_hours: int, limit: int
    ) -> bool:
        since = (datetime.now(timezone.utc) - timedelta(hours=interval_hours)).replace(
            tzinfo=None
        )
        count = await self.database.fetchval(
            "SELECT COUNT(*) FROM usage_log WHERE user_id = $1 AND event_ts > $2",
            user_id,
            since,
        )
        return count < limit

    async def _check_global_rate_limit(self, interval_hours: int, limit: int) -> bool:
        since = (datetime.now(timezone.utc) - timedelta(hours=interval_hours)).replace(
            tzinfo=None
        )
        count = await self.database.fetchval(
            "SELECT COUNT(*) FROM usage_log WHERE event_ts > $1", since
        )
        return count < limit

    async def _log_inference(self, user_id: str) -> None:
        await self.database.execute(
            "INSERT INTO usage_log (user_id) VALUES ($1)", user_id
        )

    async def _cleanup_old_rate_instances(self) -> None:
        try:
            interval_hours = max(
                self.config.settings["user_rate_limit_interval"],
                self.config.settings["global_rate_limit_interval"],
            )
            oldest_allowed = (
                datetime.now(timezone.utc) - timedelta(hours=interval_hours)
            ).replace(tzinfo=None)

            delete_query = """
                DELETE FROM usage_log
                WHERE event_ts <= $1
            """
            self.log.info(
                f"Cleaning up rate instances older than {interval_hours} hours..."
            )
            deleted_count = await self.database.execute(delete_query, oldest_allowed)
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
