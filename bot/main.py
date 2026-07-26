"""Discord reaction-labeling bot — gateway wiring around bot/labeler.py.

Posts a fresh webcam capture to one Discord channel on an hourly daylight
schedule, seeds the 👍/⛅/👎 label reactions, and records human reactions into
the same labels.yaml the classifier UI and `training batch` already share.
Reactions on the Worker's transition notifications count too — the Worker
persists the announced frame and footers its capture key — so a 👎 on a false
"the mountain is out!" is a training label against the exact offending frame.
Reactions added while the bot was down are recovered by a startup sweep of
recent channel history.

Requires the `bot` dependency group:  uv run --group bot bot run
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import click
import discord
from discord.ext import tasks

from bot import labeler
from collect.collector import WeatherFetcher, perform_capture
from train.config_loader import ConfigLoader

logger = logging.getLogger("mountain.bot")


class MountainBot(discord.Client):
    """One channel, two jobs: post labelable captures, harvest reactions."""

    def __init__(
        self,
        settings: labeler.BotSettings,
        config_loader: ConfigLoader,
        data_root: str,
        post_once: bool = False,
    ):
        # Default (non-privileged) intents cover guilds + reactions. The
        # message_content intent is NOT needed: the bot only reads embeds of
        # its own messages, which Discord always delivers in full.
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.config_loader = config_loader
        self.data_root = data_root
        self.post_once = post_once
        # Raw backend, never the training cache — a stale cached labels.yaml
        # as a merge base would drop labels added to R2 by other surfaces.
        self.storage = labeler.uncached_storage(config_loader.get_storage(data_root))
        self.weather = WeatherFetcher(config_loader.metar_station)
        self._last_post: datetime | None = None
        self._swept = False

    async def setup_hook(self) -> None:
        if not self.post_once:
            self.post_loop.start()

    async def on_ready(self) -> None:
        logger.info(
            f"Logged in as {self.user}; labeling channel {self.settings.channel_id}"
        )
        if self.post_once:
            try:
                posted = await self.post_capture()
                logger.info("post-once %s", "complete" if posted else "failed")
            finally:
                await self.close()
            return
        if not self._swept:  # on_ready re-fires on reconnect; sweep only once
            self._swept = True
            try:
                await self.sweep_missed_reactions()
            except Exception:
                logger.exception("startup reaction sweep failed (continuing)")

    # ---------- Posting ----------

    @tasks.loop(seconds=60)
    async def post_loop(self) -> None:
        now = datetime.now(UTC)
        if not labeler.next_post_due(
            self._last_post, now, self.settings.post_interval_seconds
        ):
            return
        if not labeler.in_daylight_window(self.settings, now):
            return
        try:
            if await self.post_capture():
                self._last_post = now
        except Exception:
            logger.exception("capture post failed (loop continues)")

    @post_loop.before_loop
    async def _wait_until_ready(self) -> None:
        await self.wait_until_ready()

    def _capture(self) -> tuple[str, bytes, datetime] | None:
        """Blocking capture → (storage key, jpeg bytes, captured_at)."""
        remote = self.storage if self.config_loader.storage_backend == "r2" else None
        image_path = perform_capture(
            self.config_loader, self.weather, self.data_root, remote_storage=remote
        )
        if not image_path:
            return None
        key = str(Path(image_path).relative_to(self.data_root))
        return key, Path(image_path).read_bytes(), datetime.now(UTC)

    async def post_capture(self) -> bool:
        """Capture, verify the image is in storage, post it with seeded reactions."""
        capture = await asyncio.to_thread(self._capture)
        if capture is None:
            logger.warning("webcam capture failed; skipping this post")
            return False
        key, jpeg, captured_at = capture

        # Never post a message whose label would point at a missing object.
        if not await asyncio.to_thread(self.storage.exists, key):
            logger.warning(f"capture {key} not in storage; skipping post")
            return False

        prediction = await asyncio.to_thread(
            labeler.fetch_prediction, self.settings.state_url
        )
        embed = discord.Embed.from_dict(
            labeler.build_capture_embed(key, captured_at, prediction)
        )
        channel = await self._channel()
        message = await channel.send(
            embed=embed, file=discord.File(io.BytesIO(jpeg), filename="capture.jpg")
        )
        for emoji in labeler.EMOJI_TO_CLASS:
            await message.add_reaction(emoji)
        logger.info(f"posted capture {key} as message {message.id}")
        return True

    async def _channel(self) -> discord.abc.Messageable:
        channel = self.get_channel(self.settings.channel_id)
        if channel is None:
            channel = await self.fetch_channel(self.settings.channel_id)
        return channel

    # ---------- Reactions → labels ----------

    async def on_raw_reaction_add(
        self, payload: discord.RawReactionActionEvent
    ) -> None:
        if payload.channel_id != self.settings.channel_id:
            return
        if self.user and payload.user_id == self.user.id:
            return
        cls = labeler.class_for_emoji(str(payload.emoji))
        if cls is None:
            return
        is_bot = bool(payload.member and payload.member.bot)
        if not labeler.labeler_allowed(
            payload.user_id, is_bot, self.settings.allowed_user_ids
        ):
            return

        key = await self._capture_key_for_message(payload.message_id)
        if key is None:
            return
        if not await asyncio.to_thread(self.storage.exists, key):
            logger.warning(f"reaction on {key} ignored: capture not in storage")
            return
        await asyncio.to_thread(
            labeler.record_label, self.storage, key, cls, user_id=payload.user_id
        )
        logger.info(
            f"labeled {key} = {cls} ({labeler.CLASS_NAMES[cls]}) by user {payload.user_id}"
        )

    async def _capture_key_for_message(self, message_id: int) -> str | None:
        channel = await self._channel()
        try:
            message = await channel.fetch_message(message_id)
        except discord.HTTPException:
            return None
        # Labelable messages come from automation: the bot's own posts and the
        # Worker's transition notifications (webhook authors are bots too).
        # Human-authored embeds never count; the footer regex plus the caller's
        # storage-existence gate keep arbitrary keys out of labels.yaml.
        if not message.author.bot or not message.embeds:
            return None
        footer = message.embeds[0].footer
        return labeler.capture_key_from_footer(footer.text if footer else None)

    async def sweep_missed_reactions(self) -> None:
        """Recover labels from reactions added while the bot was offline."""
        channel = await self._channel()
        after = datetime.now(UTC) - timedelta(hours=self.settings.sweep_hours)
        current = await asyncio.to_thread(labeler.load_labels, self.storage)
        swept = 0
        async for message in channel.history(limit=500, after=after):
            # Same acceptance rule as _capture_key_for_message: any bot-authored
            # embed (own posts + Worker notifications) with a parsing footer.
            if not message.author.bot or not message.embeds:
                continue
            footer = message.embeds[0].footer
            key = labeler.capture_key_from_footer(footer.text if footer else None)
            if key is None:
                continue
            if not await asyncio.to_thread(self.storage.exists, key):
                continue
            reaction_users: dict[str, list[int]] = {}
            for reaction in message.reactions:
                emoji = str(reaction.emoji)
                if labeler.class_for_emoji(emoji) is None:
                    continue
                users = [u.id async for u in reaction.users() if not u.bot]
                if users:
                    reaction_users[emoji] = users
            label = labeler.label_from_reaction_users(
                reaction_users, self.settings.allowed_user_ids
            )
            if label is None or current.get(key) == label:
                continue
            await asyncio.to_thread(
                labeler.record_label,
                self.storage,
                key,
                label,
                user_id=0,
                source="discord-sweep",
            )
            current[key] = label
            swept += 1
        logger.info(f"reaction sweep complete: {swept} label(s) recovered")


# ---------- CLI ----------


def _build(config: str, data_root: str | None, post_once: bool = False) -> MountainBot:
    loader = ConfigLoader(config)
    root = data_root or os.environ.get("MOUNTAIN_DATA_ROOT", "data")
    try:
        settings = labeler.BotSettings.from_mapping(loader.data)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    return MountainBot(settings, loader, root, post_once=post_once)


@click.group()
def cli() -> None:
    """Discord reaction-labeling bot for is-the-mountain-out (see BOT.md)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@cli.command()
@click.option("--config", default="mountain.toml", help="Path to mountain.toml.")
@click.option(
    "--data-root", default=None, help="Defaults to $MOUNTAIN_DATA_ROOT or 'data'."
)
def run(config: str, data_root: str | None) -> None:
    """Run the bot: hourly daylight capture posts + reaction labeling."""
    bot = _build(config, data_root)
    bot.run(bot.settings.token, log_handler=None)


@cli.command("post-once")
@click.option("--config", default="mountain.toml", help="Path to mountain.toml.")
@click.option(
    "--data-root", default=None, help="Defaults to $MOUNTAIN_DATA_ROOT or 'data'."
)
def post_once(config: str, data_root: str | None) -> None:
    """Capture and post a single labelable message, then exit (setup check)."""
    bot = _build(config, data_root, post_once=True)
    bot.run(bot.settings.token, log_handler=None)


if __name__ == "__main__":
    cli()
