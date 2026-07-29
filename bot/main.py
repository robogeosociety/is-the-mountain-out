"""Discord reaction-labeling bot — gateway wiring around bot/labeler.py.

Harvests 👍/⛅/👎 reactions on the Worker's visibility-change notifications into
the same labels.yaml the classifier UI and `training batch` already share, so a
👎 on a false "the mountain is out!" is a training label against the exact
offending frame. Reactions added while the bot was down are recovered by a
startup sweep of recent channel history.

The bot does NOT post on a schedule. It used to publish an hourly capture to
label, which produced 24 near-identical "Not Out (100.0%)" quiz posts a day —
noise that buried the notifications that actually mean something. The Worker's
state-change posts are now the whole labeling surface, and the Worker writes
them with THIS bot's token (worker/src/discord-mountain-notify.ts) so they are
bot-authored: without the privileged Message Content intent, Discord blanks the
embeds of messages any other author wrote, which is exactly why reactions on
the old webhook notifications were silently dropped. `bot post-once` remains as
a manual setup check.

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

from bot import labeler
from collect.collector import WeatherFetcher, perform_capture
from train.config_loader import ConfigLoader

logger = logging.getLogger("mountain.bot")


class MountainBot(discord.Client):
    """One channel, one job: turn reactions into training labels."""

    def __init__(
        self,
        settings: labeler.BotSettings,
        config_loader: ConfigLoader,
        data_root: str,
        post_once: bool = False,
    ):
        # Default (non-privileged) intents cover guilds + reactions. The
        # message_content intent is NOT needed *because* every labelable post is
        # written with this bot's token, and Discord always delivers a bot its
        # own messages in full. Anything authored elsewhere arrives with
        # `embeds: []`, which is precisely why the old webhook notifications
        # were unlabelable — keep that invariant when adding a posting surface.
        super().__init__(intents=discord.Intents.default())
        self.settings = settings
        self.config_loader = config_loader
        self.data_root = data_root
        self.post_once = post_once
        # Raw backend, never the training cache — a stale cached labels.yaml
        # as a merge base would drop labels added to R2 by other surfaces.
        self.storage = labeler.uncached_storage(config_loader.get_storage(data_root))
        self.weather = WeatherFetcher(config_loader.metar_station)
        self._swept = False

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

    # ---------- Posting (manual setup check only — see module docstring) ----------

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

        labelable = await self._labelable_message(payload.message_id)
        if labelable is None:
            return
        message, key = labelable
        if not await asyncio.to_thread(self.storage.exists, key):
            logger.warning(f"reaction on {key} ignored: capture not in storage")
            return
        result = await asyncio.to_thread(
            labeler.record_label, self.storage, key, cls, user_id=payload.user_id
        )
        logger.info(
            f"labeled {key} = {cls} ({labeler.CLASS_NAMES[cls]}) by user {payload.user_id}"
        )
        await self._edit_telemetry(message, cls, result)

    async def _edit_telemetry(
        self, message: discord.Message, cls: int, result: labeler.LabelResult
    ) -> None:
        """Acknowledge a recorded label on the post itself (edit-in-place).

        Discord only lets a bot edit its own messages — which now includes the
        Worker's notifications, since the Worker posts them with this bot's
        token. Best-effort: an edit failure never loses the recorded label.
        """
        if not self.user or message.author.id != self.user.id or not message.embeds:
            return
        try:
            updated = labeler.apply_telemetry(
                message.embeds[0].to_dict(), labeler.telemetry_field(cls, result)
            )
            await message.edit(embed=discord.Embed.from_dict(updated))
        except discord.HTTPException as exc:
            logger.warning(f"telemetry edit failed (label already recorded): {exc}")

    async def _labelable_message(
        self, message_id: int
    ) -> tuple[discord.Message, str] | None:
        channel = await self._channel()
        try:
            message = await channel.fetch_message(message_id)
        except discord.HTTPException:
            return None
        # Labelable messages come from automation: the Worker's visibility
        # notifications and `bot post-once` captures, both written with this
        # bot's token. Human-authored embeds never count; the footer regex plus
        # the caller's storage-existence gate keep arbitrary keys out of
        # labels.yaml. (`message.embeds` is only ever populated for messages
        # this bot authored — see the module docstring on Message Content.)
        if not message.author.bot or not message.embeds:
            return None
        footer = message.embeds[0].footer
        key = labeler.capture_key_from_footer(footer.text if footer else None)
        if key is None:
            return None
        return message, key

    async def sweep_missed_reactions(self) -> None:
        """Recover labels from reactions added while the bot was offline."""
        channel = await self._channel()
        after = datetime.now(UTC) - timedelta(hours=self.settings.sweep_hours)
        current = await asyncio.to_thread(labeler.load_labels, self.storage)
        swept = 0
        async for message in channel.history(limit=500, after=after):
            # Same acceptance rule as _labelable_message: a bot-authored embed
            # with a footer that parses as a capture key.
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
            result = await asyncio.to_thread(
                labeler.record_label,
                self.storage,
                key,
                label,
                user_id=0,
                source="discord-sweep",
            )
            current[key] = label
            swept += 1
            await self._edit_telemetry(message, label, result)
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
    """Run the bot: reaction labeling + startup sweep of missed reactions."""
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
