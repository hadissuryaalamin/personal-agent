"""A tray icon, so a headless agent is visible without opening Task Manager.

The agent starts from a scheduled task under pythonw.exe: no console, no
window, nothing. Whether it is running, loading, or dead has until now been a
question you answered with Task Manager or scripts/status.ps1. A dot in the
notification area answers it at a glance, and answers the more useful question
too -- whether it is actually listening right now.

NOTHING HERE MAY TAKE THE AGENT DOWN WITH IT

A cosmetic feature must not be able to break a working voice assistant, so
every entry point swallows its own exceptions and every function is a no-op
when the tray is unavailable. Missing pystray, a locked notification area, a
failed icon draw: all of them log once and leave the agent running.

QUITTING

The menu's Quit raises KeyboardInterrupt in the main thread, which is what
agent.main already catches around backend.run() -- so the shutdown path is the
same one Ctrl+C has always used, rather than a second one written for the
tray. If that does not land within a few seconds (a hotkey backend blocked in
native code cannot be interrupted), a watchdog exits the process outright.
"""

from __future__ import annotations

import _thread
import logging
import os
import threading
import time

from . import config

log = logging.getLogger(__name__)

# state -> (colour, what the tooltip says)
STATES = {
    "starting": ((150, 150, 150), "starting up"),
    "idle":      ((80, 170, 90), "ready"),
    "recording": ((220, 60, 60), "listening"),
    "thinking": ((230, 160, 40), "thinking"),
    "speaking": ((70, 130, 220), "speaking"),
}

_icon = None
_state = "starting"
_failed = False


def _image(colour):
    """A filled circle. Drawn rather than shipped as a .ico so the colour can
    carry the state -- one asset per state would be four binaries to keep in
    step with this file."""
    from PIL import Image, ImageDraw

    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, size - 4, size - 4), fill=colour + (255,))
    return img


def _tooltip() -> str:
    # Plain hyphen, not an em dash: this string reaches a Win32 tooltip and a
    # cp1252 console, and one of them mangles the nicer character.
    return f"personal-agent - {STATES[_state][1]}"


def state(name: str) -> None:
    """Move the icon to a new state. Safe to call from any thread, and safe to
    call when there is no tray at all."""
    global _state
    if name not in STATES:
        return
    _state = name
    if _icon is None:
        return
    try:
        _icon.icon = _image(STATES[name][0])
        _icon.title = _tooltip()
    except Exception:
        log.debug("tray: could not update the icon", exc_info=True)


def _open_log() -> None:
    try:
        if config.LOG_FILE.exists():
            os.startfile(config.LOG_FILE)  # the user's own log, on their machine
    except Exception:
        log.debug("tray: could not open the log", exc_info=True)


def _quit() -> None:
    log.info("tray: quit requested")
    stop()

    def watchdog():
        # KeyboardInterrupt cannot interrupt a thread parked in native code, and
        # the hotkey listeners are. Give the clean path a moment, then leave.
        time.sleep(3)
        log.warning("tray: clean shutdown did not land, exiting hard")
        os._exit(0)

    threading.Thread(target=watchdog, name="tray-watchdog", daemon=True).start()
    _thread.interrupt_main()


def start() -> bool:
    """Put the icon in the notification area. Returns whether it worked."""
    global _icon, _failed

    if not getattr(config, "TRAY_ENABLED", True):
        return False
    if _icon is not None or _failed:
        return _icon is not None

    try:
        import pystray
    except ImportError:
        _failed = True
        log.info("tray: pystray is not installed, running without an icon")
        return False

    try:
        menu = pystray.Menu(
            pystray.MenuItem(lambda _: _tooltip(), lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Open log", lambda: _open_log()),
            pystray.MenuItem("Quit", lambda: _quit()),
        )
        _icon = pystray.Icon("personal-agent", _image(STATES[_state][0]),
                             _tooltip(), menu)
        # run() owns a message loop, so it gets its own thread; the agent's main
        # thread stays on the hotkey listener where it already was.
        threading.Thread(target=_icon.run, name="tray", daemon=True).start()
        log.info("tray: icon started")
        return True
    except Exception:
        _failed = True
        _icon = None
        log.warning("tray: could not start, running without an icon", exc_info=True)
        return False


def stop() -> None:
    global _icon
    if _icon is None:
        return
    try:
        _icon.stop()
    except Exception:
        log.debug("tray: could not stop cleanly", exc_info=True)
    _icon = None


if __name__ == "__main__":
    #   python -m src.tray      cycle through every state, 2 s each
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not start():
        raise SystemExit("tray did not start")
    try:
        for name in ("starting", "idle", "recording", "thinking", "speaking"):
            state(name)
            print(f"  {name:<10} {_tooltip()}")
            time.sleep(2)
        print("\n  cycled once; Ctrl+C to finish")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop()
