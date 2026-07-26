import sys
import types
from unittest.mock import MagicMock


def _make_rumps_stub():
    rumps_mod = types.ModuleType("rumps")

    class MenuItem:
        def __init__(self, title, callback=None):
            self.title = title
            self.callback = callback

    class Timer:
        def __init__(self, callback, interval):
            self.callback = callback
            self.interval = interval

        def start(self):
            pass

        def stop(self):
            pass

    class App:
        def __init__(self, name, title="", quit_button=None):
            self.name = name
            self.title = title
            self.menu = []

        def run(self):
            pass

    rumps_mod.App = App
    rumps_mod.MenuItem = MenuItem
    rumps_mod.Timer = Timer
    rumps_mod.separator = "---"
    rumps_mod.quit_application = MagicMock()
    return rumps_mod


# Apply mock globally for collect tests
if "rumps" not in sys.modules:
    sys.modules["rumps"] = _make_rumps_stub()
