"""Move the mouse pointer in a way that works on KDE Wayland.

``pyautogui.moveTo`` moves the cursor with the X11 XTest extension. On KDE
Wayland (through XWayland) XTest *motion* events are dropped, so ``moveTo``
silently does nothing and the following click ends up wherever the physical
mouse happens to be. That makes the bot right click the wrong spot.

The core X11 ``XWarpPointer`` request *is* honoured by the compositor, so use
that for movement and let pyautogui (XTest button events, which do work) do the
clicking.
"""

import time

import pyautogui

try:
    from Xlib import display as _xdisplay
except ImportError:  # pragma: no cover - X11 bindings missing
    _xdisplay = None

_display = None


def _get_display():
    global _display
    if _display is None and _xdisplay is not None:
        try:
            _display = _xdisplay.Display()
        except Exception:
            _display = False
    return _display or None


def move_to(x, y):
    """Move the pointer to absolute screen coordinates (pyautogui's space)."""
    x, y = int(round(x)), int(round(y))
    d = _get_display()
    if d is None:
        pyautogui.moveTo(x, y)
        return
    d.screen().root.warp_pointer(x, y)
    d.sync()


def click(x, y, button="left"):
    """Move to (x, y) and click there."""
    move_to(x, y)
    time.sleep(0.05)  # let the compositor/game see the new pointer position
    pyautogui.click(button=button)


def right_click(x, y):
    move_to(x, y)
    time.sleep(0.05)
    pyautogui.rightClick()
