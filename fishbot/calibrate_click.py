"""Check where the bot clicks, by comparing it with where *you* say the bobber is.

Run it while a bobber is in the water (from the fishbot folder):

    python calibrate_click.py

It captures a frame, finds the bobber exactly like the bot does and prints the
position it would click. Then you put the mouse on the bobber float yourself,
press Enter, and it prints the difference. That tells you (and me) whether the
bot's idea of "on the bobber" matches yours, or whether the cursor ends up
somewhere else on screen.
"""

import os
import sys

import cv2
import pyautogui

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fish  # noqa: E402
import pointer  # noqa: E402


def main():
    fish.show_match_img = False
    bobber = fish.load_bobber(fish.bobber_img)
    frac = fish.click_point or fish.bobber_hotspot(bobber) or (0.5, 0.5)
    fish.bobber_click_frac = (float(frac[0]), float(frac[1]))
    print(f"click point inside the matched box: {frac[0] * 100:.0f}% / {frac[1] * 100:.0f}%")
    print(f"click offset: {fish.click_offset}")

    fish.pick_backend()
    frame = fish.grab(fish.game_window)
    info = fish.find_bobber(frame, bobber, None)
    if not info:
        print("\nNo bobber found in the capture.")
        print("Cast first, wait until the bobber is in the water and run it again.")
        fish.capture_backend.close()
        return

    x, y, w, h = info
    bot_x, bot_y = fish.bobber_click_pos(info)
    print(f"\nmatch box (x, y, w, h) : {info}")
    print(f"box centre would be    : ({x + w // 2}, {y + h // 2})")
    print(f"bot would click at     : ({bot_x}, {bot_y})")

    input("\nNow move YOUR mouse onto the bobber float and press Enter ...")
    you_x, you_y = pyautogui.position()
    print(f"your position          : ({you_x}, {you_y})")
    print(f"difference (bot - you) : ({bot_x - you_x:+d}, {bot_y - you_y:+d})")

    vis = frame.copy()
    rx, ry = x - fish.window["left"], y - fish.window["top"]
    cv2.rectangle(vis, (rx, ry), (rx + w, ry + h), (0, 255, 0), 2)
    cv2.drawMarker(
        vis,
        (bot_x - fish.window["left"], bot_y - fish.window["top"]),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        30,
        2,
    )
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "click_check.png")
    cv2.imwrite(out, vis)
    print(f"annotated capture saved to {out}")

    input("\nPress Enter to let the bot move the cursor to its click point (no click) ...")
    pointer.move_to(bot_x, bot_y)
    print(
        "The cursor is now where the bot would click.\n"
        "If it does not sit on the bobber, note how far off it is - that is the\n"
        "number that needs to go into click_offset (or click_point) in fish.py."
    )
    fish.capture_backend.close()


if __name__ == "__main__":
    main()
