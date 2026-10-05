# Wow Fishbot - Linux / KDE edition

> **This script is made for Linux with the KDE Plasma desktop.** It was written
> and tested on **KDE Plasma Wayland (KWin)**. Screen capture goes through the
> XDG desktop portal with a pipewire screen cast (`capture.py`) and the pointer
> is moved with `XWarpPointer` (`pointer.py`), both because the plain X11 and
> Wayland tooling does not work there. Windows and macOS are not supported.

> **This fork is AI-developed.** The Linux/KDE support, the bobber click/diff
> logic, the calibration helper and this documentation were written in an
> AI-assisted session (OpenAI Codex). The upstream project is by
> Snacks-Razorgore. Read the code before you run it and use it at your own risk.


## Introduction
The bot utilizes opencv template matching. This means it searches for an image in an image. Read more here: https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html.

It takes a screenshot of your game window and searches for the bobber (template) in it.
If it finds the bobber it takes a screenshot of that specific area and compares it to a new image every 0.2sec.
If the difference is too big (assumed fish is hooked) it moves the mouse with `XWarpPointer` and right clicks the bobber.

This script only interacts with the OS and not the game itself by sending keystrokes and mouseclicks.


## Requirements
- Linux with the **KDE Plasma** desktop. Developed and tested on KDE Plasma Wayland (KWin); other compositors are untested.
- Python 3 installed
- `gstreamer` (needs `gst-launch-1.0` in `PATH`) and `pipewire` for the screen cast. Without them the bot falls back to the slow XDG portal screenshots.
- `kscreen-doctor` (part of KDE) so the bot can map the screen cast onto your screen coordinates
- Some command line knowledge to run the script
- A text editor to tweak the settings. VSCode with the Python plugin gives you syntax highlighting, but any editor works.

### Python dependencies
`pip install -r requirements.txt` installs these:

| Package | Import | What it is used for |
| --- | --- | --- |
| `opencv-python` | `cv2` | Template matching to find the bobber, the diff maths (grayscale, blur, crop), writing the debug PNGs and the match preview window. Used by `fish.py`, `capture.py` and `calibrate_click.py`. |
| `numpy` | `np` | The pixel arrays everything else operates on; the bobber diff is a numpy expression. |
| `PyAutoGUI` | `pyautogui` | Sends the cast/lure keys and the mouse button events, and provides the global action pause (`pyautogui.PAUSE`). The pointer *movement* itself goes through `pointer.py`. |
| `pyscreenshot` | `pyscreenshot` | Screenshot backend for the fallback path, used only when the pipewire screen cast is unavailable (`capture.py` imports it as `ImageGrab`). This is the slow 3-4s-per-grab mode. |
| `Pillow` | `PIL` | Image handling for pyscreenshot (its `pil` backend). Not imported by the bot directly. |
| `jeepney` | `jeepney` | Low-level DBus, used to talk to the XDG ScreenCast portal that starts the pipewire screen cast (`capture.py`). Linux only; without it the bot falls back to pyscreenshot. |
| `python-xlib` | `Xlib` | `XWarpPointer` in `pointer.py`, because KWin drops XTest pointer motion events coming from XWayland. Linux only. |

The system packages the screen cast needs (**gstreamer**/`gst-launch-1.0`, **pipewire**, **kscreen-doctor**) come from your distro, not from pip - see Requirements above.

## Installation
1. In the command line cd into location where you cloned the repo
2. (Optional) Create a python venv and activate it.
3. ´pip install -r requirements.txt´
4. cd into /fishbot

## Screen capture on KDE Wayland
On Wayland the desktop portal only offers a full screen shot per request, and each of those takes ~3-4 seconds (the portal takes the shot, encodes a PNG and writes it to disk). That is far too slow for the fishing loop.

Instead of doing that on every grab, the bot opens a screen cast once (`ScreenCast` portal + pipewire) and reads live frames from it with gstreamer. A grab then costs a few milliseconds. The first start asks for permission to share a screen - **pick the monitor that has the game on it**. The calibration is stored in `fishbot/.screencast_cache.json`, so later starts are instant. Delete that file (and `fishbot/.screencast_thumb.png`) if you moved the game to another monitor or changed the display scaling.

Mouse movement uses `XWarpPointer` (see `pointer.py`) instead of `pyautogui.moveTo`: KWin drops XTest pointer *motion* events coming from XWayland, so with plain pyautogui the cursor never actually moved and every click landed wherever the physical mouse happened to be.

Relevant settings in `fish.py`:
- **fast_capture**: `True` (default) uses the screen cast, `False` forces the old one-screenshot-per-grab behaviour.
- **capture_fps**: How many fresh frames per second come off the screen cast (default 8). Raise it if bites get missed, lower it if you want less CPU load. Keep it above the poll rate (5/s) or the same frame gets compared twice.
- **force_scale**: Overrides the auto-calibrated stream->portal scale. `None` (default) uses the value worked out from the screen geometry. Set a float, e.g. `1.0`, to pin it if the automatic calibration gets it wrong.
- **click_point**: Where inside the matched bobber box the bot clicks, as `(x, y)` fractions of the box size. `None` (default) finds the bobber float in the template automatically, because the matched box also contains the feather sticking up in the air - the middle of that box is not on the bobber. Set it manually, e.g. `(0.35, 0.75)`, if the auto detection is off for your template.
- **click_offset**: Extra pixels added to the click position, e.g. `(0, -6)` to nudge the click up. Use it if the cursor lands slightly off the bobber.
- **show_match_img**: While `True` the preview draws the match box in green and the position the bot will actually click in red.
- **diff_threshold**, **diff_confirm**, **diff_warmup**: A bite is only accepted when the diff beats `diff_threshold` for `diff_confirm` frames in a row, and never during the first `diff_warmup` frames (the background is still settling there). The bot prints the diff while it approaches the threshold and appends every watch to `diff_log.csv`, so you can tune these from real runs.
- **diff_background**, **diff_bg_alpha**: Compare against a slowly adapting background instead of the first frame (`accumulateWeighted`), which cancels the slow water/lighting drift. `diff_bg_alpha` is the adaptation rate; a real bite lasts longer than the background can follow, so it still stands out.
- **diff_focus**, **diff_focus_xy**, **diff_blur**: Weight the diff towards the float with a Gaussian at `diff_focus_xy` (box fractions, default bottom centre) instead of averaging the whole box, and blur first. Without this the feather/leaves/water dominate and the bite only shows up as a slow, ambiguous climb. `diff_focus = 0` compares the whole box evenly.
- **diff_log**, **diff_log_file**: Append one row per watch (result, threshold, Min/Avg/Max and the raw diff series) to a CSV so you can tune the thresholds offline. Set `diff_log = False` to turn it off.

`python calibrate_click.py` (from the `fishbot` folder, with a bobber in the water) prints where the bot would click, asks you to put your own mouse on the bobber, prints the difference and then moves the cursor to the bot's click point so you can see the real offset. Use it to work out the numbers for `click_point` / `click_offset`.

The screen cast is mapped onto the bot's screen coordinates from the monitor geometry (`kscreen-doctor`), so an animated game screen cannot throw the alignment off, and no extra screenshot is needed on start up.


## Script config
### Don't touch:
If you just want it to work out of the box, no need to change these. Just follow the instructions
- **game_size**: Should be set to your in game resolution. In theory any should work but only tested on 3200x1800.
- **capture_region**: How big in % from the bottom of the game window and up should be scanned for a bobber (0.66 i.e 2/3 seem to work well).
- **bobber_mask**: None or path to the template mask if you use one.
- **pyautogui.PAUSE**: How long in seconds python will wait after a keystroke/mouse action.
- **monitor**: NOT IMPLEMENTED. Intended for if you have more than 1 monitor.

### Game
- **throw_key**: The key on your actionbar for casting.
- **lure**: If a lure should be applied at the start and at the end of every *lure_interval*.
- **lure_key**: The key on your actionbar for applying lure.
- **lure_interval**: How many minutes before applying a new lure.


### Debugging
I'd recommend leaving ´log_match_val´ on. But rest can be useful for troubleshooting. ´show_match_image´ is useful to see if you get a match on something that isn't the bobber which indicates the ´match_threshold´ needs tweaking. But it's also very satisfying to leave on just to see when it detects it.
- **debugging** Sets all of the below to True
- **log_region_val** Print x,y,w,h of screen region which is searched for a match
- **show_match_img** If a match is found show a brief image of where the match area is
- **log_match_val** Print info about the match value to console (used to find the bobber in the image)
- **log_bobber_loc** Print the coordinates to console after a match is found (x,y,w,h)
- **log_diff_val** Print info about the diff value to console (used to tell when a fish is hooked)

### Do tweak
Change ´bobber_img´ when you swap to a new spot. The thresholds generally need to be set once per spot and that's it.
- **bobber_img**: Path to the image template which will be used to look for a match. Every time you change spot, this should be updated for best results.
- **match_threshold**: Is pretty good at .5 but can need a nudge up or down if you don't find the bobber. Depends on the spot
- **diff_threshold**: Needs a nudge per resolution/spot. Tune it from the diff values the bot prints and the runs it writes to `diff_log.csv` (see the diff_* settings above).


## Instructions:
1. Set the game in windowed mode and change resolution to the value you set in ´game_size´ (default 3200x1800)
2. Move the window to the top left corner of the display
3. In the action bars bind the cast key to 1, and lure key to 2 (or what you changed the script settings to)
4. In the game hide UI (Alt+Z usually) and zoom the camera to first person PoV.
5. Throw a cast **manually** and take a region screenshot of **only the bobber**
6. Save it in the same folder as the script with a .png extension (E.g "terokkar.png")
7. Change the variable ´bobber_img´ to the file name (E.g "terokkar.png")
8. Start the script
9. Monitor for a bit, if it works then congratz. Sit back and relax :)
10. If it doesn't yield you any fish, stop the script with Ctrl+C
11. Read over the *Game* and *Debugging* section of **Script config** to get an idea of what you need to change. Then check symptom below
- **Problem**: It's re-casting to much
   - **Solution**: Decrease the ´match_threshold´ variable

- **Problem**: It doesn't respond when a fish is on the hook
   - **Solution**: This can depend on 2 things. Either it think it found the bobber while it didn't, and is watching the wrong spot, or the ´diff_threshold´ is to high. To determine which, turn on ´show_match_img´ and see if it draws a rectangle around the bobber. If it does, the threshold is to high. If it draws the rectangle on the wrong spot (i.e not around the bobber), the ´match_threshold´ is too low.

## Using a mask
With a mask you don't have to take a new screenshot whenever you change to a new spot. It kind of worked but the success rate was much lower. But leaving it here in case someone wants to tinker with it. To use a mask set ´bobber_mask´ to the name of the mask file. Read more on template masks here:  https://gregorkovalcik.github.io/opencv_contrib/tutorial_template_matching.html

## Ideas I never got around to do:
* Add a keyboard listener to issue commands (pause/start, abort, apply lure off intervall)
* Train a haar cascade model of the bobber (will drastically increase bobber detection)
* Shutdown timer (HS and log out after X minutes)
* Print remaining lure timer
* Jump, dance, use a toy or go afk for a few minutes at random intervalls to not make it seem like your botting


Enjoy!
