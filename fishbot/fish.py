import cv2
import numpy as np
import os
import time
import pyautogui
import atexit
import capture
import pointer


# ------ CONFIG --------#
game_size = {"width": 3200, "height": 1800}  # Game screen size
capture_region = 0.66  # From bottom up. 0.5 captures half bottom
throw_key = "1"  # Fish keybind
lure = False  # Enable auto application of lure (press lure_key every lure_interval)
lure_key = "2"  # Macro keybind for applying lure.
lure_interval = 11  # How often (in min) to apply lure
monitor = 1  # Monitor to capture (NOT IMPLEMENTED)
bobber_img = "zereth_mortis.png"  # Path to template of bobber
bobber_mask = None  # Path to template mask (must have same dimensions and nr of channels as template). Set = None to not use a mask
pyautogui.PAUSE = 1  # How long in seconds python will wait after a keystroke/mouse action
match_threshold = 0.55  # Bobber template match threshold. Adjust this if the bot has troubles finding the bobber. Higher = Better match
diff_threshold = 600  # Bobber img comparison threshhold. Adjust this if the bot clicks the bobber too soon or not at all. Higher = Bigger diff
fast_capture = True  # Use a live screen cast instead of a portal screenshot per grab (WAY faster)
capture_fps = 4  # How many frames per second the screen cast delivers
force_scale = 1.0  # Force stream->portal scale instead of auto-calibrating it. None = auto
click_point = None  # Where inside the matched bobber box to click, as (x, y) fractions of the
                    # box size, e.g. (0.35, 0.75). None = find the bobber float in the template
click_offset = (0, 0)  # Extra pixels added to the click position (use to nudge a constant offset)
# Debugging options. Use these to
debugging = False  # See what the bot is thinking of (Overrides all of the below to True)
log_region_val = False  # Print x,y,w,h of screen region which is searched for a match
show_match_img = True  # If a match is found show a brief image of where the match area is
log_match_val = True  # Print info about the match value to console (used to find the bobber in the image)
log_bobber_loc = False  # Print the coordinates to console after a match is found (x,y,w,h)
log_diff_val = True  # Print info about the diff value to console (used to tell when a fish is hooked)
save_screenshots = True  # Save the captures to screenshot_dir so you can inspect them
screenshot_dir = "screenshots"  # Folder the captures are written to (gitignored)
# ------------------------#


bobbers_clicked = 0
lures_used = 0
timeouts = 0
window = {
    "top": int(game_size["height"] * abs(capture_region - 1)),
    "left": 0,
    "width": int(game_size["width"]),
    "height": int(game_size["height"] * capture_region),
}  # Adjust region to config game_size & captuire region

game_window = {
    "x1": 0,
    "y1": int(game_size["height"] * abs(capture_region - 1)),
    "x2": int(game_size["width"]),
    "y2": int(game_size["height"]),
}  # Adjust region to config game_size & captuire region

# ----- SCREEN CAPTURE -----#
# capture.py serves the grabs. It prefers a persistent pipewire screen cast
# (a few ms per grab) and falls back to plain portal screenshots (~3 s per grab).
capture_backend = None

# Click position inside the matched bobber box, as fractions; set at startup
bobber_click_frac = (0.5, 0.5)

# ----- DEFINE FUNCTIONS ------#


# Start the capture backend (screen cast when possible)
def pick_backend():
    global capture_backend
    capture_backend = capture.create_capture(
        roi=game_window,
        fps=capture_fps,
        debug=debugging,
        use_screencast=fast_capture,
        force_scale=force_scale,
    )
    atexit.register(capture_backend.close)
    return capture_backend


# Grab a region of the screen straight into memory (no file is written)
def grab(region):
    if debugging or log_region_val:
        print(f"Grabbing region: {region}")
    return capture_backend.grab(region)


# Save an in-memory frame to disk (for debugging / building new templates)
def save_shot(img, name):
    if not save_screenshots:
        return
    os.makedirs(screenshot_dir, exist_ok=True)
    path = os.path.join(screenshot_dir, name)
    cv2.imwrite(path, img)
    if debugging:
        print(f"Saved screenshot: {path}")


# The matched box is the whole bobber: the float at the water surface plus the
# feather sticking up in the air, so the middle of the box is up in the air and
# not where the game wants the click. This finds the float inside the template.
def bobber_hotspot(template):
    img = template[:, :, :3] if template.ndim == 3 and template.shape[2] == 4 else template
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 130) & (hsv[:, :, 2] > 70)).astype(np.uint8)
    mask[: int(h * 0.5), :] = 0  # the float sits below the feather
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[best, cv2.CC_STAT_AREA] < 0.01 * w * h:
        return None
    return (float(centroids[best][0]) / w, float(centroids[best][1]) / h)


# Input position of wow screen and click once to make it the active window
def start_click(x, y):
    print("Starting fishing session")
    x, y = x + 100, y + 100  # top left corner + some pixels
    pointer.click(x, y)


def throw(key):
    print("Throwing...")
    pyautogui.press(key)
    # Let bobble settle in water
    time.sleep(1)


def apply_lure(key, lure_count, sleep=5):
    print("Applying lure...")
    pyautogui.press(key)
    lure_count += 1
    time.sleep(sleep)
    return lure_count


def click_bobber(loc):
    # Click the float inside the matched box (not the middle of the box)
    x, y = bobber_click_pos(loc)
    pointer.right_click(x, y)

    print(f"Got em!")

    pointer.move_to(100, 100)


def bobber_click_pos(loc):
    """Absolute screen position the bot will click for a matched bobber box."""
    x = loc[0] + int(round(loc[2] * bobber_click_frac[0])) + click_offset[0]
    y = loc[1] + int(round(loc[3] * bobber_click_frac[1])) + click_offset[1]
    return x, y


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    # return the MSE, the lower the error, the more "similar"
    return err


# Load the bobber template once, keep it in memory
def load_bobber(path):
    bobber = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bobber is None:
        raise FileNotFoundError(f"Could not read bobber template: {path}")
    return bobber


# Load the optional match mask once, keep it in memory
def load_mask(path):
    if not path:
        return None
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read bobber mask: {path}")
    return mask


# matchTemplate requires source and template to have the same nr of channels
def _match_channels(source, template):
    if template.ndim == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    t_channels = template.shape[2]
    s_channels = source.shape[2]
    if t_channels == 4 and s_channels == 3:
        source = cv2.cvtColor(source, cv2.COLOR_BGR2BGRA)
    elif t_channels == 3 and s_channels == 4:
        source = source[:, :, :3]
    return source, template


# Find the bobber in an already-grabbed (in memory) source image
def find_bobber(source, bobber, mask=None, show=True):
    source, bobber = _match_channels(source, bobber)
    h = bobber.shape[0]
    w = bobber.shape[1]
    # template match
    if mask is not None:
        match = cv2.matchTemplate(source, bobber, cv2.TM_CCORR_NORMED, mask=mask)
    else:
        match = cv2.matchTemplate(source, bobber, cv2.TM_CCOEFF_NORMED)
    # Get coordinates and value of best/worst match
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)
    # If isn't good enough
    if max_val < match_threshold:
        if debugging or log_match_val:
            print(f"Match not good enough: {round(max_val,2)} / {match_threshold}")
        else:
            print("I can't see the bobber, trying again...")
        return False
    # Rectangle coordinates
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    if (
        debugging or log_match_val
    ):  # Print info about the match value to console
        print(f"Match found: {round(max_val, 2)} / {match_threshold}")
    else:
        print("I think I see the bobber!")
    if (debugging or show_match_img) and show:  # Show picture of region with rectangle around the match
        preview = source[:, :, :3].copy()
        cv2.rectangle(preview, top_left, bottom_right, (0, 255, 0), 2)
        cv2.drawMarker(
            preview,
            (
                top_left[0] + int(round(w * bobber_click_frac[0])) + click_offset[0],
                top_left[1] + int(round(h * bobber_click_frac[1])) + click_offset[1],
            ),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            2,
        )
        cv2.imshow("Fish", preview)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

    # Adjust coordinates to take into account entire screen (not just the region of screen capture)
    x = top_left[0] + window["left"]
    y = top_left[1] + window["top"]

    return (x, y, w, h)


# Grab first image of bobber then keep comparing it to a new one. If difference is above threshold break out (and right click)
def watch_bobber(rect):
    dict_rect = {"x1": rect[0], "y1": rect[1], "x2": rect[0] + rect[2], "y2": rect[1] + rect[3]}
    if debugging or log_bobber_loc:
        print(f"Bobber coordinates: {rect}")
    nothooked = cv2.cvtColor(grab(dict_rect), cv2.COLOR_BGR2GRAY)
    save_shot(nothooked, "bobber_nothooked.png")

    # Grab a new image every 0.5s and compare it to original
    print("Waiting for fish...")
    diff_list = []
    for i in range(50):
        hooked = cv2.cvtColor(grab(dict_rect), cv2.COLOR_BGR2GRAY)
        diff = mse(nothooked, hooked)
        diff_list.append(diff)
        # Probably hooked
        if diff > diff_threshold:
            save_shot(hooked, "bobber_hooked.png")
            if debugging or log_diff_val:
                try:
                    print(
                        f"""\nWent above at: {max(diff_list)}\nThreshold: {diff_threshold}\nAvg diff: {sum(diff_list[1:-1])/(len(diff_list)-2)}\nMax diff: {max(diff_list)}\nMin diff: {min(diff_list)}\n-------"""
                    )
                except ZeroDivisionError:
                    print("Couldn't print the diff_threshold debug values")
            return True
        else:
            time.sleep(0.5)
    save_shot(hooked, "bobber_timeout.png")
    if debugging or log_diff_val:
        print("Timed out. Match was false or threshold to high")
        print(
            f"""Threshold: {diff_threshold}\nAvg diff: {sum(diff_list[1:]) / (len(diff_list) - 1)}\nMax diff: {max(diff_list)}\nMin diff: {min(diff_list[1:])}"""
        )
    return False


if __name__ == "__main__":
    bobber = load_bobber(bobber_img)  # cached in memory for the whole session
    mask = load_mask(bobber_mask)
    if click_point:
        bobber_click_frac = (float(click_point[0]), float(click_point[1]))
        print(f"Bobber click point (config): {bobber_click_frac[0]*100:.0f}% "
              f"/ {bobber_click_frac[1]*100:.0f}% of the matched box")
    else:
        frac = bobber_hotspot(bobber)
        if frac:
            bobber_click_frac = frac
            print(f"Bobber click point (auto): {frac[0]*100:.0f}% / {frac[1]*100:.0f}% "
                  f"of the matched box")
        else:
            print("Could not locate the bobber float in the template, clicking the box centre")
    pick_backend()
    start_click(window["top"], window["left"])
    start = time.time()
    if lure:
        lures_used = apply_lure(lure_key, lures_used)
    while True:
        try:
            timer = time.time() - start
            hours = int(timer // 3600)
            minutes = int(timer // 60 - hours * 60)
            seconds = int(timer - hours * 3600 - minutes * 60)
            print("\nSession:")
            if timer > 3600:
                print(f"Fished for {hours}h {minutes}m {seconds}s")
            elif timer > 60:
                print(f"Fished for {minutes}m {seconds}s")
            else:
                print(f"Fished for {seconds}s")
            print(f"Bobbers clicked: {bobbers_clicked}")
            print(f"Lures used: {lures_used}")
            print(f"Timeouts: {timeouts}")
            if lure:
                if timer + 6 > lures_used * lure_interval * 60:
                    lures_used = apply_lure(lure_key, lures_used)
            throw(throw_key)
            frame = grab(game_window)
            save_shot(frame, f"fishtemp_{frame.shape[1]}x{frame.shape[0]}.png")
            bobber_info = find_bobber(frame, bobber, mask)
            if not bobber_info:  # Match below threshold
                continue
            if watch_bobber(bobber_info):
                # The bobber drifts and bobs with the water, and the box we found
                # was from several seconds ago. Look once more right before clicking
                # (cheap now) and use the fresh position if the match still holds.
                fresh = find_bobber(grab(game_window), bobber, mask, show=False)
                click_bobber(fresh or bobber_info)
                bobbers_clicked += 1

            else:
                print("Exiting loop...\n")
                timeouts += 1
                continue
        except OSError as err:
            print(f"OSError: {err}")
        except capture.CaptureError as err:
            print(f"Capture problem ({err}), restarting the capture backend...")
            pick_backend()
        except pyautogui.FailSafeException as err:
            print(f"Mouse moved outside monitor: {err}")
