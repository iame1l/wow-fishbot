import cv2
import numpy as np
import pyscreenshot as ImageGrab
import time
import pyautogui


# ------ CONFIG --------#
game_size = {"width": 1280, "height": 720}  # Game screen size
capture_region = 0.66  # From bottom up. 0.5 captures half bottom
throw_key = "1"  # Fish keybind
lure = True  # Enable auto application of lure (press lure_key every lure_interval)
lure_key = "2"  # Macro keybind for applying lure.
lure_interval = 11  # How often (in min) to apply lure
monitor = 1  # Monitor to capture (NOT IMPLEMENTED)
bobber_img = "zereth_mortis.png"  # Path to template of bobber
bobber_mask = None  # Path to template mask (must have same dimensions and nr of channels as template). Set = None to not use a mask
pyautogui.PAUSE = 1  # How long in seconds python will wait after a keystroke/mouse action
match_threshold = 0.6  # Bobber template match threshold. Adjust this if the bot has troubles finding the bobber. Higher = Better match
diff_threshold = 1200  # Bobber img comparison threshhold. Adjust this if the bot clicks the bobber too soon or not at all. Higher = Bigger diff
# Debugging options. Use these to
debugging = False  # See what the bot is thinking of (Overrides all of the below to True)
log_region_val = False  # Print x,y,w,h of screen region which is searched for a match
show_match_img = True  # If a match is found show a brief image of where the match area is
log_match_val = True  # Print info about the match value to console (used to find the bobber in the image)
log_bobber_loc = False  # Print the coordinates to console after a match is found (x,y,w,h)
log_diff_val = True  # Print info about the diff value to console (used to tell when a fish is hooked)
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

# ----- SCREEN CAPTURE (Wayland) -----#
# On Wayland pyscreenshot disables every X11 backend (mss/xlib only return black
# frames here), so it falls back to Wayland-capable backends: the XDG desktop
# portal (freedesktop_dbus) or external tools (grim/gnome-screenshot/spectacle).
# We pick the first backend that returns a non-black frame once, then reuse it,
# instead of letting pyscreenshot probe every backend on every single grab.
CAPTURE_BACKENDS = ["freedesktop_dbus", "pil", "grim", "gnome_dbus"]
capture_backend = None

# ----- DEFINE FUNCTIONS ------#


# Pick the first capture backend that actually returns a frame, and keep it
def pick_backend():
    global capture_backend
    for name in CAPTURE_BACKENDS:
        try:
            im = ImageGrab.grab(backend=name)
            if np.asarray(im).max() > 0:  # reject all-black frames
                capture_backend = name
                print(f"Capture backend: {name} ({im.size[0]}x{im.size[1]})")
                return
        except Exception as err:
            if debugging:
                print(f"Capture backend {name} unavailable: {err}")
    raise RuntimeError("No working screenshot backend found")


# Grab a region of the screen straight into memory (no file is written)
def grab(region):
    if debugging or log_region_val:
        print(f"Grabbing region: {region}")
    bbox = (region["x1"], region["y1"], region["x2"], region["y2"])
    im = ImageGrab.grab(bbox=bbox, backend=capture_backend).convert("RGB")
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)  # RGB -> OpenCV BGR


# Input position of wow screen and click once to make it the active window
def start_click(x, y):
    print("Starting fishing session")
    x, y = x + 100, y + 100  # top left corner + some pixels
    pyautogui.click(x, y)


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
    # Click middle of bobber box
    x, y = loc[0] + loc[2] // 2, loc[1] + loc[3] // 2
    pyautogui.moveTo(x, y)
    pyautogui.rightClick()

    print(f"Got em!")

    pyautogui.moveTo(100, 100)


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
def find_bobber(source, bobber, mask=None):
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
    if debugging or show_match_img:  # Show picture of region with rectangle around the match
        preview = source[:, :, :3].copy()
        cv2.rectangle(preview, top_left, bottom_right, (0, 255, 0), 2)
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

    # Grab a new image every 0.5s and compare it to original
    print("Waiting for fish...")
    diff_list = []
    for i in range(40):
        hooked = cv2.cvtColor(grab(dict_rect), cv2.COLOR_BGR2GRAY)
        diff = mse(nothooked, hooked)
        diff_list.append(diff)
        # Probably hooked
        if diff > diff_threshold:
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
    if debugging or log_diff_val:
        print("Timed out. Match was false or threshold to high")
        print(
            f"""Threshold: {diff_threshold}\nAvg diff: {sum(diff_list[1:]) / (len(diff_list) - 1)}\nMax diff: {max(diff_list)}\nMin diff: {min(diff_list[1:])}"""
        )
    return False


if __name__ == "__main__":
    bobber = load_bobber(bobber_img)  # cached in memory for the whole session
    mask = load_mask(bobber_mask)
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
            bobber_info = find_bobber(grab(game_window), bobber, mask)
            if not bobber_info:  # Match below threshold
                continue
            if watch_bobber(bobber_info):
                click_bobber(bobber_info)
                bobbers_clicked += 1

            else:
                print("Exiting loop...\n")
                timeouts += 1
                continue
        except OSError as err:
            print(f"OSError: {err}")
        except pyautogui.FailSafeException as err:
            print(f"Mouse moved outside monitor: {err}")
