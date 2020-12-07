from pynput import keyboard
from time import time
ready = 1
angle = 90
t = 0
def on_press(key):
    global ready
    global t
    global angle
    if angle > 0:
        if key == keyboard.Key.left:
            angle -= 2
            print(angle)
    if angle < 180:
        if key == keyboard.Key.right:
            angle += 2
            print(angle)
    if key == keyboard.Key.space:
        if time() - t >= 5 and ready == 1:
            print("Locked and loaded!")
            ready = 0
        elif ready == 1:
            print("The cannon needs a break.")
            ready = 0
    try:
        if key.char == "a":
            print("you moved to the left")
        elif key.char == "d":
            print("you moved to the right")
        if key.char == "w":
            print("you moved upwards")
        elif key.char == "s":
            print("you moved downwards")

    except AttributeError:
        pass



def on_release(key):
    global ready
    global t
    if key == keyboard.Key.esc:
        # Stop listener
        return False
    if key == keyboard.Key.space:
        ready = 1
        if time() - t >= 5:
            print("you fired a cannonball")

            t = time()
# Collect events until released
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
