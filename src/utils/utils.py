import os
import sys
import requests

CONSUL_URL_SERVER = "http://localhost:8500/v1/catalog/service/fl-server"
CHUNK_SIZE = 1024  # 1KB chunks
DIABETES_MLP_INPUT_SIZE = 16

class Styles:
    def __init__(self):
        # Text colors
        self.FG_RED = '\033[0;31m'
        self.FG_GREEN = '\033[0;32m'
        self.FG_YELLOW = '\033[0;33m'
        self.FG_BLUE = '\033[0;34m'
        self.FG_MAGENTA = '\033[0;35m'
        self.FG_CYAN = '\033[0;36m'
        self.FG_WHITE = '\033[0;37m'

        # Background colors
        self.BG_RED = '\033[41m'
        self.BG_GREEN = '\033[42m'
        self.BG_YELLOW = '\033[43m'
        self.BG_BLUE = '\033[44m'
        self.BG_MAGENTA = '\033[45m'
        self.BG_CYAN = '\033[46m'
        self.BG_WHITE = '\033[47m'

        # Text styles
        self.BOLD = '\033[1m'
        self.UNDERLINE = '\033[4m'
        self.BLINK = '\033[5m'
        self.REVERSE = '\033[7m'
        self.HIDDEN = '\033[8m'
        self.RESET = '\033[0m'

STYLES = Styles()

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def wait_for_enter():
    print()
    print("Press Enter to continue...", end="", flush=True)
    while True:
        char = sys.stdin.read(1)    # Reads one character at a time
        if char == "\n":            # Only proceed if Enter is pressed
            break

# Update this code to get the ip address of the server as well as return that
def get_server_address():
    response = requests.get(CONSUL_URL_SERVER).json()
    if response:
        service = response[0]
        return "localhost", service['ServicePort']
    return None, None

# Update this code later to handle dynamic ip address
def get_ip():
    return "localhost"