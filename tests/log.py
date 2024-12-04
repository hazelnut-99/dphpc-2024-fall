from colorama import Fore, Style, init

init(autoreset=True)

def log_success(message):
    print(Fore.GREEN + "[✓] " + message + Style.RESET_ALL)


def log_error(message):
    print(Fore.RED + "[x] " + message + Style.RESET_ALL)


def log_info(message):
    print(Fore.BLUE + "[i] " + message + Style.RESET_ALL)
