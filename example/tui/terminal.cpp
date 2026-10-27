/***************
 * @file terminal.cpp
 * @brief 即时读取单键，退出时恢复输入模式、颜色与光标
 ***************/

#include "terminal.hpp"

#include <iostream>

#if !defined(_WIN32)
#include <cerrno>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

DemoTerminal::DemoTerminal(bool plain) {
    if (plain) {
        return;
    }
#if defined(_WIN32)
    input = GetStdHandle(STD_INPUT_HANDLE);
    output = GetStdHandle(STD_OUTPUT_HANDLE);
    if (!GetConsoleMode(input, &originalInput) || !GetConsoleMode(output, &originalOutput)) {
        return;
    }
    const DWORD mode = (originalInput | ENABLE_EXTENDED_FLAGS | ENABLE_WINDOW_INPUT)
        & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT | ENABLE_QUICK_EDIT_MODE);
    if (!SetConsoleMode(input, mode)) {
        return;
    }
    active = true;
    ansi = SetConsoleMode(output, originalOutput | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)
        || tcgetattr(STDIN_FILENO, &original) != 0) {
        return;
    }
    auto raw = original;
    raw.c_lflag &= ~static_cast<tcflag_t>(ICANON | ECHO | ISIG | IEXTEN);
    raw.c_iflag &= ~static_cast<tcflag_t>(IXON | ICRNL);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
        return;
    }
    active = true;
    ansi = true;
#endif
    if (ansi) {
        std::cout << "\x1b[?1049h\x1b[?25l" << std::flush;
    }
}

DemoTerminal::~DemoTerminal() {
    if (!active) {
        return;
    }
    if (ansi) {
        std::cout << "\x1b[0m\x1b[?25h\x1b[?1049l" << std::flush;
    }
#if defined(_WIN32)
    SetConsoleMode(input, originalInput);
    SetConsoleMode(output, originalOutput);
#else
    tcsetattr(STDIN_FILENO, TCSANOW, &original);
#endif
}

bool DemoTerminal::interactive() const {
    return active;
}

bool DemoTerminal::colors() const {
    return ansi;
}

int DemoTerminal::readKey() {
    if (!active) {
        return std::cin.get();
    }
#if defined(_WIN32)
    INPUT_RECORD record{};
    DWORD count = 0;
    while (ReadConsoleInputW(input, &record, 1, &count) && count != 0) {
        if (record.EventType == WINDOW_BUFFER_SIZE_EVENT) {
            return 12;
        }
        if (record.EventType != KEY_EVENT || !record.Event.KeyEvent.bKeyDown) {
            continue;
        }
        const auto& key = record.Event.KeyEvent;
        if ((key.dwControlKeyState & (LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED)) != 0
            && key.wVirtualKeyCode == 'C') {
            return 3;
        }
        const auto character = key.uChar.UnicodeChar;
        if (character > 0 && character < 128) {
            return static_cast<int>(character);
        }
    }
    return -1;
#else
    unsigned char key = 0;
    while (true) {
        const auto count = ::read(STDIN_FILENO, &key, 1);
        if (count == 1) {
            return key;
        }
        if (count == 0 || errno != EINTR) {
            return -1;
        }
    }
#endif
}

void DemoTerminal::writeFrame(const std::string& frame) const {
    if (!active) {
        std::cout << frame << std::flush;
        return;
    }
    const auto limit = columns() > 1 ? columns() - 1 : 1;
    std::string fitted;
    std::size_t column = 0;
    for (std::size_t i = 0; i < frame.size(); ++i) {
        const char value = frame[i];
        if (value == '\x1b' && i + 1 < frame.size() && frame[i + 1] == '[') {
            fitted += value;
            fitted += frame[++i];
            while (++i < frame.size()) {
                fitted += frame[i];
                if (frame[i] >= '@' && frame[i] <= '~') {
                    break;
                }
            }
        } else if (value == '\n') {
            fitted += value;
            column = 0;
        } else if (column++ < limit) {
            fitted += value;
        }
    }
    std::cout << fitted << std::flush;
}

std::size_t DemoTerminal::columns() const {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (active && GetConsoleScreenBufferInfo(output, &info)) {
        return static_cast<std::size_t>(info.srWindow.Right - info.srWindow.Left + 1);
    }
#else
    winsize size{};
    if (active && ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_col != 0) {
        return size.ws_col;
    }
#endif
    return 120;
}

std::size_t DemoTerminal::rows() const {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO info{};
    if (active && GetConsoleScreenBufferInfo(output, &info)) {
        return static_cast<std::size_t>(info.srWindow.Bottom - info.srWindow.Top + 1);
    }
#else
    winsize size{};
    if (active && ioctl(STDOUT_FILENO, TIOCGWINSZ, &size) == 0 && size.ws_row != 0) {
        return size.ws_row;
    }
#endif
    return 40;
}
