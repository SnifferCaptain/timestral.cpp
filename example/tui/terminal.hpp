/***************
 * @file terminal.hpp
 * @brief 仅示例层使用的即时按键与终端状态RAII
 ***************/

#pragma once

#include <cstddef>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <termios.h>
#endif

class DemoTerminal {
public:
    explicit DemoTerminal(bool plain = false);
    ~DemoTerminal();
    DemoTerminal(const DemoTerminal&) = delete;
    DemoTerminal& operator=(const DemoTerminal&) = delete;
    bool interactive() const;
    bool colors() const;
    int readKey(); // EOF为-1，Ctrl+C为3，resize为12
    std::size_t columns() const;
    std::size_t rows() const;
    void writeFrame(const std::string& frame) const;

private:
    bool active = false;
    bool ansi = false;
#if defined(_WIN32)
    HANDLE input = INVALID_HANDLE_VALUE;
    HANDLE output = INVALID_HANDLE_VALUE;
    DWORD originalInput = 0;
    DWORD originalOutput = 0;
#else
    termios original{};
#endif
};
