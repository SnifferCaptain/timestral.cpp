/***************
 * @file streamingTui.cpp
 * @brief 即时数字键、多通道推进及彩色分位数带界面
 ***************/

#include <algorithm>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "streaming.hpp"
#include "terminal.hpp"
#include "timestral/utils/backend.hpp"
#include "timestral/utils/config.hpp"
#include "timestral/utils/safetensors.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            throw std::invalid_argument("usage: timestralTui MODEL CONFIG [--plain] [--once] [--threads N]");
        }
        bool once = false;
        bool plain = false;
        bool bands = true;
        std::size_t threads = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--once") {
                once = true;
            } else if (option == "--plain") {
                plain = true;
            } else if (option == "--no-bands") {
                bands = false;
            } else if (option == "--threads") {
                if (i + 1 >= argc) {
                    throw std::invalid_argument("--threads requires a count");
                }
                threads = std::stoul(argv[++i]);
            } else {
                throw std::invalid_argument("unknown option: " + option);
            }
        }
        const std::filesystem::path directory(argv[1]);
        const auto modelConfig = timestral::ModelConfig::load(directory / "config.json");
        std::shared_ptr<const timestral::Model> model = timestral::loadBackendModel(
            modelConfig,
            timestral::utils::Safetensors::load(directory / "model.safetensors"),
            threads
        );
        const auto config = StreamingConfig::load(argv[2]);
        StreamingDemo demo(model, config);
        DemoTerminal terminal(plain || once);
        const auto draw = [&](const std::string& status) {
            auto width = config.width;
            auto height = config.height;
            if (terminal.interactive()) {
                width = std::min(width, terminal.columns() > 24 ? terminal.columns() - 4 : 20);
                const auto reserved = config.channels.size() + 19;
                height = std::min(height, terminal.rows() > reserved + 5
                    ? terminal.rows() - reserved : 5);
            }
            demo.setView(bands, terminal.colors(), width, height);
            std::ostringstream frame;
            demo.render(frame);
            frame << "[1-9] channel  [a] all  [r] recompute  [b] p10/p90  [q/Ctrl+C] quit\n"
                << (terminal.interactive() ? "LIVE KEY: no Enter required" : "PLAIN / redirected input")
                << " | threads=" << model->threadCount() << " | " << status << '\n';
            terminal.writeFrame(frame.str());
        };
        demo.update();
        draw("Ready");
        if (once) {
            return 0;
        }
        while (true) {
            const int key = terminal.readKey();
            if (key < 0 || key == 'q' || key == 'Q' || key == 3 || key == 4) {
                break;
            }
            if (key == '\r' || key == '\n' || key == '\t' || key == ' ') {
                continue;
            }
            if (key == 'b' || key == 'B') {
                bands = !bands;
                draw(bands ? "p10/p90 shown" : "p10/p90 hidden");
                continue;
            }
            if (key == 12) {
                draw("Resized");
                continue;
            }
            try {
                if (terminal.interactive()) {
                    draw("Computing...");
                }
                if (key >= '1' && key <= '9') {
                    demo.advance(static_cast<std::size_t>(key - '1'));
                } else if (key == 'a' || key == 'A') {
                    for (std::size_t i = 0; i < config.channels.size(); ++i) {
                        const auto& ends = demo.progress();
                        const auto shortest = static_cast<std::size_t>(
                            std::min_element(ends.begin(), ends.end()) - ends.begin()
                        );
                        demo.advance(shortest);
                    }
                } else if (key != 'r' && key != 'R') {
                    throw std::invalid_argument("Unknown key; use 1-9, a, r, b or q");
                }
                demo.update();
                draw("Updated");
            } catch (const std::exception& error) {
                draw(std::string("ERROR: ") + error.what());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "timestralTui: " << error.what() << '\n';
        return 1;
    }
}
