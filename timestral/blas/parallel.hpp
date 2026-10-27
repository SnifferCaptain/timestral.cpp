/***************
 * @file parallel.hpp
 * @brief BLAS固定工作线程与互不重叠的输出块分派
 ***************/

#pragma once

#include <cfenv>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace timestral::blas {

// 调用线程参与计算，后台线程跨算子复用，同池嵌套提交在当前线程串行执行
// 每次提交记录调用者的浮点环境，工作线程在取任务前恢复，避免舍入模式与调用者不同
class Parallel {
public:
    explicit Parallel(std::size_t threads = 1);
    ~Parallel();
    Parallel(const Parallel&) = delete;
    Parallel& operator=(const Parallel&) = delete;

    std::size_t count() const;
    void run(std::size_t tasks, const std::function<void(std::size_t)>& function);
    void tiles(std::size_t rows, std::size_t columns,
        const std::function<void(std::size_t, std::size_t, std::size_t, std::size_t)>& function);

private:
    void worker(std::size_t index);
    void consume(std::size_t index);

    const std::size_t threadLimit;
    std::mutex submission;
    std::mutex state;
    std::condition_variable ready;
    std::condition_variable done;
    std::vector<std::thread> workers;
    std::function<void(std::size_t)> task;
    std::exception_ptr failure;
    std::size_t taskCount = 0;
    std::size_t completed = 0;
    std::uint64_t generation = 0;
    std::fenv_t environment{};
    bool stopping = false;
    bool active = false;
    std::thread::id caller;
};

} // namespace timestral::blas
