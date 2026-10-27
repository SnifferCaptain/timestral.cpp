/***************
 * @file parallel.cpp
 * @brief 跨算子复用线程，按完整输出块保留每个点积的K顺序
 ***************/

#include "timestral/blas/parallel.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace timestral::blas {

static std::size_t resolvedThreads(std::size_t threads) {
    if (threads == 0) {
        return std::max<std::size_t>(1, std::min<std::size_t>(4, std::thread::hardware_concurrency()));
    }
    if (threads > static_cast<std::size_t>((std::numeric_limits<int>::max)())) {
        throw std::invalid_argument("blas: thread count exceeds int range");
    }
    return threads;
}

Parallel::Parallel(std::size_t threads)
    : threadLimit(resolvedThreads(threads)) {
    try {
        workers.reserve(threadLimit - 1);
        for (std::size_t index = 1; index < threadLimit; ++index) {
            workers.emplace_back([this, index] { worker(index); });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(state);
            stopping = true;
        }
        ready.notify_all();
        for (auto& thread : workers) {
            thread.join();
        }
        throw;
    }
}

Parallel::~Parallel() {
    {
        std::lock_guard<std::mutex> lock(state);
        stopping = true;
    }
    ready.notify_all();
    for (auto& thread : workers) {
        thread.join();
    }
}

std::size_t Parallel::count() const {
    return threadLimit;
}

void Parallel::consume(std::size_t index) {
    try {
        for (std::size_t i = index; i < taskCount; i += threadLimit) {
            task(i);
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(state);
        if (!failure) {
            failure = std::current_exception();
        }
    }
}

void Parallel::worker(std::size_t index) {
    std::uint64_t seen = 0;
    std::unique_lock<std::mutex> lock(state);
    for (;;) {
        ready.wait(lock, [&] { return stopping || generation != seen; });
        if (stopping) {
            return;
        }
        seen = generation;
        if (std::fesetenv(&environment) != 0) {
            if (!failure) {
                failure = std::make_exception_ptr(std::runtime_error("blas: cannot apply floating environment"));
            }
            ++completed;
            done.notify_one();
        } else {
            lock.unlock();
            consume(index);
            lock.lock();
            ++completed;
            done.notify_one();
        }
    }
}

void Parallel::run(std::size_t tasks, const std::function<void(std::size_t)>& function) {
    if (threadLimit == 1 || tasks <= 1) {
        for (std::size_t i = 0; i < tasks; ++i) {
            function(i);
        }
        return;
    }
    bool nested = false;
    {
        std::lock_guard<std::mutex> lock(state);
        const auto current = std::this_thread::get_id();
        nested = active && (caller == current
            || std::any_of(workers.begin(), workers.end(), [&](const std::thread& thread) { return thread.get_id() == current; }));
    }
    if (nested) {
        for (std::size_t i = 0; i < tasks; ++i) {
            function(i);
        }
        return;
    }
    std::lock_guard<std::mutex> submit(submission);
    std::fenv_t current{};
    if (std::fegetenv(&current) != 0) {
        throw std::runtime_error("blas: cannot read floating environment");
    }
    {
        std::lock_guard<std::mutex> lock(state);
        environment = current;
        task = function;
        active = true;
        caller = std::this_thread::get_id();
        taskCount = tasks;
        completed = 0;
        failure = {};
        ++generation;
    }
    ready.notify_all();
    consume(0);
    std::unique_lock<std::mutex> lock(state);
    done.wait(lock, [&] { return completed == workers.size(); });
    task = {};
    active = false;
    const auto error = failure;
    lock.unlock();
    if (error) {
        std::rethrow_exception(error);
    }
}

void Parallel::tiles(std::size_t rows, std::size_t columns,
    const std::function<void(std::size_t, std::size_t, std::size_t, std::size_t)>& function) {
    if (rows == 0 || columns == 0) {
        return;
    }
    constexpr std::size_t rowBlock = 120;
    const auto rowParts = rows / rowBlock + (rows % rowBlock != 0);
    const auto panels = columns / 16 + (columns % 16 != 0);
    const auto targetColumns = rowParts >= threadLimit ? std::size_t{1}
        : threadLimit / rowParts + (threadLimit % rowParts != 0);
    const auto columnParts = std::min(panels, targetColumns);
    const auto panelsPerPart = panels / columnParts + (panels % columnParts != 0);
    const auto actualColumns = panels / panelsPerPart + (panels % panelsPerPart != 0);
    run(rowParts * actualColumns, [&](std::size_t taskIndex) {
        const auto row = taskIndex / actualColumns * rowBlock;
        const auto column = taskIndex % actualColumns * panelsPerPart * 16;
        function(row, std::min(rowBlock, rows - row), column,
            std::min(panelsPerPart * 16, columns - column));
    });
}

} // namespace timestral::blas
