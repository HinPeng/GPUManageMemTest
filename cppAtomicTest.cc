#include <iostream>
#include <atomic>
#include <vector>
#include <thread>

const int N = 15;

void GetValue(int n){
    static std::atomic<int> log_counter{0};
    int counter_value = log_counter.load(std::memory_order_relaxed);
    if (counter_value < 10){
        printf("Yes, From %d, counter_value is: %d!\n", n, counter_value);
        log_counter.store(counter_value + 1, std::memory_order_relaxed);
    }
    else
        printf("No, From %d: counter_value is: %d!\n", n, counter_value);
}
int main()
{
    std::vector<std::thread> thread_pool;
    for (int i = 0; i < N; ++i){
        thread_pool.push_back(std::thread(GetValue, i));
    }
    for (int i = 0; i < N; ++i){
        thread_pool[i].join();
    }

    
    return 0;
}