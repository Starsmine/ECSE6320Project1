#include <iostream>
#include <liburing.h>

int main() {
    struct io_uring ring;
    io_uring_queue_init(32, &ring, 0);
    io_uring_queue_exit(&ring);
    std::cout << "io_uring works!\n";
    return 0;
}
