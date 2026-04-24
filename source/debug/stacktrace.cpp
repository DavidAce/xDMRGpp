#include "stacktrace.h"
#include <csignal>
#include <cstdio>
#include <cstdlib>

void debug::signal_callback_handler(int status) {
    if(status != 0) debug::print_stack_trace();
    switch(status) {
        case SIGABRT: {
            std::fprintf(stderr, "Exit SIGABRT: %d\n", status);
            break;
        }
        case SIGBUS: {
            std::fprintf(stderr, "Exit SIGBUS: %d\n", status);
            break;
        }
        case SIGILL: {
            std::fprintf(stderr, "Exit SIGILL: %d\n", status);
            break;
        }
        case SIGFPE: {
            std::fprintf(stderr, "Exit SIGFPE: %d\n", status);
            std::quick_exit(status);
        }
        case SIGSEGV: {
            std::fprintf(stderr, "Exit SIGSEGV: %d\n", status);
            std::quick_exit(status);
        }
        case SIGTRAP: {
            std::fprintf(stderr, "Exit SIGTRAP: %d\n", status);
            break;
        }
        case SIGSYS: {
            std::fprintf(stderr, "Exit SIGSYS: %d\n", status);
            break;
        }
        default: {
            std::fprintf(stderr, "Exit %d\n", status);
            break;
        }
    }
    std::exit(status);
}

void debug::register_callbacks() {
    /* Only synchronous fault signals should route through the stack-trace handler. */
    signal(SIGABRT, signal_callback_handler); // Abnormal termination.
    signal(SIGBUS, signal_callback_handler);  // Bus error.
    signal(SIGFPE, signal_callback_handler);   // Erroneous arithmetic operation.
    signal(SIGILL, signal_callback_handler);   // Illegal instruction.
    signal(SIGSEGV, signal_callback_handler);  // Invalid access to storage.
    signal(SIGTRAP, signal_callback_handler);  // Trace/breakpoint trap.
    signal(SIGSYS, signal_callback_handler);   // Bad system call.
}

#if __has_include(<backward.hpp>)
    #if defined(BACKWARD_REDEFINE_DW)
        #undef BACKWARD_HAS_DW
        #define BACKWARD_HAS_DW 1
    #endif
    #include <backward.hpp>
void debug::print_stack_trace() {
    backward::StackTrace st;
    st.load_here(128);
    // Skip this scope (1) , as well as the signal_callback_handler scope (2)
    st.skip_n_firsts(2);
    backward::Printer p;
    p.print(st);
}
#else
void debug::print_stack_trace() {}
#endif
