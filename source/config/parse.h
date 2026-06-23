#pragma once

namespace settings {
    enum class ParseAction { RUN, EXIT };

    struct ParseResult {
        ParseAction action    = ParseAction::RUN;
        int         exit_code = 0;
    };

    ParseResult parse(int argc, char *argv[]);
}
