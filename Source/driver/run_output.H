#ifndef RUN_OUTPUT_H
#define RUN_OUTPUT_H

#include <iostream>
#include <sstream>

class StdoutCapture {

public:
    StdoutCapture (const StdoutCapture&) = delete;

    StdoutCapture (StdoutCapture&&) = delete;

    StdoutCapture& operator= (const StdoutCapture&) = delete;
    StdoutCapture& operator= (StdoutCapture&&) = delete;
    StdoutCapture () : prior_stream_buf(std::cout.rdbuf()) {
        std::cout.rdbuf(buffered_text.rdbuf());
    }
    ~StdoutCapture () {
        std::cout.rdbuf(prior_stream_buf);
    }

    std::string getBufferedText () {
        return buffered_text.str();
    }

private:
    std::ostringstream buffered_text;
    std::streambuf* prior_stream_buf;

};

extern std::string inputs_name;

#endif
