#ifndef HYDROSIM_IO_H
#define HYDROSIM_IO_H

#include <iostream>
#include <sstream>

extern std::string inputs_name;


class StdoutCapture {

public:
    StdoutCapture () : prior_stream_buf(std::cout.rdbuf()) {
        std::cout.rdbuf(buffered_text.rdbuf());
    }

    ~StdoutCapture () {
        std::cout.rdbuf(prior_stream_buf);
    }

    StdoutCapture (const StdoutCapture&) = delete;
    StdoutCapture (StdoutCapture&&) = delete;
    StdoutCapture& operator= (const StdoutCapture&) = delete;
    StdoutCapture& operator= (StdoutCapture&&) = delete;

    std::string getBufferedText () {
        return buffered_text.str();
    }

private:
    std::ostringstream buffered_text;
    std::streambuf* prior_stream_buf;

};

#endif
