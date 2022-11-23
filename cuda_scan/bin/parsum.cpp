#include <iostream>
#include <string>
#include <sum_host.h>
#include <test_data.h>

const std::string PAR_SCAN_USAGE = R"(
usage: parsum par_scan

args:
    num_threads     number of threads used to compute sum
)";

void par_scan(int argc, char *argv[])
{
    // check for correct number of args
    if (argc != 3)
    {
        std::cout << PAR_SCAN_USAGE << std::endl;
        exit(1);
    }

    int num_threads = stoi((std::string)argv[2]);
    std::cout << sum_par_scan_cu(test_data, TEST_DATA_SIZE, num_threads) << std::endl;
}

const std::string PAR_USAGE = R"(
usage: parsum par

args:
    num_threads     number of threads used to compute sum
)";

void par(int argc, char *argv[])
{
    // check for correct number of args
    if (argc != 3)
    {
        std::cout << PAR_USAGE << std::endl;
        exit(1);
    }

    int num_threads = stoi((std::string)argv[2]);
    std::cout << sum_par_cu(test_data, TEST_DATA_SIZE, num_threads) << std::endl;
}

const std::string MPS_USAGE = R"(
usage: parsum <command> [<args>]

commands:
    par_scan    compute sum of test data using the par+scan algorithm on 1024 threads
    par         compute sum of test data using the par algorithm on 64 threads

contact:
    nick.machnik@gmail.com
)";

auto main(int argc, char *argv[]) -> int
{
    if (argc == 1)
    {
        std::cout << MPS_USAGE << std::endl;
        return EXIT_SUCCESS;
    }

    std::string cmd = (std::string)argv[1];

    if ((cmd == "--help") || (cmd == "-h"))
    {
        std::cout << MPS_USAGE << std::endl;
    }
    else if (cmd == "par")
    {
        par(argc, argv);
    }
    else if (cmd == "par_scan")
    {
        par_scan(argc, argv);
    }
    else
    {
        std::cout << MPS_USAGE << std::endl;
    }

    return EXIT_SUCCESS;
}
