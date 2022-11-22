#include <array>
#include <string>

const std::string PAR_USAGE = R"(
usage: mps par <file>

arguments:

)";

void par(int argc, char *argv[])
{
    // check for correct number of args
    if (argc != 3)
    {
        std::cout << PAR_USAGE << std::endl;
        exit(1);
    }
}

const std::string PAR_USAGE = R"(
usage: mps par <file>

arguments:

)";

void par(int argc, char *argv[])
{
    // check for correct number of args
    if (argc != 3)
    {
        std::cout << PAR_USAGE << std::endl;
        exit(1);
    }
}

const std::string MPS_USAGE = R"(
usage: mps <command> [<args>]

commands:
    printbf Interpret contents of binary file as floats and print to stoud
    prep    Prepare input (PLINK) .bed file for mps
    scorr   Compute the marker/phenotype correlation matrix in sparse format
    corr    Compute the marker/phenotype correlation matrix
    mcorrk  Compute pearson correlations between markers as sin(pi / 2 tau_b)
    mcorrp  Compute pearson correlations between markers
    cups    use cuPC to compute the parent set for each phenotype

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
    else if (cmd == "prep")
    {
        par(argc, argv);
    }
    else if (cmd == "corr")
    {
        par_scan(argc, argv);
    }
    else
    {
        std::cout << MPS_USAGE << std::endl;
    }

    return EXIT_SUCCESS;
}
