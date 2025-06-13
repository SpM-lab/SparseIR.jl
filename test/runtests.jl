using TestItemRunner

is_stable_julia = !occursin("DEV", string(VERSION))

if v"1.11" <= VERSION < v"1.13-"
    @run_package_tests
else
    @run_package_tests filter = ti -> !(:jet_tests in ti.tags)
end
