using TestItemRunner

if VERSION >= v"1.11"
    @run_package_tests
else
    @run_package_tests filter = ti -> !(:above_julia1_11 in ti.tags)
end
