@testitem "Aqua" begin
	using Test
	import Aqua
	@testset "Aqua" begin
	    Aqua.test_all(Tensor4All; deps_compat = false)
	end
end
