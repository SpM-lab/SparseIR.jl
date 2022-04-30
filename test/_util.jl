function typestable(@nospecialize(f), @nospecialize(t); checkonlyany=false)
    v = code_typed(f, t)
    stable = true
    for vi in v
        for (name, ty) in zip(vi[1].slotnames, vi[1].slottypes)
            !(ty isa Type) && continue
            if (checkonlyany && ty === Any) ||
               (!checkonlyany && (!Base.isdispatchelem(ty) || ty == Core.Box))
                stable = false
                println("Type instability is detected! the variable is $(name) ::$ty")
            end
        end
    end
    return stable
end
