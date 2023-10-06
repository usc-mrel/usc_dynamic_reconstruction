function test_fatrix_adjoint(G)
    test_x = randn(G.idim) + 1i * randn(G.idim);
    test_y = randn(G.odim) + 1i * randn(G.odim);
    
    Gx = G * test_x;
    Ghy = G' * test_y;

    % abs(sum(Gx(:) .* conj(test_y(:))))
    % abs(sum(test_x(:) .* conj(Ghy(:))))
    
    subtraction = abs(sum(Gx(:) .* conj(test_y(:)))) ...
        - abs(sum(test_x(:) .* conj(Ghy(:))));
    
    assert(abs(subtraction) < 1e-2, "adjoint test failed!");
    
    fprintf("adjoint test passed!\n");
    
end