"""
Verification script to test RNN operations imports
"""

print("Testing RNN imports from ilovetools.ml.rnn...")

try:
    from ilovetools.ml.rnn import (
        rnn_cell_forward,
        rnn_forward,
        lstm_cell_forward,
        lstm_forward,
        gru_cell_forward,
        gru_forward,
        bidirectional_rnn_forward,
        sigmoid,
        initialize_rnn_weights,
        clip_gradients,
        vanilla_rnn,
        lstm,
        gru,
        bidirectional_rnn,
    )
    print("✓ All RNN functions imported successfully")
    
    # Test basic functionality
    import numpy as np
    
    # Test Basic RNN
    x = np.random.randn(32, 10, 128)  # (batch, seq_len, input_size)
    h_0 = np.zeros((32, 256))
    
    weights_rnn = initialize_rnn_weights(128, 256, cell_type='rnn')
    outputs, hidden_states = rnn_forward(
        x, h_0, weights_rnn['W_xh'], weights_rnn['W_hh'], weights_rnn['b_h']
    )
    print(f"✓ Basic RNN works: {x.shape} -> {outputs.shape}")
    
    # Test LSTM
    c_0 = np.zeros((32, 256))
    weights_lstm = initialize_rnn_weights(128, 256, cell_type='lstm')
    outputs, h_states, c_states = lstm_forward(
        x, h_0, c_0,
        weights_lstm['W_f'], weights_lstm['W_i'], weights_lstm['W_c'], weights_lstm['W_o'],
        weights_lstm['b_f'], weights_lstm['b_i'], weights_lstm['b_c'], weights_lstm['b_o']
    )
    print(f"✓ LSTM works: {x.shape} -> {outputs.shape}")
    print(f"  Cell states shape: {c_states.shape}")
    
    # Test GRU
    weights_gru = initialize_rnn_weights(128, 256, cell_type='gru')
    outputs, hidden_states = gru_forward(
        x, h_0,
        weights_gru['W_z'], weights_gru['W_r'], weights_gru['W_h'],
        weights_gru['b_z'], weights_gru['b_r'], weights_gru['b_h']
    )
    print(f"✓ GRU works: {x.shape} -> {outputs.shape}")
    
    # Test Bidirectional RNN
    h_0_f = np.zeros((32, 256))
    h_0_b = np.zeros((32, 256))
    weights_f = initialize_rnn_weights(128, 256, cell_type='rnn')
    weights_b = initialize_rnn_weights(128, 256, cell_type='rnn')
    
    outputs, f_states, b_states = bidirectional_rnn_forward(
        x, h_0_f, h_0_b,
        weights_f['W_xh'], weights_f['W_hh'], weights_f['b_h'],
        weights_b['W_xh'], weights_b['W_hh'], weights_b['b_h']
    )
    print(f"✓ Bidirectional RNN works: {x.shape} -> {outputs.shape}")
    print(f"  Output has 2x hidden_size: {outputs.shape[2]} = 2 * 256")
    
    # Test Sigmoid
    test_sigmoid = sigmoid(np.array([-1, 0, 1]))
    print(f"✓ Sigmoid works: sigmoid([-1, 0, 1]) = {test_sigmoid}")
    
    # Test Gradient Clipping
    large_grads = np.random.randn(100, 100) * 10
    clipped = clip_gradients(large_grads, max_norm=5.0)
    norm = np.linalg.norm(clipped)
    print(f"✓ Gradient clipping works: norm {np.linalg.norm(large_grads):.2f} -> {norm:.2f}")
    
    # Test Weight Initialization
    for cell_type in ['rnn', 'lstm', 'gru']:
        weights = initialize_rnn_weights(128, 256, cell_type=cell_type)
        print(f"✓ {cell_type.upper()} weight initialization: {len(weights)} weight matrices")
    
    print("\n✅ All verifications passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
