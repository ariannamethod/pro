from autoadapt import LayerMutator, LoRALayer


def test_lora_save_and_load(tmp_path):
    mut = LayerMutator(use_lora=True)
    layer = LoRALayer(
        name="dense",
        rank=2,
        alpha=1.0,
        matrix_a=[[1.0, 0.0], [0.0, 1.0]],
        matrix_b=[[0.5, 0.0], [0.0, 0.5]],
    )
    mut.add_lora_layer(layer)
    mut.save(tmp_path)

    mut2 = LayerMutator(use_lora=True)
    mut2.load(tmp_path)
    assert "dense" in mut2.lora_layers
    loaded = mut2.lora_layers["dense"]
    assert loaded.matrix_a == layer.matrix_a
    assert loaded.matrix_b == layer.matrix_b
