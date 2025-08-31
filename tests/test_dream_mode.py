import json
import dream_mode


def test_simulate_dialogue_uses_external_data(tmp_path):
    data = {"prompts": ["hi"], "responses": ["bye"]}
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(data))
    dialogue = dream_mode._simulate_dialogue(turns=1, data_path=str(data_file))
    assert dialogue == ["hi", "bye"]
