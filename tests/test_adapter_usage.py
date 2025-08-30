import pytest
import pro_memory


@pytest.mark.asyncio
async def test_adapter_usage(tmp_path):
    pro_memory.DB_PATH = str(tmp_path / "mem.db")
    await pro_memory.init_db()
    await pro_memory.increment_adapter_usage("a1")
    await pro_memory.increment_adapter_usage("a1")
    total = await pro_memory.total_adapter_usage()
    assert total == 2
    await pro_memory.reset_adapter_usage()
    total2 = await pro_memory.total_adapter_usage()
    assert total2 == 0
    await pro_memory.close_db()
