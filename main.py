import asyncio
from api_manager import BinanceManager
from utils import convert_df_to_talib_pvformat
from indicator_generator import TALibBatchProcessor
from config_manager import get_config_manager


def orchestrate_processing(df):
    cfg = get_config_manager()
    gen_mode = cfg.get_generator_mode()
    param_mode = cfg.get_parameter_mode()
    selected = cfg.get_selected_indicators()

    print(f"配置: generator_mode={gen_mode}, parameter_mode={param_mode}, indicators={selected}")

    processor = TALibBatchProcessor(df)

    if gen_mode == 'single':
        if not selected:
            print("[ERROR] single 模式但未指定 indicator"); return
        ind = selected[0]
        if param_mode == 'multiple':
            processor.run_on_func_all(ind)
        else:
            processor.run_on_func_default(ind)

    elif gen_mode == 'multiple':
        if not selected:
            print("[ERROR] multiple 模式但 indicators 为空"); return
        if param_mode == 'multiple':
            processor.run_indicators_all_params(selected)
        else:
            processor.run_on_multiple_funcs_default(selected)

    else:  # all
        if param_mode == 'multiple':
            processor.run_all_indicators_all_params()
        else:
            processor.run_all_indicators_default_mp()

async def main():
    manager = BinanceManager()
    df_raw = await manager.get_kline()
    df = convert_df_to_talib_pvformat(df_raw)
    print(f"已获取 {len(df)} 行 K 线数据，开始生成指标…")
    orchestrate_processing(df)

if __name__ == "__main__":
    asyncio.run(main())

