
from whisper_api.bin.whisper_trainer import WhisperTrainer
from whisper_api.utils.common_utils import (
    load_whisper_config,
    log_info
)

log_info("加载配置文件")
config = load_whisper_config("C://Users//pony//Desktop//WhisperMultitaskFinetuning//example//aishell-sample//config//whisper_multitask.yaml")

log_info("加载模型训练器")
whisper_trainer = WhisperTrainer(config)

log_info("开始训练")
whisper_trainer.train_start()

















