from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    backend: Literal["sim", "hardware"] = Field(default="sim", alias="WAVECORE_BACKEND")
    modes: int = Field(default=32, alias="WAVECORE_MODES")
    alpha: float = Field(default=0.3, alias="WAVECORE_ALPHA")
    depth: int = Field(default=4, alias="WAVECORE_DEPTH")
    shots: int = Field(default=2000, alias="WAVECORE_SHOTS")
    onnx_opset: int = Field(default=18, alias="WAVECORE_ONNX_OPSET")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", alias="WAVECORE_LOG_LEVEL"
    )
    seed: int | None = Field(default=None, alias="WAVECORE_SEED")

    model_config = SettingsConfigDict(
        env_file=(".env",), env_prefix="", extra="ignore", case_sensitive=False
    )


settings = Settings()
