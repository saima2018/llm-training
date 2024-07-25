from dynaconf import Dynaconf

config = Dynaconf(
    envvar_prefix="ALGO",
    load_dotenv=True,
    environments=True,
    settings_files=["config/evaluation.yaml", "config/train.yaml"],
)

# config变量最好在主函数分解完
