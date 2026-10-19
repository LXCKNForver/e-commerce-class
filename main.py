from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"), override=True)

from app.llm import build_llm


PROJECT_NAME = "电商售后服务智能体"
FIRST_QUESTION = "订单 SO-2001 已签收，我能退货吗？"


def main() -> None:
    messages = [{"role": "user", "content": FIRST_QUESTION}]
    answer = build_llm().chat(messages)
    print(f"项目：{PROJECT_NAME}")
    print(f"问题：{FIRST_QUESTION}")
    print(f"回答：{answer}")


if __name__ == "__main__":
    main()
