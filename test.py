import asyncio
from metagpt.logs import logger
from coder import GPTCoder


class write_code:
    def __init__(self, message) -> None:
        self.message = message

    async def run(self) -> str:
        role = GPTCoder()
        # logger.info(self.message)
        result = await role.run(self.message)
        # logger.info(result)
        return result


if __name__ == "__main__":
    print(
        asyncio.run(
            write_code(
                "Write a python function that can add two numbers and provide two runnnable test cases. Return ```python your_code_here ``` with NO other texts, your code:"
            ).run()
        )
    )
