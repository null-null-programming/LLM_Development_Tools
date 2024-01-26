from metagpt.roles import Role
from write_code import GPTWriteCode
from metagpt.schema import Message
from metagpt.logs import logger


class GPTCoder(Role):
    name: str = "Alice"
    profile: str = "GPTCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([GPTWriteCode])

    async def _act(self) -> Message:
        # logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo
        msg = self.get_memories(k=1)[0]
        code_text = await todo.run(msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))
        return msg
