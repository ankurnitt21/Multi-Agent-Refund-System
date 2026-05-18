import asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database.db_setup import async_session_factory
from database.models import PromptRegistry as PromptRegistryModel


class PromptRegistry:
    _instance = None
    _cache: dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def load_active_prompts(self):
        async with async_session_factory() as session:
            result = await session.execute(
                select(PromptRegistryModel).where(PromptRegistryModel.is_active == True)
            )
            prompts = result.scalars().all()
            self._cache = {p.prompt_name: p.content for p in prompts}

    def get(self, prompt_name: str) -> str:
        return self._cache.get(prompt_name, "")

    async def activate_version(self, prompt_name: str, version: str):
        async with async_session_factory() as session:
            # Deactivate all versions of this prompt
            await session.execute(
                update(PromptRegistryModel)
                .where(PromptRegistryModel.prompt_name == prompt_name)
                .values(is_active=False)
            )
            # Activate the specified version
            await session.execute(
                update(PromptRegistryModel)
                .where(
                    PromptRegistryModel.prompt_name == prompt_name,
                    PromptRegistryModel.version == version,
                )
                .values(is_active=True)
            )
            await session.commit()
        await self.load_active_prompts()

    async def create_version(
        self, prompt_name: str, version: str, content: str,
        description: str, created_by: str
    ) -> int:
        async with async_session_factory() as session:
            new_prompt = PromptRegistryModel(
                prompt_name=prompt_name,
                version=version,
                content=content,
                is_active=False,
                description=description,
                created_by=created_by,
            )
            session.add(new_prompt)
            await session.commit()
            await session.refresh(new_prompt)
            return new_prompt.id

    async def list_versions(self, prompt_name: str) -> list[dict]:
        async with async_session_factory() as session:
            result = await session.execute(
                select(PromptRegistryModel)
                .where(PromptRegistryModel.prompt_name == prompt_name)
                .order_by(PromptRegistryModel.created_at.desc())
            )
            prompts = result.scalars().all()
            return [
                {
                    "id": p.id,
                    "version": p.version,
                    "is_active": p.is_active,
                    "description": p.description,
                    "created_at": str(p.created_at) if p.created_at else None,
                    "created_by": p.created_by,
                }
                for p in prompts
            ]

    async def _auto_refresh_loop(self):
        while True:
            await asyncio.sleep(300)  # 5 minutes
            await self.load_active_prompts()

    def start_auto_refresh(self):
        asyncio.create_task(self._auto_refresh_loop())
