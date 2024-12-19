from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from api.news_router import router as news_router

app = FastAPI()


@app.get("/", include_in_schema=False)
async def redirect_from_root():
    return RedirectResponse(url="/docs")


app.include_router(news_router)
