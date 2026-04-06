# scripts/run_server.py

import uvicorn
from env.config import SERVER_HOST, SERVER_PORT


def main():
    print("\n🚀 Starting Email Triage Server")
    print(f"🌐 URL  : http://{SERVER_HOST}:{SERVER_PORT}")
    print(f"📘 Docs : http://{SERVER_HOST}:{SERVER_PORT}/docs\n")

    uvicorn.run(
        "server.app:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()