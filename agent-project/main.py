"""CLI entry point for the suggestion agent."""

from agent.agent import Agent
from config import settings
from tools import registry
from colorama import init, Fore


def main():
    init(autoreset=True)
    settings.tool_registry = registry
    agent = Agent(settings=settings)
    while True:
        try:
            user_input = input(Fore.CYAN + "You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        response = agent.run(user_input)
        print(Fore.GREEN + f"Agent: {response}")


if __name__ == "__main__":
    main()
