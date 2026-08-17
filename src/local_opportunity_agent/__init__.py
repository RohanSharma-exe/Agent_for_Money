from local_opportunity_agent.core.settings import load_settings


def main() -> None:
    settings = load_settings()
    print(f"Local Opportunity Agent: {settings.env}")
