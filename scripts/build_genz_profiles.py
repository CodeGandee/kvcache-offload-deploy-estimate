"""Generate GenZ model-core timing tables in LLMServingSim bundle format."""

from kvcache_offload_deploy_estimate.genz_llmservingsim import generate_profile_bundles


def main() -> None:
    paths = generate_profile_bundles()
    print(f"wrote {len(paths)} profile files")


if __name__ == "__main__":
    main()
