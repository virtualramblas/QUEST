import argparse
import sys
from datasets import load_dataset
sys.path.append('..')
from w5h_classifier import W5HClassifier, print_profile

def get_sql_prompts_for_domain(dataset, domain_value, sample_count=None):
    """
    Retrieves a list of sql_prompt values for a given domain from the dataset.

    Args:
        dataset: The Hugging Face dataset (e.g., train_dataset).
        domain_value (str): The domain to filter by.
        sample_count (int, optional): The maximum number of sql_prompts to return.
                                      If None, all prompts for the domain are returned.

    Returns:
        list: A list of sql_prompt strings matching the domain, up to sample_count.
    """
    # Filter the dataset by the given domain value and sql_task_type
    filtered_dataset = dataset.filter(lambda example: example['domain'] == domain_value and example['sql_task_type'] == 'data retrieval')

    # Get all 'sql_prompt' values from the filtered dataset
    sql_prompts = filtered_dataset['sql_prompt']

    # Limit the number of prompts if sample_count is provided
    if sample_count is not None and sample_count < len(sql_prompts):
        sql_prompts = sql_prompts[:sample_count]

    return sql_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ArXiv research agent powered by Hermes Agent + Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--domain", "-d",
        required=True,
        help="The domain to which the prompts belong to.",
    )
    parser.add_argument(
        "--samples", "-s",
        default=10,
        help="The number of prompt samples to parse.",
    )

    args = parser.parse_args()

    # Download the dataset from HF Hub
    dataset_id = "gretelai/synthetic_text_to_sql"
    dataset = load_dataset(dataset_id)

    # Access the 'train' split of the dataset
    train_dataset = dataset['train']

    sql_prompts = get_sql_prompts_for_domain(train_dataset, args.domain, int(args.samples))
    #print(sql_prompts)
    classifier = W5HClassifier(
        model="qwen2.5:7b-instruct-q4_K_M",
        temperature=0.1
    )

    for prompt in sql_prompts:
        profile = classifier.classify(prompt)
        print_profile(profile)