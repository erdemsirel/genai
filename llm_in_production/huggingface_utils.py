import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@torch.no_grad()  # disable gradient tracking
def get_probs_next_word_top_k(
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    text: str,
    k: int = 10,
    temperature: float = 1.0,
) -> tuple[list[str], list[float]]:
    """
    Get the top k words and their probabilities for the next word in the sequence.
    :param tokenizer: The tokenizer to use.
    :param model: The model to use.
    :param text: The text for which to predict the next word.
    :param k: The number of top words to consider. Needed to else the model will return to many words to visualize.
    :param temperature: The temperature to use for the softmax, the higher the more random the output.
    :return: A tuple of the top k words and their probabilities.
    """
    # Encode the text using the tokenizer
    tokens = tokenizer.encode_plus(text, return_tensors="pt")

    # Get the logits from the model
    outputs = model(**tokens)
    probs = softmax_with_temperature(outputs.logits, temperature=temperature)
    # Get the prob for the last word
    next_word_prob = probs[0, -1, :]

    # Get the top k words and their probabilities
    topk_probs, topk_indices = next_word_prob.topk(k)
    # Coverting to list of floats and ints
    topk_probs = topk_probs.detach().numpy().tolist()
    topk_tokens = topk_indices.detach().numpy().tolist()
    # Convert the tokens to words
    topk_words = [tokenizer.decode([token]) for token in topk_tokens]
    return topk_words, topk_probs


def softmax_with_temperature(
    x: torch.tensor, temperature: float = 1.0, eps: float = 1e-9, dim=-1
) -> torch.tensor:
    """
    Softmax with temperature.
    :param x: The input tensor.
    :param temperature: The temperature to use.
    :param eps: A small value to avoid division by zero.
    :param dim: The dimension to apply the softmax on.
    :return: The softmaxed tensor.
    """
    return torch.softmax(x / (temperature + eps), dim=dim)


@torch.no_grad()  # disable gradient tracking
def get_probs_next_word_top_p(
    tokenizer: GPT2Tokenizer,
    model: GPT2LMHeadModel,
    text: str,
    top_p: int = 0.15,
    temperature: float = 1.0,
):
    """
    Get the most likely words next words prediction for a given text.
    Only return words with cumulative probability of top_p.
    :param tokenizer: The tokenizer to use.
    :param model: The model to use.
    :param text: The text for which to predict the next word.
    :param top_p: The cumulative probability to use.
    :param temperature: The temperature to use for the softmax, the higher the more random the output.
    :return: A tuple of the top words and their probabilities.
    """
    # Encode the text using the tokenizer
    tokens = tokenizer.encode_plus(text, return_tensors="pt")

    # Get the logits from the model
    outputs = model(**tokens)
    probs = softmax_with_temperature(outputs.logits, temperature=temperature)
    # Get the prob for the last word
    next_word_prob = probs[0, -1, :]

    # Sort the probabilities and tokens
    sorted_probs, sorted_indices = torch.sort(next_word_prob, descending=True)
    # Calculate the cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # Create a boolean mask where cumulative probability is less than top_p
    mask = cumulative_probs < top_p

    # Use the mask to select the top tokens and their probabilities
    top_tokens = sorted_indices[mask].detach().numpy().tolist()
    top_probs = sorted_probs[mask].detach().numpy().tolist()

    # If no tokens are selected, select the first one
    if not top_tokens:
        top_tokens = [sorted_indices[0].item()]
        top_probs = [sorted_probs[0].item()]

    # Convert the tokens to words
    top_words = [tokenizer.decode([token]) for token in top_tokens]

    return top_words, top_probs


def get_device() -> str:
    if torch.cuda.is_available():
        devices = "cuda"
    elif torch.backends.mps.is_available():
        devices = "mps"
    else:
        devices = "cpu"

    return devices
