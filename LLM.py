""" This file contains the code for calling all LLM APIs. """

import os
from functools import partial
import tiktoken
# from schema import TooLongPromptError, LLMError
import os
from anthropic import Anthropic

anthropic_client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

enc = tiktoken.get_encoding("cl100k_base")

try:
    from helm.common.authentication import Authentication
    from helm.common.request import Request, RequestResult
    from helm.proxy.accounts import Account
    from helm.proxy.services.remote_service import RemoteService
    # setup CRFM API
    auth = Authentication(api_key=open("crfm_api_key.txt").read().strip())
    service = RemoteService("https://crfm-models.stanford.edu")
    account: Account = service.get_account(auth)
except Exception as e:
    print(e)
    print("Could not load CRFM API key crfm_api_key.txt.")

try:   
    import anthropic
    #setup anthropic API key
    anthropic_client = anthropic.Anthropic(api_key=open("claude_api_key.txt").read().strip())
except Exception as e:
    print(e)
    print("Could not load anthropic API key claude_api_key.txt.")

try:
    import openai
    from openai import OpenAI
    organization, api_key  =  open("openai_api_key.txt").read().strip().split(":")    
    os.environ["OPENAI_API_KEY"] = api_key 
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(e)
    print("Could not load OpenAI API key openai_api_key.txt.")


def log_to_file(log_file, prompt, completion, model, max_tokens_to_sample):
    """ Log the prompt and completion to a file."""
    with open(log_file, "a") as f:
        f.write("\n===================prompt=====================\n")
        f.write(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}")
        num_prompt_tokens = len(enc.encode(f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"))
        f.write(f"\n==================={model} response ({max_tokens_to_sample})=====================\n")
        f.write(completion)
        num_sample_tokens = len(enc.encode(completion))
        f.write("\n===================tokens=====================\n")
        f.write(f"Number of prompt tokens: {num_prompt_tokens}\n")
        f.write(f"Number of sampled tokens: {num_sample_tokens}\n")
        f.write("\n\n")


def complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT], model="claude-v1", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):
    """ Call the Claude API to complete a prompt."""
    import time

    ai_prompt = anthropic.AI_PROMPT
    if "ai_prompt" in kwargs and kwargs["ai_prompt"] is not None:
        ai_prompt = kwargs["ai_prompt"]
        del kwargs["ai_prompt"]
    
    # Handle Claude 3 models (new Messages API)
    if model.startswith("claude-3"):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # For Claude 3, don't use HUMAN_PROMPT/AI_PROMPT tokens
                messages = [
                    {'role': 'user', 'content': prompt}
                ]
                
                rsp = anthropic_client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens_to_sample,
                    temperature=temperature,
                    stop_sequences=stop_sequences if stop_sequences != [anthropic.HUMAN_PROMPT] else [],
                    **kwargs
                )
                
                completion = rsp.content[0].text
                if log_file is not None:
                    log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
                return completion
                
            except anthropic._exceptions.OverloadedError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Claude API overloaded. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Claude API overloaded after {max_retries} attempts.")
                    raise e
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Claude API rate limit hit. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Claude API rate limit exceeded after {max_retries} attempts.")
                    raise e
            except anthropic.APIStatusError as e:
                print(f"Claude API error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise TooLongPromptError()
            except Exception as e:
                print(f"Unexpected error calling Claude API: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise LLMError(e)
    
    # Handle older Claude models (legacy Completions API)
    else:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                rsp = anthropic_client.completions.create(
                    prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {ai_prompt}",
                    stop_sequences=stop_sequences,
                    model=model,
                    temperature=temperature,
                    max_tokens_to_sample=max_tokens_to_sample,
                    **kwargs
                )
                
                completion = rsp.completion
                if log_file is not None:
                    log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
                return completion
                
            except anthropic.OverloadedError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Claude API overloaded. Retrying in {wait_time} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"Claude API overloaded after {max_retries} attempts.")
                    raise e
            except anthropic.APIStatusError as e:
                print(f"Claude API error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise TooLongPromptError()
            except Exception as e:
                print(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    raise LLMError(e)


def get_embedding_crfm(text, model="openai/gpt-4-0314"):
    request = Request(model="openai/text-similarity-ada-001", prompt=text, embedding=True)
    request_result: RequestResult = service.make_request(auth, request)
    return request_result.embedding 

def complete_text_crfm(prompt=None, stop_sequences = None, model="openai/gpt-4-0314",  max_tokens_to_sample=2000, temperature = 0.5, log_file=None, messages = None, **kwargs):

    random = log_file
    if messages:
        request = Request(
                prompt=prompt, 
                messages=messages,
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
            )
    else:
        print("model", model)
        print("max_tokens", max_tokens_to_sample)
        request = Request(
                prompt=prompt, 
                model=model, 
                stop_sequences=stop_sequences,
                temperature = temperature,
                max_tokens = max_tokens_to_sample,
                random = random
        )

    try:      
        request_result: RequestResult = service.make_request(auth, request)
    except Exception as e:
        # probably too long prompt
        print(e)
        exit()
        # raise TooLongPromptError()

    if request_result.success == False:
        print(request.error)
        # raise LLMError(request.error)
    completion = request_result.completions[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion


def complete_text_openai(prompt, stop_sequences=[], model="gpt-3.5-turbo", max_tokens_to_sample=2000, temperature=0.5, log_file=None, **kwargs):

    """ Call the OpenAI API to complete a prompt."""
    raw_request = {
          "model": model,
        #   "temperature": temperature,
        #   "max_completion_tokens": max_tokens_to_sample,
        #   "stop": stop_sequences or None,  # API doesn't like empty list
          **kwargs
    }
    if model.startswith("gpt-3.5") or model.startswith("gpt-4") or model.startswith("o1"):
        # Requires openai==1.42.0
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(**{"messages": messages,**raw_request})
        completion = response.choices[0].message.content
    else:
        response = client.completions.create(**{"prompt": prompt,**raw_request})
        completion = response.choices[0].text
    if log_file is not None:
        log_to_file(log_file, prompt, completion, model, max_tokens_to_sample)
    return completion

def complete_text(prompt, log_file, model, **kwargs):
    """ Complete text using the specified model with appropriate API. """

    if model.startswith("claude"):
        # use anthropic API
        completion = complete_text_claude(prompt, stop_sequences=[anthropic.HUMAN_PROMPT, "Observation:"], log_file=log_file, model=model, **kwargs)
    elif "/" in model:
        # use CRFM API since this specifies organization like "openai/..."
        completion = complete_text_crfm(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    else:
        # use OpenAI API
        completion = complete_text_openai(prompt, stop_sequences=["Observation:"], log_file=log_file, model=model, **kwargs)
    return completion

# specify fast models for summarization etc
FAST_MODEL = "claude-3-haiku-20240307"
def complete_text_fast(prompt, **kwargs):
    return complete_text(prompt = prompt, model = FAST_MODEL, temperature =0.01, **kwargs)