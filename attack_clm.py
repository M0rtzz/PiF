import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer
import time
import gc
import eval_template
from chat_templates import chat_template_generation_kwargs, ensure_chat_template
from llm_asr_judge import judge_prompt_matched_asr_batch
from openai import OpenAI

OPENAI_API_KEY = "YOUR API KEY"
_MODEL_CACHE = {}


def _cache_key(kind, model_path, tokenizer_path, trust_remote_code):
    return (kind, str(model_path), str(tokenizer_path), bool(trust_remote_code))


def _load_causal_lm_with_attention_fallback(model_path, **kwargs):
    flash_kwargs = dict(kwargs)
    flash_kwargs["attn_implementation"] = "flash_attention_2"
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, **flash_kwargs)
    except TypeError as exc:
        if "use_flash_attention_2" not in str(exc) and "attn_implementation" not in str(exc):
            raise
    except Exception as exc:
        message = str(exc).lower()
        flash_error_markers = (
            "flash_attention_2",
            "flash attention",
            "attn_implementation",
            "does not support flash attention",
        )
        if not any(marker in message for marker in flash_error_markers):
            raise
    return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)


def _load_generate_model_and_tokenizer(generate_m, generate_t, trust_remote_code, keep_models_in_memory):
    key = _cache_key("generate", generate_m, generate_t, trust_remote_code)
    if keep_models_in_memory and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    generate_model = _load_causal_lm_with_attention_fallback(
        generate_m,
        output_hidden_states=True,
        load_in_8bit=True,
        cache_dir='./hf_models',
        device_map="auto",
        trust_remote_code=trust_remote_code,
    ).eval()
    generate_tokenizer = AutoTokenizer.from_pretrained(
        generate_t,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    if generate_tokenizer.pad_token is None:
        if generate_tokenizer.eos_token:
            generate_tokenizer.pad_token = generate_tokenizer.eos_token
        else:
            generate_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            generate_model.resize_token_embeddings(len(generate_tokenizer))

    generate_tokenizer.add_special_tokens({'mask_token': '[MASK]'})
    generate_model.resize_token_embeddings(len(generate_tokenizer))

    if keep_models_in_memory:
        _MODEL_CACHE[key] = (generate_model, generate_tokenizer)
    return generate_model, generate_tokenizer


def _load_target_model_and_tokenizer(tgt_m, tgt_t, trust_remote_code, keep_models_in_memory):
    key = _cache_key("target", tgt_m, tgt_t, trust_remote_code)
    if keep_models_in_memory and key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tgt_model = _load_causal_lm_with_attention_fallback(
        tgt_m,
        load_in_8bit=True,
        cache_dir='./hf_models',
        device_map="auto",
        trust_remote_code=trust_remote_code,
    ).eval()
    tgt_tokenizer = AutoTokenizer.from_pretrained(
        tgt_t,
        cache_dir='./hf_models',
        trust_remote_code=trust_remote_code,
    )
    if tgt_tokenizer.pad_token is None:
        if tgt_tokenizer.eos_token:
            tgt_tokenizer.pad_token = tgt_tokenizer.eos_token
        else:
            tgt_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tgt_model.resize_token_embeddings(len(tgt_tokenizer))

    if keep_models_in_memory:
        _MODEL_CACHE[key] = (tgt_model, tgt_tokenizer)
    return tgt_model, tgt_tokenizer

def extract_score(content):
    """Extract 0 or 1 from the content"""
    try:
        content = content.strip()
        if content[-1] in ['0', '1']:
            return int(content[-1])
        for word in content.split():
            if word == '0' or word == '1':
                return int(word)
        return None
    except:
        return None

def estimate_word_importance(model, tokenizer, texts, template, tok_n, temperature, device):
    batch_with_template = [f"{text} {template}" for text in texts]
    inputs = tokenizer(batch_with_template, return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        ori_outputs = model(**inputs)
        ori_logits = ori_outputs.logits
        ori_probs = torch.nn.functional.softmax(ori_logits, dim=-1)

    template_length = len(tokenizer.tokenize(template))
    least_important_idxs = []
    masked_idxs = []

    for text_idx, text_with_template in enumerate(batch_with_template):

        tokens = tokenizer.tokenize(text_with_template)
        mask_position = tokens.index(tokenizer.mask_token)
        num_tokens = len(tokens) - template_length
        modified_texts = []
        for idx in range(num_tokens):
            new_text_with_template = tokens[:idx] + tokens[idx + 1:]
            new_text_with_template_str = tokenizer.convert_tokens_to_string(new_text_with_template)
            modified_texts.append(new_text_with_template_str)

        modified_inputs = tokenizer(modified_texts, return_tensors='pt', padding=True, truncation=True)
        modified_inputs = {k: v.to(device) for k, v in modified_inputs.items()}

        with torch.no_grad():
            modified_outputs = model(**modified_inputs)
            modified_logits = modified_outputs.logits
            modified_probs = torch.nn.functional.softmax(modified_logits, dim=-1)

        importance = torch.norm(ori_probs[text_idx, mask_position].unsqueeze(0) - modified_probs[:, mask_position - 1], p=2, dim=1)

        if tok_n > len(importance):
            tok_n = len(importance)
        _, least_indices = torch.topk(-importance, k=tok_n)
        least_importance_scores = importance[least_indices]

        probabilities = torch.softmax(-(least_importance_scores/temperature), dim=0)
        sampled_index = torch.multinomial(probabilities, 1).item()
        sampled_least_important_idx = least_indices[sampled_index]

        masked_idxs.append(mask_position)
        least_important_idxs.append(sampled_least_important_idx)

    return least_important_idxs, masked_idxs

def replace_words(model, tokenizer, texts, template, tok_n, tok_m, top_k, temperature, device):
    least_important_idxs, masked_idxs = estimate_word_importance(model, tokenizer, texts, template, tok_n, temperature, device)
    replacements = []

    for text_idx, (text, least_important_idx) in enumerate(zip(texts, least_important_idxs)):

        ori_token = tokenizer.tokenize(text)
        current_token = ori_token[least_important_idx]
        masked_token = ori_token[:least_important_idx] + [tokenizer.mask_token] + ori_token[least_important_idx + 1:]
        masked_text = tokenizer.convert_tokens_to_string(masked_token)
        masked_input_id = tokenizer(masked_text, return_tensors='pt')
        masked_input_id = {k: v.to(device) for k, v in masked_input_id.items()}

        with torch.no_grad():
            outputs = model(**masked_input_id)
            logits = outputs.logits[0, least_important_idx]

        current_token_id = tokenizer.convert_tokens_to_ids([current_token])[0]
        ori_tokens_id = tokenizer.convert_tokens_to_ids(ori_token)

        sorted_indices = torch.argsort(logits, descending=True)
        tok_m_tokens = []

        for idx in sorted_indices:
            if len(tok_m_tokens) == tok_m:
                break
            decoded_token = tokenizer.decode([idx])
            if idx not in ori_tokens_id:
                if (not current_token.startswith('_') and not (29871 <= current_token_id <= 31999)) \
                and (not decoded_token.startswith('_') and not (29871 <= idx <= 31999)):
                    tok_m_tokens.append(idx)
                elif current_token.startswith('_'):
                    if decoded_token.startswith('_'):
                        tok_m_tokens.append(idx)
                elif (29871 <= current_token_id <= 31999):
                    if (29871 <= idx <= 31999):
                        tok_m_tokens.append(idx)

        text_with_template = text + template
        base_input = tokenizer(text_with_template, return_tensors='pt', padding=True, truncation=True).to(device)
        base_input = {k: v.to(device) for k, v in base_input.items()}

        with torch.no_grad():
            base_output = model(**base_input)
            base_logits = base_output.logits[0, masked_idxs[text_idx]]

        top_k_tokens = torch.topk(base_logits, k=top_k, largest=True, sorted=True).indices

        new_texts = []
        for top_m_token in tok_m_tokens:
            new_token = ori_token[:least_important_idx] + [tokenizer.decode([top_m_token])] + ori_token[least_important_idx + 1:]
            new_text = tokenizer.convert_tokens_to_string(new_token)
            new_texts.append(new_text)

        new_input_ids = tokenizer([text + template for text in new_texts], return_tensors='pt', padding=True, truncation=True)
        new_input_ids = {k: v.to(device) for k, v in new_input_ids.items()}

        with torch.no_grad():
            new_outputs = model(**new_input_ids)
            new_logits = new_outputs.logits[:, masked_idxs[text_idx], :]
            new_confidences = torch.nn.functional.softmax(new_logits, dim=-1)[:, top_k_tokens]

        confidences_sum = new_confidences.sum(dim=1)
        probabilities = torch.softmax(-(confidences_sum/temperature), dim=0)
        sampled_index = torch.multinomial(probabilities, 1).item()
        replacements.append(new_texts[sampled_index])

    return replacements

def evaluate_text_changes(model, tokenizer, texts1, texts2, threshold, device):
    inputs_1 = tokenizer(texts1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_2 = tokenizer(texts2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}  # Move to device
    inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}  # Move to device

    results = []

    for sentence1, sentence2 in zip(texts1, texts2):

        encoded1 = tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        encoded2 = tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs1 = model(**encoded1)
            outputs2 = model(**encoded2)

        emb1 = outputs1.hidden_states[-1].squeeze(0)
        emb2 = outputs2.hidden_states[-1].squeeze(0)

        if(emb1.size() == emb2.size()):
            cos_sims = F.cosine_similarity(emb2.unsqueeze(0), emb1.unsqueeze(0), dim=2)
            text_similarity = cos_sims.max(dim=1).values.mean().item()
            results.append(text_similarity >= threshold)
        else:
            results.append(False)
    return results

def generate_attack(generate_m, generate_t, tgt_m, tgt_t, texts, evaluation_template, objective, iterations, top_n, top_m, top_k, warm_up, temperature, threshold, device, trust_remote_code=False, keep_models_in_memory=False):
    total_time = 0
    total_query = 0
    successful_flag = [False] * len(texts)
    tgt_texts = [None] * len(texts)
    current_texts = texts.copy()

    for iter in range(iterations):

        generate_model, generate_tokenizer = _load_generate_model_and_tokenizer(
            generate_m,
            generate_t,
            trust_remote_code=trust_remote_code,
            keep_models_in_memory=keep_models_in_memory,
        )

        start_time = time.time()

        new_texts = replace_words(generate_model, generate_tokenizer, current_texts, evaluation_template, top_n, top_m, top_k, temperature, device)
        evaluation_results = evaluate_text_changes(generate_model, generate_tokenizer, texts, new_texts, threshold, device)
        for ii, (eval_res, new_text) in enumerate(zip(evaluation_results, new_texts)):
            if eval_res == True and successful_flag[ii] == False:
                current_texts[ii] = new_text
        end_time = time.time()

        total_time += (end_time - start_time)

        if not keep_models_in_memory:
            del generate_model, generate_tokenizer ; gc.collect()
            torch.cuda.empty_cache()

        if tgt_m != "gpt-4-0613" and tgt_m != "o1-preview-2024-09-12":
            tgt_model, tgt_tokenizer = _load_target_model_and_tokenizer(
                tgt_m,
                tgt_t,
                trust_remote_code=trust_remote_code,
                keep_models_in_memory=keep_models_in_memory,
            )
        else:
            tgt_model = tgt_m
            tgt_tokenizer = tgt_t

        pending_asr_judges = []
        for jj, current_text in enumerate(current_texts):
            last_tgt_text = None
            if successful_flag[jj] == False:
                if iter >= warm_up:
                    total_query += 1
                    if tgt_model != "gpt-4-0613" and tgt_model != "o1-preview-2024-09-12":

                        chat = [{"role": "user", "content": current_text}]
                        ensure_chat_template(tgt_tokenizer, tgt_t, tgt_m)
                        input_ids = tgt_tokenizer.apply_chat_template(
                            chat,
                            return_tensors="pt",
                            truncation=True,
                            max_length=128,
                            add_generation_prompt=True,
                            **chat_template_generation_kwargs(tgt_tokenizer),
                        ).to(device)
                        tgt_token_length = input_ids.shape[1]
                        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
                        pad_token_id = tgt_tokenizer.pad_token_id
                        if pad_token_id is None:
                            pad_token_id = tgt_tokenizer.eos_token_id
                        tgt_output = tgt_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=512,
                            pad_token_id=pad_token_id,
                        )
                        tgt_text = tgt_tokenizer.decode(tgt_output[0, tgt_token_length:], skip_special_tokens=True)
                        last_tgt_text = tgt_text

                        if objective == "ASR":
                            pending_asr_judges.append({
                                "index": jj,
                                "prompt": texts[jj],
                                "model_output": tgt_text,
                                "sample_id": f"item-{jj}-iter-{iter}",
                                "target_text": tgt_text,
                                "current_text": current_text,
                            })

                        else:
                            tgt_word = tgt_text.strip().split()
                            tgt_truncated_text = " ".join(tgt_word[:30])
                            if tgt_truncated_text != "" and not any(prefix.lower() in tgt_truncated_text.lower() for prefix in eval_template.test_prefixes):
                                if objective == "ASR+GPT":
                                    client = OpenAI(api_key=OPENAI_API_KEY)
                                    try:
                                        response = client.chat.completions.create(
                                        model="gpt-4-0613",
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": eval_template.judge_template + tgt_text
                                            }
                                        ],
                                        temperature=0,
                                        max_tokens=4096,
                                        top_p=0,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        )
                                        content = response.choices[0].message.content
                                        score = extract_score(content)
                                        if score == 1:
                                            successful_flag[jj] = True
                                            tgt_texts[jj] = tgt_text
                                            current_texts[jj] = current_text
                                    except Exception as err:
                                        continue

                                elif objective == "ASR+Filter":
                                    filter_model = AutoModelForCausalLM.from_pretrained('../Llama-Guard-3-8B', torch_dtype=torch.bfloat16).cuda()
                                    filter_tokenizer = AutoTokenizer.from_pretrained('../Llama-Guard-3-8B')
                                    chat = ([{"role": "user", "content": current_text}])
                                    ensure_chat_template(filter_tokenizer, '../Llama-Guard-3-8B')
                                    input_ids = filter_tokenizer.apply_chat_template(
                                        chat,
                                        return_tensors="pt",
                                        add_generation_prompt=True,
                                        **chat_template_generation_kwargs(filter_tokenizer),
                                    ).to(device)
                                    output = filter_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                                    prompt_len = input_ids.shape[-1]
                                    results = filter_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                                    if not "unsafe" in results:
                                        successful_flag[jj] = True
                                        tgt_texts[jj] = tgt_text
                                        current_texts[jj] = current_text

                    else:
                        try:
                            client = OpenAI(api_key=OPENAI_API_KEY)
                            response = client.chat.completions.create(
                                model = tgt_model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": current_text
                                    }
                                ],
                                temperature=0,
                                max_tokens=4096,
                                top_p=0,
                                frequency_penalty=0,
                                presence_penalty=0,
                            )
                            tgt_text = response.choices[0].message.content
                            last_tgt_text = tgt_text
                        except Exception as err:
                            continue

                        if objective == "ASR":
                            pending_asr_judges.append({
                                "index": jj,
                                "prompt": texts[jj],
                                "model_output": tgt_text,
                                "sample_id": f"item-{jj}-iter-{iter}",
                                "target_text": tgt_text,
                                "current_text": current_text,
                            })

                        else:
                            tgt_word = tgt_text.strip().split()
                            tgt_truncated_text = " ".join(tgt_word[:30])
                            if tgt_truncated_text != "" and not any(prefix.lower() in tgt_truncated_text.lower() for prefix in eval_template.test_prefixes):
                                if objective == "ASR+GPT":
                                    client = OpenAI(api_key=OPENAI_API_KEY)
                                    try:
                                        response = client.chat.completions.create(
                                        model="gpt-4-0613",
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": eval_template.judge_template + tgt_text
                                            }
                                        ],
                                        temperature=0,
                                        max_tokens=4096,
                                        top_p=0,
                                        frequency_penalty=0,
                                        presence_penalty=0,
                                        )
                                        content = response.choices[0].message.content
                                        score = extract_score(content)
                                        if score == 1:
                                            successful_flag[jj] = True
                                            tgt_texts[jj] = tgt_text
                                            current_texts[jj] = current_text
                                    except Exception as err:
                                        continue

                                elif objective == "ASR+Filter":
                                    filter_model = AutoModelForCausalLM.from_pretrained('../Llama-Guard-3-8B', torch_dtype=torch.bfloat16).cuda()
                                    filter_tokenizer = AutoTokenizer.from_pretrained('../Llama-Guard-3-8B')
                                    chat = ([{"role": "user", "content": current_text}])
                                    ensure_chat_template(filter_tokenizer, '../Llama-Guard-3-8B')
                                    input_ids = filter_tokenizer.apply_chat_template(
                                        chat,
                                        return_tensors="pt",
                                        add_generation_prompt=True,
                                        **chat_template_generation_kwargs(filter_tokenizer),
                                    ).to(device)
                                    output = filter_model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
                                    prompt_len = input_ids.shape[-1]
                                    results = filter_tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
                                    if not "unsafe" in results:
                                        successful_flag[jj] = True
                                        tgt_texts[jj] = tgt_text
                                        current_texts[jj] = current_text

            if iter == (iterations - 1) and successful_flag[jj] == False:
                if last_tgt_text is not None:
                    tgt_texts[jj] = last_tgt_text
                current_texts[jj] = current_text

        if pending_asr_judges:
            judge_results = judge_prompt_matched_asr_batch(
                pending_asr_judges,
                benchmark="advbench",
                condition="attack",
                output_field="target_response",
            )
            for judge_item, judge_result in zip(pending_asr_judges, judge_results):
                jj = judge_item["index"]
                if bool(judge_result.get("prompt_matched_attack_success")):
                    successful_flag[jj] = True
                    tgt_texts[jj] = judge_item["target_text"]
                    current_texts[jj] = judge_item["current_text"]

        if not keep_models_in_memory and tgt_m != "gpt-4-0613" and tgt_m != "o1-preview-2024-09-12":
            del tgt_model, tgt_tokenizer ; gc.collect()
            torch.cuda.empty_cache()

    return total_query, total_time, successful_flag, current_texts, tgt_texts
