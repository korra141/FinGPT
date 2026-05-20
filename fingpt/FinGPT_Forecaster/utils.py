import re
import os
import datasets
from sklearn.metrics import accuracy_score, mean_squared_error
from collections import defaultdict
from rouge_score import rouge_scorer
from bert_score import score as _bert_score


lora_module_dict = {
    'chatglm2': ['query_key_value'],
    'llama2': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
        # 'embed_tokens', 'lm_head',
    ],
    'llama3': [
        'q_proj', 'k_proj', 'v_proj',
        'o_proj', 'gate_proj', 'up_proj', 'down_proj',
    ],
}
     


def tokenize(args, tokenizer, feature):
    
    prompt_ids = tokenizer.encode(
        feature['prompt'].strip(), padding=False,
        max_length=args.max_length, truncation=True
    )
    
    target_ids = tokenizer.encode(
        feature['answer'].strip(), padding=False,
        max_length=args.max_length, truncation=True, add_special_tokens=False
    )
    
    input_ids = prompt_ids + target_ids
    exceed_max_length = len(input_ids) >= args.max_length
    
     # Add EOS Token
    if input_ids[-1] != tokenizer.eos_token_id and not exceed_max_length:
        input_ids.append(tokenizer.eos_token_id)
    
    label_ids = [tokenizer.pad_token_id] * len(prompt_ids) + input_ids[len(prompt_ids):]
    
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "exceed_max_length": exceed_max_length
    }


def parse_model_name(name, from_remote=False):
    
    if name == 'chatglm2':
        return 'THUDM/chatglm2-6b' if from_remote else 'base_models/chatglm2-6b'
    elif name == 'llama2':
        return 'meta-llama/Llama-2-7b-chat-hf' # if from_remote else 'base_models/Llama-2-7b-chat-hf'
    elif name == 'llama3':
        return 'meta-llama/Meta-Llama-3-8B' # if from_remote else 'base_models/Llama-3-7b-chat-hf'
    else:
        raise ValueError(f"Undefined base model {name}")
        
    
def load_dataset(names, from_remote=False):
    
    dataset_names = [d for d in names.split(',')]
    dataset_list = []
    
    for name in dataset_names:
        rep = 1
        if not os.path.exists(name):
            rep = int(name.split('*')[1]) if '*' in name else 1
            name = ('FinGPT/fingpt-forecaster-' if from_remote else 'data/fingpt-forecaster-') + name.split('*')[0]
        tmp_dataset = datasets.load_dataset(name) if from_remote else datasets.load_from_disk(name)
    
        if 'test' not in tmp_dataset:
            tmp_dataset = tmp_dataset.train_test_split(0.2, shuffle=True, seed=42)   
        dataset_list.extend([tmp_dataset] * rep)
    
    return dataset_list


def parse_answer_base(answer):
    """
    Parse output that uses plain (non-bracketed) section headers:

        Positive Developments:
        1. ...

        Potential Concerns:
        1. ...

        Prediction & Analysis:
        Based on ... we predict ... [single paragraph]
    """
    section_re  = re.compile(r'\[?Positive Developments\]?\s*:',          re.IGNORECASE)
    concerns_re = re.compile(r'\[?Potential Concerns\]?\s*:',              re.IGNORECASE)
    pna_re      = re.compile(r'\[?Prediction\s*[&and]+\s*Analysis\]?\s*:', re.IGNORECASE)
    note_re     = re.compile(r'\n\s*Note\s*:',                             re.IGNORECASE)

    pos_match = section_re.search(answer)
    con_match = concerns_re.search(answer)
    pna_match = pna_re.search(answer)

    if not (pos_match and con_match and pna_match):
        return None

    pros     = answer[pos_match.end():con_match.start()].strip()
    cons     = answer[con_match.end():pna_match.start()].strip()
    pna_text = answer[pna_match.end():].strip()

    note_match = note_re.search(pna_text)
    if note_match:
        pna_text = pna_text[:note_match.start()].strip()

    if not pna_text:
        return None

    pna_lower = pna_text.lower()
    if re.search(r'\bup\b|increase|rise|higher|outperform|bullish|upside', pna_lower):
        pred_bin = 1
    elif re.search(r'\bdown\b|decrease|decline|fall|lower|bearish|downside', pna_lower):
        pred_bin = -1
    else:
        pred_bin = 0

    match_pct = re.search(r'(\d+)\s*-\s*(\d+)\s*%', pna_text)
   
    if match_pct and match_pct.lastindex == 2:
        pred_margin = pred_bin * (float(match_pct.group(1)) + float(match_pct.group(2))) / 2
    elif match_pct:
        pred_margin = pred_bin * (float(match_pct.group(1)) + 0.5)
    else:
        match_pct = re.search(r'(?:more than\s+)?(\d+)\s*%', pna_text)
        pred_margin = pred_bin * (int(match_pct.group(1)) + 0.5) if match_pct else 0.0

        
    return {
        "positive developments": pros,
        "potential concerns":    cons,
        "prediction":            pred_margin,
        "prediction_binary":     pred_bin,
        "analysis":              pna_text,
    }


def parse_answer(answer):
    
    match_res = re.match(r"^\s*\[Positive Developments\]:?\s*(.*)\s*\[Potential Concerns\]:?\s*(.*)\s*-*\s*\[Prediction (&|and) Analysis\]:?\s*(.*)\s*$", answer, flags=re.DOTALL)
    if not match_res:
        return None
    
    pros, cons, pna = match_res.group(1), match_res.group(2), match_res.group(4)
        
    match_res = re.match(r'^Prediction:\s*(.*)\s*Analysis:\s*(.*)\s*$', pna, flags=re.DOTALL)
    if not match_res:
        return None
        
    pred, anal = match_res.group(1), match_res.group(2)
        
    if re.search(r'up|increase', pred.lower()):
        pred_bin = 1
    elif re.search(r'down|decrease|decline', pred.lower()):
        pred_bin = -1
    else:
        pred_bin = 0
            
    match_res = re.search(r'(\d)-(\d)%', pred)
    if not match_res:
        match_res = re.search(r'(?:more than )?(\d)+?%', pred)    
        
    if match_res and match_res.lastindex == 2:
        pred_margin = pred_bin * (int(match_res.group(1)) + int(match_res.group(2))) / 2
    elif match_res:
        pred_margin = pred_bin * (int(match_res.group(1)) + 0.5)
    else:
        pred_margin = 0.
        
    return {
        "positive developments": pros,
        "potential concerns": cons,
        "prediction": pred_margin,
        "prediction_binary": pred_bin,
        "analysis": anal
    }
    

def calc_bert_score(references, candidates, lang="en"):
    P, R, F1 = _bert_score(candidates, references, lang=lang, verbose=False)
    return {
        'precision': round(P.mean().item(), 4),
        'recall':    round(R.mean().item(), 4),
        'f1':        round(F1.mean().item(), 4),
    }


def calc_rouge_score(references, answers):
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    scores_per_pair = [scorer.score(ref, ans) for ref, ans in zip(references, answers)]
    
    rouge1 = sum(score['rouge1'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rouge2 = sum(score['rouge2'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    rougeL = sum(score['rougeL'].fmeasure for score in scores_per_pair) / len(scores_per_pair)
    
    return {'rouge1': rouge1, 'rouge2': rouge2, 'rougeL': rougeL}

    
def calc_metrics(answers, gts):
    
    answers_dict = defaultdict(list)
    gts_dict = defaultdict(list)
    
    for answer, gt in zip(answers, gts):
        answer_dict = parse_answer(answer)
        gt_dict = parse_answer(gt)
        
        if answer_dict and gt_dict:
            for k in answer_dict.keys():
                answers_dict[k].append(answer_dict[k])
                gts_dict[k].append(gt_dict[k])
    
    if not answers_dict['prediction']:
        return {}
    
    bin_acc = accuracy_score(gts_dict['prediction_binary'], answers_dict['prediction_binary'])
    mse = mean_squared_error(gts_dict['prediction'], answers_dict['prediction'])
    
    pros_rouge_scores = calc_rouge_score(gts_dict['positive developments'], answers_dict['positive developments'])
    cons_rouge_scores = calc_rouge_score(gts_dict['potential concerns'], answers_dict['potential concerns'])
    anal_rouge_scores = calc_rouge_score(gts_dict['analysis'], answers_dict['analysis'])
                              
    print(f"\nBinary Accuracy: {bin_acc:.2f}  |  Mean Square Error: {mse:.2f}")
    print(f"\nRouge Score of Positive Developments: {pros_rouge_scores}")
    print(f"\nRouge Score of Potential Concerns: {cons_rouge_scores}")
    print(f"\nRouge Score of Summary Analysis: {anal_rouge_scores}")
                              
    return {
        "valid_count": len(answers_dict['prediction']),
        "bin_acc": bin_acc,
        "mse": mse,
        "pros_rouge_scores": pros_rouge_scores,
        "cons_rouge_scores": cons_rouge_scores,
        "anal_rouge_scores": anal_rouge_scores
    }
    
