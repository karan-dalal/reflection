import os
import torch
import random
import json
import gzip
from transformers import LlamaTokenizer, LlamaForCausalLM

os.environ["HF_HOME"] = "self-reflection/cache"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = LlamaTokenizer.from_pretrained("models/vicuna", cache_dir=os.environ["HF_HOME"])
model = LlamaForCausalLM.from_pretrained("models/vicuna", cache_dir=os.environ["HF_HOME"]).to(device)

critique = [
    "Identify specific ways in which the assistant’s last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.",
    "Explain ways in which the assistant’s proceeding response may be harmful to either the human or others. For example, if the assistant says anything that encourages dangerous or illegal activity, make sure to point it out. Other examples of harm include racism, sexism, and other social biases.",
    "Identify all ways in which the assistant’s last response is harmful, unethical, or socially biased. Furthermore, provide specific details on how the assistant can improve its response.",
    "Identify specific ways in which the assistant’s proceeding response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity.",
    "Discuss the assistant’s last response to the human and whether it is in any way insensitive, sexist, racist, or socially inappropriate.",
    "Discuss whether the assistant said anything that may be interpreted as offensive or harmful. If possible, also point out any harmful assumptions in the human’s dialogue.",
    "What’s the strongest argument you could give that the assistant’s last response was in some way harmful, unethical, racist, sexist, toxic, dangerous, or illegal.",
    "Did the assistant’s proceeding response include anything that may be inappropriate for young children? If so, discuss how it can be revised to be more appropriate.",
    "Think carefully about whether the human’s request succeeded in eliciting responses from the assistant that are derogatory, toxic, racist, sexist or socially harmful in any way, and discuss how the assistant should’ve responded instead.",
    "Discuss whether the assistant provided any advice that may encourage illegal or dangerous activity from either the human or others.",
    "Discuss ways in which the assistant’s response may be considered controversial or objectionable based on commonsense ethical and moral standards.",
    "Discuss whether the human is deliberately trying to bait the assistant into generating harmful content, and whether they succeeded in doing so.",
    "Please comment on whether the assistant’s last response is thoughtful and empathetic. Does it seem like the kind of thing a sensitive friend or therapist might say? Give the strongest argument you can for how it could improve.",
    "Take a look at the assistant’s last response and note if it was in any way misogynistic or otherwise gender biased. Give the strongest argument you can that something the assistant said could be interpreted as misogynistic.",
    "Read the assistant’s last response carefully and identify anything it said that provides advice or assistance in potentially criminal activities, such as violence, theft, hacking, robbery, or any other illegal activity. Consider the strongest argument you could give that the assistant response might encourage crime.",
    "Critique the assistant’s last response and identify anything it said that was harmful, unethical, dangerous, or illegal."
    ]
revision = [
    "Please rewrite the assistant's response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.",
    "Please rewrite the assistant's response to remove any and all harmful content, including anything dangerous, illegal, racist, sexist, or socially biased.",
    "Please rewrite the assistant's response to remove all harmful, unethical, or socially biased content, and move the conversation in a positive direction.",
    "Rewrite the assistant's response to remove all harmful content, and politely point out any harmful assumptions from the human, such as racism, sexism, toxicity.",
    "Revise the assistant’s last response to remove content that is in any way insensitive, sexist, racist, or socially inappropriate.",
    "Please write a new version of the assistant’s response that removes offensive and harmful content, and politely addresses any problematic assumptions from the human.",
    "Amend the assistant’s last response to remove any and all harmful, unethical, racist, toxic, dangerous, and illegal content.",
    "Please rewrite the assistant's response to make it more age-appropriate for children.",
    "Please rewrite the assistant’s response so as to remove all harmful content, especially anything derogatory, toxic, racist, sexist, or socially harmful.",
    "Revise the assistant’s response so as to refrain from giving legally questionable or dangerous advice.",
    "Revise the assistant’s response if possible so that it’s less controversial and objectionable based on commonsense ethical and moral standards.",
    "Write a revision of the assistant’s response that refrains from saying anything harmful.",
    "Please write a new assistant response so that it is more like what a thoughtful, empathetic, caring, sensitive friend or therapist would say.",
    "Please write a new response that does not have any trace of misogyny or gender bias.",
    "Rewrite the assistant's response to avoid assisting the human with criminal plans or activities, such as violence, theft, hacking, robbery, or anything illegal.",
    "Please rewrite the assistant's response to remove any and all harmful, unethical, dangerous, or illegal content."
    ]
iters = 2

def load_data(file_path):
    prompts = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for obj in data:
            human_statement = obj['transcript'].split("\n\n")[1]
            human_statement = human_statement.replace("Human: ", "")
            prompts.append(human_statement)
    return prompts

def generate_prompt(input: str) -> str:
    template = "A chat between a user and an assistant.\n\nUSER: {user_message}\nASSISTANT:"
    return template.format(user_message=input)

def generate_critique(critique: str, history: list) -> str:
    # template = "A chat between a user and an assistant.\n\nUSER: Prompt: {prompt} Previous Assistant's Response: {old_response}\n{critique}\nASSISTANT:"
    template = "A chat between a user and an assistant.\n\nUSER: {prompt}\nASSISTANT: {old_response}\nUSER: {critique}\nASSISTANT:" # Conversation Style
    return template.format(prompt=history[0], old_response=history[1], critique=critique)

def generate_revision(revision: str, history: list) -> str:
    # template = "A chat between a user and an assistant.\n\nUSER: Prompt: {prompt} Previous Assistant's Response: {old_response} Critique: {critique_response}\n{revision}\nASSISTANT:"
    template = "A chat between a user and an assistant.\n\nUSER: {prompt}\nASSISTANT: {old_response}\nUSER: {critique}\nASSISTANT: {critique_response}\nUSER: {revision}\nASSISTANT:" # Conversation Style
    return template.format(prompt=history[0], old_response=history[1], critique=history[2], critique_response=history[3], revision=revision)

def run_experiment(prompts, iters):
    counter = 0
    transcript = []
    for prompt in prompts:
        history = []
        indices = list(range(len(critique)))
        index_sample = random.sample(indices, iters)

        print(f"Prompt {counter}: ", prompt)
        vicuna_prompt = generate_prompt(prompt)
        inputs = tokenizer(vicuna_prompt, return_tensors="pt").to(device)
        out = model.generate(inputs.input_ids, max_new_tokens=256, do_sample=True, temperature=1.0, top_p=1.0,)
        output = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        response = output[len(vicuna_prompt):].lstrip()
        history.extend([prompt, response])

        for _ in range(iters):
            index = index_sample.pop(0)

            critique_prompt = generate_critique(critique[index], history)
            critique_inputs = tokenizer(critique_prompt, return_tensors="pt").to(device)
            critique_out = model.generate(critique_inputs.input_ids, max_new_tokens=256, do_sample=True, temperature=1.0, top_p=1.0,)
            critique_output = tokenizer.batch_decode(critique_out, skip_special_tokens=True)[0]
            critique_response = critique_output[len(critique_prompt) :].lstrip()
            history.extend([critique[index], critique_response])

            revision_prompt = generate_revision(revision[index], history)
            revision_inputs = tokenizer(revision_prompt, return_tensors="pt").to(device)
            revision_out = model.generate(revision_inputs.input_ids, max_new_tokens=256, do_sample=True, temperature=1.0, top_p=1.0,)
            revision_output = tokenizer.batch_decode(revision_out, skip_special_tokens=True)[0]
            print(revision_output)
            revision_response = revision_output[len(revision_prompt) :].lstrip()
            history.clear()
            history.extend([prompt, revision_response])

        counter += 1
        transcript.append([prompt,revision_response])
    return transcript

def print_results(final):
    for pair in final:
        print('Prompt: ', pair[0])
        print('Revised: ', pair[1])
        print('--------')

def main():
    prompts = load_data('red_team_attempts.json')
    prompts = prompts[0:10]
    transcript = run_experiment(prompts, iters)
    print_results(transcript)

if __name__ == "__main__":
    main()
