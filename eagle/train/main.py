import argparse

parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--basepath', type=str, default='/home/lyh/weights/hf/vicuna_v13/7B/')
parser.add_argument('--configpath', type=str, default="config.json")
parser.add_argument('--run_name', type=str, default="original_eagle_v1_one_layer")
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--num_hidden_layers', type=int, default=1)
parser.add_argument('--expansion_factor', type=int, default=1)
parser.add_argument('--add_next_token_loss', type=str, default="no")
parser.add_argument('--save_to_hf', type=str, default="no")
parser.add_argument('--include_top_k_loss', type=str, default="no")
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--train_lm_head_em_table', type=str, default="no")
parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
parser.add_argument('--tmpdir', type=str, default='0')
parser.add_argument('--cpdir', type=str, default='0')
args = parser.parse_args()

train_config = {
    "lr": args.lr,
    "bs": args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "datapath": f"{args.tmpdir}",
    "is_warmup": True,
    "num_epochs": 20,
    # Depending on your data and model size, the larger the model, the higher the sample efficiency. We recommend setting it between 20-40.
    "num_warmup_steps": 2000,
    "total_steps": 800000,
    "p_w": 0.1,
    "v_w": 1.0,
    "next_token_w": 0.1,
    "head_w": 0.1,
    "num_workers": 2,
    "embeding": True,
    "act": "No",
    "data_noise": True,
    "noise": "uniform",
    "mean": 0.0,
    "std": 0.2,
    "residual": "true,norm",
    "max_len": 2048,
    # During training, truncating the training sequences means that the larger the setting, the more training data is used, and the better the effect, but it also consumes more VRAM.
    "config_path": args.configpath,
    "b1": 0.9,
    "b2": 0.95,
    "grad_clip": 0.5,
    "save_freq": 5
}
import json
from safetensors import safe_open
# from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate import Accelerator
from accelerate.utils import set_seed

set_seed(0)
accelerator = Accelerator(mixed_precision='bf16',
                          gradient_accumulation_steps=train_config["gradient_accumulation_steps"])
from ..model.cnets import Model
from ..model.configs import EConfig
from typing import Any, Dict, List

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# import accelerate
import numpy as np
from transformers import get_linear_schedule_with_warmup, AutoConfig

if accelerator.is_main_process:
    import wandb

    wandb.init(project="eagle_mamba_offline_research_h100",
               group="Eagle_for_Llama-3.1-8B-Instruct",
               name=args.run_name,
               config=train_config)

baseconfig = AutoConfig.from_pretrained(args.basepath)

teacher_head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
student_head = torch.nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)

try:
    with open(os.path.join(args.basepath, "model.safetensors.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    with safe_open(os.path.join(args.basepath, head_path),
                   framework="pt",
                   device="cpu") as f:
        tensor_slice = f.get_slice("lm_head.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
except Exception as e:
    print(e)
    with open(os.path.join(args.basepath, "pytorch_model.bin.index.json"), "r") as f:
        index_json = json.loads(f.read())
        head_path = index_json["weight_map"]["lm_head.weight"]
    weights = torch.load(os.path.join(args.basepath, head_path))
    tensor = weights["lm_head.weight"].float()

student_head.weight.data = tensor
student_head.eval()

teacher_head.weight.data = tensor
teacher_head.eval()

for param in teacher_head.parameters():
    param.requires_grad = False

# if args.train_lm_head_em_table == "yes":
#    for param in student_head.parameters():
#        param.requires_grad = True
# else:

for param in student_head.parameters():
    param.requires_grad = False

def print_model_size(model: torch.nn.Module, model_path: str) -> None:
    """Print model name, the number of trainable parameters and initialization
    time.

    Args:
        model: The PyTorch model.
        model_path: name or path for model.
    """
    print(f"--> Model {model_path}")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--> {model_path} has {total_trainable_params / 1e6} Million params to train.")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"--> {model_path} has {total_params / 1e6} Million params in total.")


def list_files(path):
    datapath = []
    for root, directories, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            datapath.append(file_path)
    return datapath


class AddGaussianNoise:
    def __init__(self, mean=0.0, std=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        noisy_tensor = tensor + noise
        data["hidden_state_big"] = noisy_tensor
        return data


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # try:
        data = torch.load(self.data[index])
        new_data = {}
        hidden_state = data['hidden_state'][:train_config["max_len"]][None, :]
        input_ids = data['input_ids'][:train_config["max_len"]][None, :]
        loss_mask = data["loss_mask"][:train_config["max_len"]][None, :]


        length = hidden_state.shape[1]
        # length_q = data['query_ids'].shape[1]
        attention_mask = [1] * length
        loss_mask = loss_mask[0].tolist()
        loss_mask[-1] = 0

        input_ids_target = input_ids[:, 1:]
        zeropadding = torch.tensor([[0]])
        input_ids_target = torch.cat((input_ids_target, zeropadding), dim=1)

        target = hidden_state[:, 1:, :]
        zeropadding = torch.zeros(1, 1, target.shape[2])
        target = torch.cat((target, zeropadding), dim=1)
        loss_mask[-1] = 0
        new_data["attention_mask"] = attention_mask
        new_data["loss_mask"] = loss_mask
        new_data["target"] = target
        new_data["hidden_state_big"] = hidden_state
        new_data["input_ids"] = input_ids_target


        if self.transform:
            new_data = self.transform(new_data)

        return new_data


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item['hidden_state_big'].shape[1] for item in features)
        batch_input_ids = torch.cat([self.paddingtensor2D(item['input_ids'], max_length) for item in features])
        batch_hidden_states = torch.cat([self.paddingtensor(item['hidden_state_big'], max_length) for item in features])
        batch_target = torch.cat([self.paddingtensor(item['target'], max_length) for item in features])
        batch_loss_mask = torch.tensor(
            [item['loss_mask'] + [0] * (max_length - len(item['loss_mask'])) for item in features])
        batch_attention_mask = torch.tensor(
            [item['attention_mask'] + [0] * (max_length - len(item['attention_mask'])) for item in features])
        # batch_loss_mask = torch.ones_like(batch_loss_mask)
        # batch_attention_mask=torch.ones_like(batch_attention_mask)
        batch = {
            "input_ids": batch_input_ids,
            "hidden_states": batch_hidden_states,
            "target": batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res

def log_of_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_func: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Compute the actual log of labels given pre-computed logits.

    This function is also useful for both Roberta model and getting
    generation logits for sampling methods.
    """
    log_p = -loss_func(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    batch_size, sequence_length, vocab_size = logits.size()

    # compute per-token log probability in a sequence.
    log_p = log_p.view(batch_size, sequence_length)

    # non-masked tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill(labels == -100, 0.0)

    # good_log_p now has the log probability of the output
    # sequence tokens corresponding to the labels at the [MASK] location.
    return torch.sum(good_log_p, dim=1)

next_token_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

def compute_loss(target, target_p, predict, loss_mask, input_ids, target_head):
    out_head = student_head(predict)
    next_token_loss = torch.tensor([0.0])
    if args.add_next_token_loss == "yes":
        target_ids = target_head.argmax(dim=2)
        labels = target_ids.masked_fill(loss_mask.squeeze(2) == 0, -100)
        logits = out_head
        next_token_loss = -torch.mean(log_of_labels(logits, labels, next_token_func))
        # labels = input_ids.masked_fill(loss_mask.squeeze(2) == 0, -100)
        # Shift so that tokens < n predict n
        # shift_logits = out_head[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # next_token_loss = -torch.mean(log_of_labels(shift_logits, shift_labels, next_token_func))

    out_logp = nn.LogSoftmax(dim=2)(out_head)
    if args.include_top_k_loss == "yes":
        _, topk_target_indices = torch.topk(target_p, k=args.topk, dim=2, largest=True, sorted=True)
        topk_target_p = torch.gather(target_p, dim=2, index=topk_target_indices)
        topk_out_logp = torch.gather(out_logp, dim=2, index=topk_target_indices)
        plogp = topk_target_p * topk_out_logp
    else:
        plogp = target_p * out_logp
    ploss = -torch.sum(torch.sum(loss_mask * plogp, 2)) / (loss_mask.sum() + 1e-5)
    vloss = criterion(predict, target)
    vloss = torch.sum(torch.mean(loss_mask * vloss, 2)) / (loss_mask.sum() + 1e-5)
    return vloss, ploss, out_head, next_token_loss

@torch.no_grad()
def getkacc(model, data, max_length=5):
    def generate(hidden_states, input_ids, max_length=4, use_cache=True):
        if use_cache:
            past_key_values = None
            for i in range(max_length):
                if past_key_values != None:
                    out_hidden, past_key_values = model(last_hidden, input_ids=token, past_key_values=past_key_values,
                                                        use_cache=True)
                else:
                    out_hidden, past_key_values = model(hidden_states, input_ids=input_ids, use_cache=True)
                last_hidden = out_hidden[:, -1:]
                last_headout = student_head(last_hidden)
                token = torch.argmax(last_headout, dim=-1)
                input_ids = torch.cat((input_ids, token), dim=1)

        else:
            raise NotImplementedError

        return input_ids

    hidden_states = data["hidden_states"]
    input_ids = data["input_ids"]
    loss_mask = data["loss_mask"]
    target = data["target"]
    total = [0 for _ in range(max_length)]
    correct = [0 for _ in range(max_length)]
    bs, seq_len = hidden_states.shape[0], hidden_states.shape[1]
    target_headout = teacher_head(target)
    target_ids = target_headout.argmax(dim=2)

    for pre_len in range(1, seq_len):
        if loss_mask[:, pre_len].sum() == 0:
            continue
        pre_hidden_states = hidden_states[:, :pre_len]
        pre_input_ids = input_ids[:, :pre_len]
        outs = generate(pre_hidden_states, pre_input_ids, max_length=max_length)
        generate_ids = outs[:, pre_len:]
        for bid in range(bs):
            for k in range(max_length):
                if loss_mask[bid, pre_len + k] == 0:
                    break
                if pre_len + k >= seq_len:
                    break
                total[k] += 1
                if generate_ids[bid, k] == target_ids[bid, pre_len + k - 1]:
                    correct[k] += 1
                else:
                    for kk in range(k + 1, max_length):
                        total[kk] += 1
                    break

    acc = [correct[i] / total[i] for i in range(len(correct))]
    return acc


if train_config["data_noise"]:
    if train_config["noise"] == "uniform":
        aug = AddUniformNoise(std=train_config["std"])
    else:
        aug = AddGaussianNoise(mean=train_config["mean"], std=train_config["std"])
else:
    aug = None

datapath = list_files(train_config["datapath"])

traindatapath = datapath[:int(len(datapath) * 0.95)]
testdatapath = datapath[int(len(datapath) * 0.95):]

traindataset = CustomDataset(traindatapath, transform=aug)
testdataset = CustomDataset(testdatapath)
train_loader = DataLoader(traindataset, batch_size=train_config["bs"], shuffle=True,
                          collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"],
                          pin_memory=True)
test_loader = DataLoader(testdataset, batch_size=train_config["bs"], shuffle=False,
                         collate_fn=DataCollatorWithPadding(), num_workers=train_config["num_workers"], pin_memory=True)

if accelerator.is_main_process:
    if not os.path.exists(args.cpdir):
        os.makedirs(args.cpdir)

config = EConfig.from_pretrained(train_config["config_path"])

config.update({"num_hidden_layers": args.num_hidden_layers})
config.update({"expansion_factor": args.expansion_factor})

if args.train_lm_head_em_table == "yes":
    config.update({"train_em_table": True})
else:
    config.update({"train_em_table": False})

print("Eagle config:\n")
print(config)
print("\n")

model = Model(config, load_emb=True, path=args.basepath)

print_model_size(model, args.basepath)

criterion = nn.SmoothL1Loss(reduction="none")
parameters = list(model.parameters())
# if args.train_lm_head_em_table == "yes":
#    parameters += list(student_head.parameters())

optimizer = optim.AdamW(parameters, lr=train_config["lr"], betas=(train_config["b1"], train_config["b2"]))

num_epochs = train_config["num_epochs"]
num_warmup_steps = train_config["num_warmup_steps"]
total_steps = train_config["total_steps"]
is_warmup = train_config["is_warmup"]

if is_warmup:
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)

    model, teacher_head, student_head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
        model, teacher_head, student_head, optimizer, train_loader, test_loader, scheduler
    )

    # model, teacher_head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    #    model, teacher_head, optimizer, train_loader, test_loader, scheduler
    # )
else:
    model, teacher_head, student_head, optimizer, train_loader, test_loader = accelerator.prepare(
        model, teacher_head, student_head, optimizer, train_loader, test_loader
    )

if args.save_to_hf == "no":
    for epoch in range(num_epochs + 1):
        top_3acc = [0 for _ in range(3)]
        correct = 0
        total = 0
        epoch_loss = 0
        num_batches = 0
        model.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                predict = model(data["hidden_states"], input_ids=data["input_ids"], attention_mask=data["attention_mask"])
                with torch.no_grad():
                    target_head = teacher_head(data["target"])
                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                loss_mask = data["loss_mask"][:, :, None]
                vloss, ploss, out_head, next_token_loss = compute_loss(data["target"], target_p, predict, loss_mask, data["input_ids"], target_head)
                loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                # if args.add_next_token_loss == "yes":
                    # loss += train_config["next_token_w"] * next_token_loss
                # loss.backward()
                accelerator.backward(loss)
                accelerator.clip_grad_value_(parameters, train_config["grad_clip"])
                optimizer.step()
                if is_warmup:
                    scheduler.step()

            with torch.no_grad():
                _, predicted = torch.max(out_head, 2)
                _, target = torch.max(target_head, 2)
                ct = loss_mask.sum().item()
                cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(out_head, target, (1, 2, 3))
                for top_i in range(len(topkacc)):
                    top_3acc[top_i] += topkacc[top_i]
                total += ct
                correct += cc
            if accelerator.is_main_process and ct != 0:
                logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"], "train/vloss": vloss.item(),
                        "train/ploss": ploss.item(), "train/loss": loss.item(), "train/acc": cc / ct,
                        "train/next_token_loss": next_token_loss.item()}
                for id, i in enumerate(top_3acc):
                    logdict[f'train/top_{id + 1}_acc'] = topkacc[id].item() / ct
                wandb.log(logdict)
                # for id,i in enumerate(top_3acc):
                #     wandb.log({f'train/top_{id+1}_acc':topkacc[id].item()/ct})

            del ploss, vloss, next_token_loss
            epoch_loss += loss.item()
            num_batches += 1

        correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
        correct, total = accelerator.gather_for_metrics((correct, total))
        correct, total = correct.sum().item(), total.sum().item()
        epoch_loss /= num_batches
        top_3acc = accelerator.gather_for_metrics(top_3acc)
        if accelerator.is_local_main_process:
            for id, i in enumerate(top_3acc):
                wandb.log({f'train/epochtop_{id + 1}_acc': i.sum().item() / total})
        if accelerator.is_local_main_process:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
            print('Train Accuracy: {:.2f}%'.format(100 * correct / total))
            wandb.log({"train/epochacc": correct / total, "train/epochloss": epoch_loss})

        if (epoch + 1) % train_config["save_freq"]:
            top_3acc = [0 for _ in range(3)]
            correct = 0
            total = 0
            epoch_loss = 0
            num_batches = 0
            model.eval()

            k_acc = [[] for i in range(5)]
            for batch_idx, data in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    if batch_idx < 10:
                        acces = getkacc(model, data, max_length=5)
                        for i in range(len(acces)):
                            k_acc[i].append(acces[i])
                    predict = model(data["hidden_states"], input_ids=data["input_ids"],
                                    attention_mask=data["attention_mask"])
                    target_head = teacher_head(data["target"])
                    target_p = nn.Softmax(dim=2)(target_head)
                    target_p = target_p.detach()
                    loss_mask = data["loss_mask"][:, :, None]
                    vloss, ploss, out_head, next_token_loss = compute_loss(data["target"], target_p, predict, loss_mask, data["input_ids"], target_head)
                    loss = train_config["v_w"] * vloss + train_config["p_w"] * ploss
                    # if args.add_next_token_loss == "yes":
                        # loss += train_config["next_token_w"] * next_token_loss
                    _, predicted = torch.max(out_head, 2)
                    _, target = torch.max(target_head, 2)
                    ct = loss_mask.sum().item()
                    cc = ((predicted == target) * loss_mask.squeeze()).sum().item()
                    out_head = out_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                    target = target.view(-1)[loss_mask.view(-1) == 1]
                    topkacc = top_accuracy(out_head, target, (1, 2, 3))
                    for top_i in range(len(topkacc)):
                        top_3acc[top_i] += topkacc[top_i]
                    total += ct
                    correct += cc
                epoch_loss += loss.item()
                num_batches += 1

            mean_acces = []
            for id, i in enumerate(k_acc):
                mean_acc = np.array(i).mean()
                mean_acc = torch.tensor(mean_acc).cuda()
                mean_acces.append(mean_acc)

            mean_acces = accelerator.gather_for_metrics(mean_acces)
            if accelerator.is_local_main_process:
                for id, i in enumerate(mean_acces):
                    mean_acc = i.mean().item()
                    wandb.log({f"test/{id}_acc": mean_acc})

            correct, total = torch.tensor(correct).cuda(), torch.tensor(total).cuda()
            correct, total = accelerator.gather_for_metrics((correct, total))
            correct, total = correct.sum().item(), total.sum().item()
            top_3acc = accelerator.gather_for_metrics(top_3acc)
            if accelerator.is_local_main_process:
                for id, i in enumerate(top_3acc):
                    wandb.log({f'test/top_{id + 1}_acc': i.sum().item() / total})
            epoch_loss /= num_batches
            if accelerator.is_local_main_process:
                print('Test Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss))
                print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
                wandb.log({"test/epochacc": correct / total, "test/epochloss": epoch_loss})
                accelerator.save_state(output_dir=f"{args.cpdir}/state_{epoch}")
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), f"{args.cpdir}/final_eagle_model.bin")
else:
    accelerator.load_state(args.save_to_hf)
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), args.save_to_hf + "/final_eagle_model.bin")