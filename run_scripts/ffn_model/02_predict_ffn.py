from pathlib import Path
import subprocess
import argparse


python_file = "src/ms_pred/ffn_pred/predict.py"
devices = ",".join(["1"])

run_models = [
    {"dataset": "nist20", "folder": "scaffold_1", "split": "scaffold_1"},
    {"dataset": "nist20", "folder": "split_1_rnd1", "split": "split_1"},
    {"dataset": "nist20", "folder": "split_1_rnd2", "split": "split_1"},
    {"dataset": "nist20", "folder": "split_1_rnd3", "split": "split_1"},

    {"dataset": "canopus_train_public", "folder": "split_1_rnd1", "split": "split_1"},
    {"dataset": "canopus_train_public", "folder": "split_1_rnd2", "split": "split_1"},
    {"dataset": "canopus_train_public", "folder": "split_1_rnd3", "split": "split_1"},
]



for run_model in run_models:
    dataset = run_model['dataset']
    folder = run_model['folder']
    split = run_model['split']

    res_folder = Path(f"results/ffn_baseline_{dataset}/{folder}")
    for model in res_folder.rglob("version_0/*.ckpt"):
        save_dir = model.parent.parent

        save_dir = save_dir / "preds"
        save_dir.mkdir(exist_ok=True)
        cmd = f"""python {python_file} \\
        --batch-size 32 \\
        --dataset-name {dataset} \\
        --split-name {split}.tsv \\
        --subset-datasets test_only  \\
        --checkpoint {model} \\
        --save-dir {save_dir} \\
        --gpu"""
        device_str = f"CUDA_VISIBLE_DEVICES={devices}"
        cmd = f"{device_str} {cmd}"
        print(cmd + "\n")
        subprocess.run(cmd, shell=True)

        out_binned = save_dir / "fp_preds.p"
        eval_cmd = f"""
        python analysis/spec_pred_eval.py \\
        --binned-pred-file {out_binned} \\
        --max-peaks 100 \\
        --min-inten 0 \\
        --formula-dir-name no_subform \\
        --dataset {dataset}
        """
        print(eval_cmd)
        subprocess.run(eval_cmd, shell=True)
