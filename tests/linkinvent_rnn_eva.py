#!/usr/bin/env python
"""
linkinvent_rnn_eva.py - Linkinvent RNN采样性能测试脚本 (加载真实Prior)

架构: BiLSTM Encoder + LSTM Decoder + Attention
支持设备: cpu, cuda, npu (昇腾)

用法:
  python linkinvent_rnn_eva.py --device cpu --test sample
  python linkinvent_rnn_eva.py --device cuda --test sample
  python linkinvent_rnn_eva.py --device npu --test sample
  python linkinvent_rnn_eva.py --device npu --test sample --total-samples 10000 --batch-size 200
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.rdchem import EditableMol
import numpy as np
import torch


# ========== 内置测试数据 ==========
ETHANE = "CC"
PROPANE = "CCC"
HEXANE = "CCCCCC"
BUTANE = "CCCC"
WARHEAD_PAIR = "*C1CCCCC1|*C1CCCC(ON)C1"
WARHEAD_TRIPLE = "*N(C)C|*Cc1cncc(C#N)c1|*C[C@@H](O)CC(=O)O"
WARHEAD_QUADRUPLE = "*C[C@@H](O)CC(=O)O|*N(C)C|*c1ccccc1|*Cc1cncc(C#N)c1"

SAMPLE_TEST_CASES = [
    ("*c1ccccc1|*c1ccccc1", "测试例1: 相同warhead"),
    ("N1C(=O)CC(C)(c2cccc(*)c2)CC1(=O)|CC1CCCN(C*)C1", "测试例2: 不同warhead"),
    ("n1cSc(CNC(=O)*)c1C|n1nC(C)c(*)c1C", "测试例3: 含杂原子warhead"),
]

# ========== RNN模型导入 ==========
from reinvent.models import LinkinventAdapter, SampledSequencesDTO
from reinvent.models.linkinvent.link_invent_model import LinkInventModel
from reinvent.models.linkinvent.dataset.dataset import Dataset
import torch.utils.data as tud

# ========== 默认Prior路径 ==========
DEFAULT_PRIOR_PATH = "/home/g00445338/REINVENT4-4.7/prior/linkinvent.prior"


def set_torch_device(device: torch.device):
    """设置torch默认设备"""
    try:
        torch.set_default_device(str(device))
    except AttributeError:
        pass

    if device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    print(f"🔧 已设置默认设备: {device}")


# ==============================================================================
# 设备检测与配置
# ==============================================================================

def get_device(device_str: str) -> torch.device:
    """根据字符串获取torch设备"""
    device_str = device_str.lower()

    if device_str == "cpu":
        return torch.device("cpu")

    elif device_str.startswith("cuda"):
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，将使用CPU")
            return torch.device("cpu")
        if ":" in device_str:
            return torch.device(device_str)
        return torch.device("cuda:0")

    elif device_str.startswith("npu"):
        try:
            if hasattr(torch, 'npu') and torch.npu.is_available():
                print("✅ 检测到昇腾NPU设备")
                if ":" in device_str:
                    return torch.device(device_str)
                return torch.device("npu:0")
            else:
                print("❌ NPU不可用，将使用CPU")
                return torch.device("cpu")
        except Exception as e:
            print(f"❌ NPU检测失败: {e}, 将使用CPU")
            return torch.device("cpu")

    else:
        print(f"⚠️ 未知设备 '{device_str}', 使用CPU")
        return torch.device("cpu")


def print_device_info(device: torch.device):
    """打印设备详细信息"""
    print(f"\n{'=' * 50}")
    print("设备信息")
    print(f"{'=' * 50}")
    print(f"设备类型: {device.type}")
    print(f"设备索引: {device.index if device.index else 0}")

    if device.type == "cuda":
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"GPU名称: {torch.cuda.get_device_name(device.index or 0)}")

    elif device.type == "npu":
        try:
            print(f"NPU数量: {torch.npu.device_count()}")
            for i in range(torch.npu.device_count()):
                print(f"NPU:{i} 名称: {torch.npu.get_device_name(i)}")
        except Exception as e:
            print(f"⚠️ NPU信息获取失败: {e}")

    print(f"{'=' * 50}\n")


# ==============================================================================
# 模型加载
# ==============================================================================

def load_rnn_prior_model(prior_path: str, device: torch.device) -> LinkinventAdapter:
    """加载Linkinvent RNN Prior模型"""
    print(f"\n{'=' * 50}")
    print("加载Linkinvent RNN Prior模型")
    print(f"{'=' * 50}")
    print(f"Prior路径: {prior_path}")
    print(f"目标设备: {device}")

    if not os.path.exists(prior_path):
        print(f"❌ Prior文件不存在: {prior_path}")
        print("   请使用 --prior 参数指定正确路径")
        sys.exit(1)

    print("正在加载...")
    print("  [1/3] 加载checkpoint到CPU...")
    save_dict = torch.load(prior_path, map_location="cpu", weights_only=False)

    print("  [2/3] 创建模型...")
    model = LinkInventModel.create_from_dict(save_dict, "inference", device)

    print("  [3/3] 设置设备...")
    set_torch_device(device)

    print("  [3/3] 移动模型到目标设备...")
    model.network.to(device)
    model.device = device

    adapter = LinkinventAdapter(model)

    print("✅ RNN模型加载成功")
    print(f"   模型类型: {model._model_type}")
    print(f"   架构: BiLSTM Encoder + LSTM Decoder + Attention")
    print(f"   词表大小: {adapter.vocabulary.len()}")
    print(f"   最大序列长度: {model.max_sequence_length}")

    param_device = next(model.network.parameters()).device
    print(f"   网络参数设备: {param_device}")

    if param_device != device:
        print(f"⚠️ 设备不一致，尝试强制移动...")
        model.network.to(device)
        model.device = device

    return adapter


# ==============================================================================
# Linker 拼接函数 (来自 merge_link.py)
# ==============================================================================

def link_fragments(frag1_smi, frag2_smi, linker_smi):
    try:
        f1 = Chem.MolFromSmiles(frag1_smi)
        f2 = Chem.MolFromSmiles(frag2_smi)
        linker = Chem.MolFromSmiles(linker_smi)

        if not f1 or not f2 or not linker:
            return "INVALID_INPUT_SMILES"

        for atom in f1.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(1)

        for atom in f2.GetAtoms():
            if atom.GetAtomicNum() == 0:
                atom.SetAtomMapNum(2)

        linker_wildcards = [atom for atom in linker.GetAtoms() if atom.GetAtomicNum() == 0]

        if len(linker_wildcards) != 2:
            return f"INVALID_LINKER(wildcards={len(linker_wildcards)}, expected 2)"

        linker_wildcards[0].SetAtomMapNum(1)
        linker_wildcards[1].SetAtomMapNum(2)

        combined = Chem.CombineMols(f1, f2)
        combined = Chem.CombineMols(combined, linker)

        welded = weld_r_groups(combined)

        if welded is None:
            return "FAILED_TO_LINK(welding_failed)"

        Chem.SanitizeMol(welded)
        return Chem.MolToSmiles(welded)

    except Exception as e:
        return f"FAILED_TO_LINK_({str(e)})"


def weld_r_groups(input_mol):
    try:
        join_dict = defaultdict(list)
        for atom in input_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                join_dict[map_num].append(atom)

        for idx, atom_list in join_dict.items():
            if len(atom_list) == 2:
                atm_1, atm_2 = atom_list
                neighbors_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()]
                neighbors_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()]

                if neighbors_1 and neighbors_2:
                    nbr_1 = neighbors_1[0]
                    nbr_2 = neighbors_2[0]
                    nbr_1.SetAtomMapNum(idx)
                    nbr_2.SetAtomMapNum(idx)
            elif len(atom_list) != 2:
                print(f"Warning: AtomMapNum {idx} has {len(atom_list)} atoms instead of 2")
                return None

        new_mol = Chem.DeleteSubstructs(input_mol, Chem.MolFromSmarts('[#0]'))

        bond_join_dict = defaultdict(list)
        for atom in new_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                bond_join_dict[map_num].append(atom.GetIdx())

        em = EditableMol(new_mol)
        for idx, atom_indices in bond_join_dict.items():
            if len(atom_indices) == 2:
                start_atm, end_atm = atom_indices
                em.AddBond(start_atm, end_atm, order=Chem.rdchem.BondType.SINGLE)

        final_mol = em.GetMol()

        for atom in final_mol.GetAtoms():
            atom.SetAtomMapNum(0)

        final_mol = Chem.RemoveHs(final_mol)

        return final_mol

    except Exception as e:
        print(f"Welding error: {e}")
        return None


# ==============================================================================
# 测试函数
# ==============================================================================

def test_likelihood(adapter: LinkinventAdapter) -> dict:
    """测试1: Likelihood计算精度"""
    print(f"\n{'=' * 50}")
    print("测试1: Likelihood计算精度")
    print(f"{'=' * 50}")

    results = {"passed": False, "details": {}}

    try:
        dto1 = SampledSequencesDTO(ETHANE, PROPANE, 0.9)
        dto2 = SampledSequencesDTO(HEXANE, BUTANE, 0.1)
        sampled_list = [dto1, dto2]

        print("输入SMILES配对:")
        print(f"  配对1: {ETHANE} → {PROPANE}")
        print(f"  配对2: {HEXANE} → {BUTANE}")

        output = adapter.likelihood_smiles(sampled_list)
        likelihood = output.likelihood

        print("\n输出结果:")
        print(f"  Likelihood shape: {likelihood.shape}")
        print(f"  Likelihood值: {likelihood.tolist()}")
        print(f"  NLL值: {(-likelihood).tolist()}")

        assert likelihood.shape == torch.Size([2]), f"shape应为[2], 实际为{likelihood.shape}"

        results["passed"] = True
        results["details"] = {
            "shape": list(likelihood.shape),
            "likelihood": likelihood.tolist(),
            "nll": (-likelihood).tolist()
        }

        print("\n✅ 测试通过!")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["details"]["error"] = str(e)

    return results


def test_sampling(adapter: LinkinventAdapter, total_samples: int = 10000,
                   batch_size: int = 200, output_dir: str = ".", device_name: str = "cpu") -> dict:
    """测试2: 采样性能测试

    对三个测试用例各进行 total_samples 次采样，记录时间和结果，并将结果写入文件
    """
    print(f"\n{'=' * 50}")
    print(f"测试2: 采样性能测试 (每组{total_samples}次)")
    print(f"{'=' * 50}")

    os.makedirs(output_dir, exist_ok=True)

    results = {"passed": True, "cases": {}}

    for idx, (warhead, case_name) in enumerate(SAMPLE_TEST_CASES, 1):
        print(f"\n--- {case_name} ---")
        print(f"  Warhead: {warhead}")

        num_batches = total_samples // batch_size
        all_input_smiles = []
        all_output_smiles = []
        all_nlls = []

        start_time = time.time()
        for _ in range(num_batches):
            warhead_list = [warhead] * batch_size
            dataset = Dataset(warhead_list, adapter.vocabulary.input)
            dataloader = tud.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, collate_fn=Dataset.collate_fn
            )
            for batch in dataloader:
                sample_results = adapter.sample(*batch)
                all_input_smiles.extend(sample_results.input)
                all_output_smiles.extend(sample_results.output)
                all_nlls.extend([float(n) for n in sample_results.nlls])

        elapsed = time.time() - start_time

        valid_count = sum(1 for o in all_output_smiles if o and o != "")
        nll_array = np.array(all_nlls)

        print(f"  总耗时: {elapsed:.2f}s")
        print(f"  采样速度: {total_samples / elapsed:.1f} samples/s")
        print(f"  实际输出: {len(all_output_smiles)} (目标: {total_samples})")
        print(f"  有效Linker: {valid_count}")
        print(f"  NLL 均值: {nll_array.mean():.4f}")
        print(f"  NLL 标准差: {nll_array.std():.4f}")
        print(f"  NLL 最小值: {nll_array.min():.4f}")
        print(f"  NLL 最大值: {nll_array.max():.4f}")

        frag1, frag2 = warhead.split("|")

        all_full_smiles = []
        merge_success_count = 0
        for linker in all_output_smiles:
            if not linker:
                all_full_smiles.append("")
                continue
            if "*" in linker and "[*]" not in linker:
                linker = linker.replace("*", "[*]")
            full_smi = link_fragments(frag1, frag2, linker)
            all_full_smiles.append(full_smi)
            if not full_smi.startswith("FAILED") and not full_smi.startswith("INVALID"):
                merge_success_count += 1

        print(f"  合并成功: {merge_success_count}/{len(all_output_smiles)}")

        case_file = os.path.join(output_dir, f"rnn_sample_case{idx}_{device_name}.csv")
        with open(case_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["input_smiles", "linker", "output_smiles"])
            for input_smi, linker, full_smi in zip(all_input_smiles, all_output_smiles, all_full_smiles):
                writer.writerow([input_smi, linker, full_smi])

        print(f"  结果已写入: {case_file}")

        results["cases"][case_name] = {
            "warhead": warhead,
            "elapsed": elapsed,
            "speed": total_samples / elapsed,
            "valid_count": valid_count,
            "merge_success": merge_success_count,
            "nll_mean": float(nll_array.mean()),
            "nll_std": float(nll_array.std()),
            "nll_min": float(nll_array.min()),
            "nll_max": float(nll_array.max()),
            "output_file": case_file,
        }

    summary_file = os.path.join(output_dir, f"rnn_sample_summary_{device_name}.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Device: {device_name}\n")
        for case_name, case_data in results["cases"].items():
            f.write(f"{case_name}: {case_data['elapsed']:.2f}s\n")

    print(f"\n{'=' * 50}")
    print("采样性能汇总")
    print(f"{'=' * 50}")
    print(f"{'测试例':<20} {'耗时(s)':<10} {'速度(s/s)':<12} {'有效数':<8} {'NLL均值':<10}")
    print("-" * 60)
    for case_name, case_data in results["cases"].items():
        print(
            f"{case_name:<20} {case_data['elapsed']:<10.2f} {case_data['speed']:<12.1f} "
            f"{case_data['valid_count']:<8} {case_data['nll_mean']:<10.4f}"
        )
    print(f"\n  汇总文件: {summary_file}")
    print(f"\n✅ 测试通过!")

    return results


def test_model_info(adapter: LinkinventAdapter) -> dict:
    """测试3: RNN模型信息"""
    print(f"\n{'=' * 50}")
    print("测试3: RNN模型信息")
    print(f"{'=' * 50}")

    model = adapter.model

    print(f"模型类型: {model._model_type}")
    print(f"模型版本: {model._version}")
    print(f"架构: BiLSTM Encoder + LSTM Decoder + Attention")
    print(f"设备: {model.device}")
    print(f"最大序列长度: {model.max_sequence_length}")

    total_params = sum(p.numel() for p in model.network.parameters())
    print("\n参数统计:")
    print(f"  总参数: {total_params}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    print("\n网络结构摘要:")
    print(model.network)

    return {"passed": True, "details": {"total_params": total_params}}


def test_device_consistency(adapter: LinkinventAdapter, expected_device: torch.device) -> dict:
    """测试4: 设备一致性"""
    print(f"\n{'=' * 50}")
    print("测试4: 设备一致性验证")
    print(f"{'=' * 50}")

    results = {"passed": False, "details": {}}

    try:
        model = adapter.model

        inconsistent = []
        for name, param in model.network.named_parameters():
            if param.device != expected_device:
                inconsistent.append((name, str(param.device)))

        for name, buffer in model.network.named_buffers():
            if buffer.device != expected_device:
                inconsistent.append((name, str(buffer.device)))

        total_params = sum(1 for _ in model.network.named_parameters())
        print(f"检查了 {total_params} 个参数")

        if inconsistent:
            print(f"\n❌ 设备不一致:")
            for name, dev in inconsistent[:5]:
                print(f"  {name}: {dev} != {expected_device}")
            results["passed"] = False
        else:
            print(f"\n✅ 所有参数在: {expected_device}")
            results["passed"] = True

    except Exception as e:
        print(f"❌ 验证失败: {e}")
        results["details"]["error"] = str(e)

    return results


# ==============================================================================
# 主函数
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Linkinvent RNN采样性能测试脚本 (加载真实Prior)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python linkinvent_rnn_eva.py --device cpu --test sample
  python linkinvent_rnn_eva.py --device npu --test sample --output-dir ./results
  python linkinvent_rnn_eva.py --device npu --test sample --total-samples 10000 --batch-size 200
        """
    )
    parser.add_argument("--device", default="cpu", help="运行设备: cpu, cuda, cuda:0, npu, npu:0, npu:1")
    parser.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--total-samples", type=int, default=10000)
    parser.add_argument("--output-dir", default=".", help="结果输出目录")
    parser.add_argument("--test", default="all", choices=["likelihood", "sample", "info", "device", "all"])

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Linkinvent RNN 采样性能测试")
    print("=" * 60)
    print(f"架构: BiLSTM Encoder + LSTM Decoder + Attention")
    print(f"运行设备: {args.device}")
    print(f"Prior路径: {args.prior}")
    print(f"测试类型: {args.test}")
    print(f"每组采样数: {args.total_samples}")
    print(f"Batch大小: {args.batch_size}")
    print("=" * 60)

    device = get_device(args.device)
    print_device_info(device)

    adapter = load_rnn_prior_model(args.prior, device)

    results = []

    if args.test == "all":
        results.append(("采样", test_sampling(adapter, args.total_samples, args.batch_size,
                                               args.output_dir, args.device)))
        # results.append(("模型信息", test_model_info(adapter)))
        # results.append(("设备一致性", test_device_consistency(adapter, device)))
    elif args.test == "sample":
        results.append(("采样", test_sampling(adapter, args.total_samples, args.batch_size,
                                               args.output_dir, args.device)))
    elif args.test == "info":
        results.append(("模型信息", test_model_info(adapter)))
    elif args.test == "device":
        results.append(("设备一致性", test_device_consistency(adapter, device)))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, r in results if r["passed"])
    for name, result in results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {name}")

    print(f"\n总计: {passed}/{len(results)} 通过")
    print("=" * 60)

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
