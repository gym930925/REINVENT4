#!/usr/bin/env python
"""
linkinvent_rnn_run.py - Linkinvent RNN精度测试脚本 (加载真实Prior)

架构: BiLSTM Encoder + LSTM Decoder + Attention
支持设备: cpu, cuda, npu (昇腾)

用法:
  python linkinvent_rnn_run.py --device cpu
  python linkinvent_rnn_run.py --device cuda
  python linkinvent_rnn_run.py --device npu
  python linkinvent_rnn_run.py --device npu --test sample --batch-size 20
"""

import argparse
import os
import sys
import torch

# ========== 内置测试数据 ==========
ETHANE = "CC"
PROPANE = "CCC"
HEXANE = "CCCCCC"
BUTANE = "CCCC"
WARHEAD_PAIR = "*C1CCCCC1|*C1CCCC(ON)C1"
WARHEAD_TRIPLE = "*N(C)C|*Cc1cncc(C#N)c1|*C[C@@H](O)CC(=O)O"

# ========== RNN模型导入 ==========
from reinvent.models import LinkinventAdapter, SampledSequencesDTO
from reinvent.models.linkinvent.link_invent_model import LinkInventModel
from reinvent.models.linkinvent.dataset.dataset import Dataset
import torch.utils.data as tud

# ========== 默认Prior路径 ==========
DEFAULT_PRIOR_PATH = "/home/g00445338/REINVENT4-4.7/prior/linkinvent.prior"

def set_torch_device(device: torch.device):
    """设置torch默认设备"""
    # 注意: PyTorch 2.0+ 支持 torch.set_default_device()
    try:
        torch.set_default_device(device.type)
    except AttributeError:
        # PyTorch < 2.0 旧版本不支持
        pass
    
    # 设置默认tensor类型
    if device.type == "cuda":
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    elif device.type == "npu":
        try:
            torch.set_default_dtype(torch.float32)  # 设置默认数据类型为 float32
            torch.set_default_device("npu")         # 设置默认设备为 NPU
        except AttributeError:
            print("⚠️ NPU tensor类型不支持，使用默认")
    
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
    print(f"\n{'='*50}")
    print("设备信息")
    print(f"{'='*50}")
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
    
    print(f"{'='*50}\n")


# ==============================================================================
# 模型加载
# ==============================================================================

def load_rnn_prior_model(prior_path: str, device: torch.device) -> LinkinventAdapter:
    """加载Linkinvent RNN Prior模型"""
    print(f"\n{'='*50}")
    print("加载Linkinvent RNN Prior模型")
    print(f"{'='*50}")
    print(f"Prior路径: {prior_path}")
    print(f"目标设备: {device}")
    
    if not os.path.exists(prior_path):
        print(f"❌ Prior文件不存在: {prior_path}")
        print("   请使用 --prior 参数指定正确路径")
        sys.exit(1)
    
    print("正在加载...")
    save_dict = torch.load(prior_path, map_location=str(device), weights_only=False)
    
    model = LinkInventModel.create_from_dict(save_dict, "inference", device)
    set_torch_device(device)
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
# 测试函数
# ==============================================================================

def test_likelihood(adapter: LinkinventAdapter) -> dict:
    """测试1: Likelihood计算精度"""
    print(f"\n{'='*50}")
    print("测试1: Likelihood计算精度")
    print(f"{'='*50}")
    
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


def test_sampling(adapter: LinkinventAdapter, warhead: str, batch_size: int = 10) -> dict:
    """测试2: 采样精度测试
    
    输入warhead对，输出linker
    """
    print(f"\n{'='*50}")
    print(f"测试2: 采样精度 (batch_size={batch_size})")
    print(f"{'='*50}")
    
    results = {"passed": False, "details": {}}
    
    try:
        print(f"输入Warhead: {warhead}")
        
        # 创建Dataset
        warhead_list = [warhead] * batch_size
        dataset = Dataset(warhead_list, adapter.vocabulary.input)
        
        dataloader = tud.DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=Dataset.collate_fn
        )
        
        print("正在采样...")
        for batch in dataloader:
            sample_results = adapter.sample(*batch)
        print("\n输出结果:")
        print(f"  输入数量: {len(sample_results.input)}")
        print(f"  输出数量: {len(sample_results.output)}")
        print(f"  NLL数量: {len(sample_results.nlls)}")
        
        # 显示采样结果
        print("\n采样结果 (前5个):")
        for i in range(min(5, len(sample_results.output))):
            print(f"  [{i+1}] Input: {sample_results.input[i]}")
            print(f"      Output: {sample_results.output[i]}")
            print(f"      NLL: {float(sample_results.nlls[i]):.4f}")
        
        # 统计有效输出
        valid_count = sum(1 for o in sample_results.output if o and o != "")
        print(f"\n有效Output数量: {valid_count}/{batch_size}")
        
        # NLL统计
        import numpy as np
        nll_array = np.array([float(n) for n in sample_results.nlls])
        
        print("\nNLL统计:")
        print(f"  平均值: {nll_array.mean():.4f}")
        print(f"  标准差: {nll_array.std():.4f}")
        print(f"  最小值: {nll_array.min():.4f}")
        print(f"  最大值: {nll_array.max():.4f}")
        
        # 验证
        assert len(sample_results.output) == batch_size, f"数量应为{batch_size}"
        
        results["passed"] = True
        results["details"] = {
            "total_samples": len(sample_results.output),
            "valid_samples": valid_count,
            "nll_mean": nll_array.mean(),
            "nll_std": nll_array.std(),
            "nll_min": nll_array.min(),
            "nll_max": nll_array.max(),
            "sample_outputs": sample_results.output[:5]
        }
        
        print("\n✅ 测试通过!")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        results["details"]["error"] = str(e)
    
    return results


def test_model_info(adapter: LinkinventAdapter) -> dict:
    """测试3: 模型信息"""
    print(f"\n{'='*50}")
    print("测试3: RNN模型信息")
    print(f"{'='*50}")
    
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
    print(f"\n{'='*50}")
    print("测试4: 设备一致性验证")
    print(f"{'='*50}")
    
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
        description="Linkinvent RNN精度测试脚本 (加载真实Prior)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python linkinvent_rnn_run.py --device cpu
  python linkinvent_rnn_run.py --device cuda
  python linkinvent_rnn_run.py --device npu
  python linkinvent_rnn_run.py --device npu --prior /path/to/linkinvent.prior
  python linkinvent_rnn_run.py --device npu --test sample --batch-size 20
        """
    )
    
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "npu"])
    parser.add_argument("--prior", default=DEFAULT_PRIOR_PATH)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--warhead", default=WARHEAD_PAIR)
    parser.add_argument("--test", default="all", choices=["likelihood", "sample", "info", "device", "all"])
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("Linkinvent RNN 精度测试 (加载真实Prior)")
    print("="*60)
    print(f"架构: BiLSTM Encoder + LSTM Decoder + Attention")
    print(f"运行设备: {args.device}")
    print(f"Prior路径: {args.prior}")
    print(f"测试类型: {args.test}")
    print("="*60)
    
    device = get_device(args.device)
    print_device_info(device)
    
    adapter = load_rnn_prior_model(args.prior, device)
    
    results = []
    
    if args.test == "all":
        results.append(("Likelihood", test_likelihood(adapter)))
        # results.append(("采样", test_sampling(adapter, args.warhead, args.batch_size)))
        # results.append(("模型信息", test_model_info(adapter)))
        # results.append(("设备一致性", test_device_consistency(adapter, device)))
    elif args.test == "likelihood":
        results.append(("Likelihood", test_likelihood(adapter)))
    elif args.test == "sample":
        results.append(("采样", test_sampling(adapter, args.warhead, args.batch_size)))
    elif args.test == "info":
        results.append(("模型信息", test_model_info(adapter)))
    elif args.test == "device":
        results.append(("设备一致性", test_device_consistency(adapter, device)))
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, r in results if r["passed"])
    for name, result in results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {name}")
    
    print(f"\n总计: {passed}/{len(results)} 通过")
    print("="*60)
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
