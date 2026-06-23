import os
from rdkit import Chem
from rdkit.Chem.rdchem import EditableMol
from collections import defaultdict

def link_fragments(frag1_smi, frag2_smi, linker_smi):
    """
    使用 RDKit 分子编辑功能将两个片段和一个连接器（Linker）拼接成完整的 SMILES
    采用"welding"技术连接匹配的通配符原子
    """
    try:
        # 1. 解析 SMILES
        f1 = Chem.MolFromSmiles(frag1_smi)
        f2 = Chem.MolFromSmiles(frag2_smi)
        linker = Chem.MolFromSmiles(linker_smi)
        
        if not f1 or not f2 or not linker:
            return "INVALID_INPUT_SMILES"
        
        # 2. 为片段的通配符分配 AtomMapNum
        # Fragment 1 的通配符标记为 1
        # Fragment 2 的通配符标记为 2
        for atom in f1.GetAtoms():
            if atom.GetAtomicNum() == 0:  # 通配符原子
                atom.SetAtomMapNum(1)
        
        for atom in f2.GetAtoms():
            if atom.GetAtomicNum() == 0:  # 通配符原子
                atom.SetAtomMapNum(2)
        
        # 3. 为 Linker 的通配符分配 AtomMapNum
        # Linker 应该有两个通配符，分别连接两个片段
        linker_wildcards = [atom for atom in linker.GetAtoms() if atom.GetAtomicNum() == 0]
        
        if len(linker_wildcards) != 2:
            return f"INVALID_LINKER(wildcards={len(linker_wildcards)}, expected 2)"
        
        # 将 Linker 的两个通配符分别标记为 1 和 2
        linker_wildcards[0].SetAtomMapNum(1)
        linker_wildcards[1].SetAtomMapNum(2)
        
        # 4. 合并所有分子
        combined = Chem.CombineMols(f1, f2)
        combined = Chem.CombineMols(combined, linker)
        
        # 5. 使用 welding 技术连接匹配的通配符
        welded = weld_r_groups(combined)
        
        if welded is None:
            return "FAILED_TO_LINK(welding_failed)"
        
        # 6. 清理并返回 SMILES
        Chem.SanitizeMol(welded)
        return Chem.MolToSmiles(welded)
        
    except Exception as e:
        return f"FAILED_TO_LINK_({str(e)})"


def weld_r_groups(input_mol):
    """
    将带有 AtomMapNum 标记的通配符原子连接起来
    参考: https://sourceforge.net/p/rdkit/mailman/rdkit-discuss/thread/D59F199B-E4D5-4C42-B00B-6727F29CCA04@dalkescientific.com/
    """
    try:
        # 第一遍：找到所有带有 AtomMapNum 的原子
        join_dict = defaultdict(list)
        for atom in input_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                join_dict[map_num].append(atom)
        
        # 第二遍：将 AtomMapNum 转移到相邻原子
        for idx, atom_list in join_dict.items():
            if len(atom_list) == 2:
                atm_1, atm_2 = atom_list
                # 获取通配符的相邻原子
                neighbors_1 = [x.GetOtherAtom(atm_1) for x in atm_1.GetBonds()]
                neighbors_2 = [x.GetOtherAtom(atm_2) for x in atm_2.GetBonds()]
                
                if neighbors_1 and neighbors_2:
                    nbr_1 = neighbors_1[0]
                    nbr_2 = neighbors_2[0]
                    # 将 AtomMapNum 转移到相邻原子
                    nbr_1.SetAtomMapNum(idx)
                    nbr_2.SetAtomMapNum(idx)
            elif len(atom_list) != 2:
                # 如果某个 map_num 对应的原子不是 2 个，说明配对有问题
                print(f"Warning: AtomMapNum {idx} has {len(atom_list)} atoms instead of 2")
                return None
        
        # 第三遍：删除所有通配符原子
        new_mol = Chem.DeleteSubstructs(input_mol, Chem.MolFromSmarts('[#0]'))
        
        # 第四遍：收集需要连接的原子对
        bond_join_dict = defaultdict(list)
        for atom in new_mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num > 0:
                bond_join_dict[map_num].append(atom.GetIdx())
        
        # 第五遍：使用 EditableMol 添加键
        em = EditableMol(new_mol)
        for idx, atom_indices in bond_join_dict.items():
            if len(atom_indices) == 2:
                start_atm, end_atm = atom_indices
                em.AddBond(start_atm, end_atm, order=Chem.rdchem.BondType.SINGLE)
        
        final_mol = em.GetMol()
        
        # 清理 AtomMapNum
        for atom in final_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        
        final_mol = Chem.RemoveHs(final_mol)
        
        return final_mol
        
    except Exception as e:
        print(f"Welding error: {e}")
        return None


def main():
    # 配置输入与输出
    WARHEAD_PAIR = "N1C(=O)CC(C)(c2cccc(*)c2)CC1(=O)|CC1CCCN(C*)C1"  # 两个片段，用 | 分隔
    input_file = "sample_case2_npu.txt"
    output_file = "joined_results.txt"
    
    # 拆分两个端点片段
    frag1, frag2 = WARHEAD_PAIR.split('|')
    
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 '{input_file}'，请检查路径。")
        return

    print("开始处理 Linker 拼接...")
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        # 写入结果文件的表头
        f_out.write("Original_Linker\tJoined_Full_SMILES\n")
        
        for line in f_in:
            linker = line.strip()
            if not linker:  # 跳过空行
                continue
            
            # 标准化 Linker 中的通配符（确保 RDKit 能识别 [*] 或 *）
            if "*" in linker and "[*]" not in linker:
                linker_rdkit = linker.replace("*", "[*]")
            else:
                linker_rdkit = linker
            
            # 执行拼接
            full_smiles = link_fragments(frag1, frag2, linker_rdkit)
            
            # 将原始 Linker 和拼接后的完整 SMILES 写入结果文件
            f_out.write(f"{linker}\t{full_smiles}\n")
            count += 1

    print(f"处理完成！成功处理 {count} 个 Linker，结果已写入 '{output_file}'。")


if __name__ == '__main__':
    main()
