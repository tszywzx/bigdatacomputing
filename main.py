import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import time

def load_data(filename, block_size=100000):
    """分块加载数据，构建稀疏链接矩阵"""
    # 第一遍：确定节点数量和每个块的边界
    max_node = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                from_node, to_node = map(int, line.strip().split())
                max_node = max(max_node, from_node, to_node)
    
    n = max_node + 1
    num_blocks = (n + block_size - 1) // block_size
    M = lil_matrix((n, n), dtype=np.float32)
    
    # 第二遍：分块处理数据
    for block_idx in range(num_blocks):
        start = block_idx * block_size
        end = min((block_idx + 1) * block_size, n)
        print(f"处理块 {block_idx+1}/{num_blocks} (节点 {start}-{end-1})")
        
        # 只加载影响当前块的边
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    from_node, to_node = map(int, line.strip().split())
                    if start <= to_node < end:
                        M[to_node, from_node] += 1
    
    return M.tocsr(), n

def normalize_matrix(M, n, block_size=10000):
    """分块归一化矩阵"""
    # 分块计算列和
    col_sums = np.zeros(n, dtype=np.float32)
    num_blocks = (n + block_size - 1) // block_size
    
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, n)
        col_sums[start:end] = M[:, start:end].sum(axis=0).A1
    
    # 处理死端
    dead_ends = np.where(col_sums == 0)[0]
    if len(dead_ends) > 0:
        print(f"发现 {len(dead_ends)} 个死端节点")
        col_sums[dead_ends] = 1  # 相当于设置为均匀分布
    
    # 分块归一化
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, n)
        for j in range(start, end):
            if col_sums[j] > 0:
                M[:, j] = M[:, j] / col_sums[j]
    
    return M

def pagerank(M, n, teleport=0.8, block_size=10000, max_iter=100, tol=1e-6):
    """分块PageRank计算"""
    v = np.ones(n, dtype=np.float32) / n
    teleport_vec = np.ones(n, dtype=np.float32) / n
    num_blocks = (n + block_size - 1) // block_size
    
    for _ in range(max_iter):
        v_new = np.zeros(n, dtype=np.float32)
        
        # 分块矩阵乘法
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, n)
            block = M[start:end, :]
            v_new[start:end] = block.dot(v)
        
        v_new = teleport * v_new + (1 - teleport) * teleport_vec
        
        if np.linalg.norm(v_new - v, 1) < tol:
            break
        v = v_new
    
    return v

def save_results(scores, filename="Res.txt", top_k=100):
    """保存Top K结果"""
    # 排除零节点（未连接的节点）
    nonzero_indices = np.where(scores > 0)[0]
    nonzero_scores = scores[nonzero_indices]
    
    # 按分数排序
    sorted_indices = np.argsort(-nonzero_scores)[:top_k]
    
    with open(filename, 'w') as f:
        for i, idx in enumerate(sorted_indices):
            node_id = nonzero_indices[idx]
            score = nonzero_scores[idx]
            # 如果是最后一行，不添加换行符
            if i == len(sorted_indices) - 1:
                f.write(f"{node_id} {score:.8f}")
            else:
                f.write(f"{node_id} {score:.8f}\n")

def find_optimal_alpha(M, n, alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]):
    for alpha in alphas:
        start = time.time()
        scores = pagerank(M, n, teleport=alpha)
        iterations = len(scores)  # 假设pagerank返回迭代历史
        print(f"α={alpha}: 迭代{iterations}次, 用时{time.time()-start:.5f}s")

def compare_results(M, n, alphas=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]):
    results = {}
    for alpha in alphas:
        scores = pagerank(M, n, teleport=alpha)
        top100 = set(np.argsort(-scores)[:100])
        results[alpha] = top100
    
    # 计算两两重合率
    for a1, a2 in [(0.85, 0.1), (0.85, 0.2), (0.85, 0.3), (0.85, 0.4), (0.85, 0.5), (0.85, 0.6), (0.85, 0.7), (0.85, 0.8), (0.85, 0.9), (0.85, 0.95), (0.85, 0.99)]:
        overlap = len(results[a1] & results[a2]) / 100
        print(f"α={a1} vs α={a2}: Top100重合率{overlap:.1%}")

def main():
    start_time = time.time()
    
    # 参数设置
    data_file = "Data.txt"
    result_file = "Res.txt"
    teleport = 0.85
    block_size = 50000  # 根据内存调整
    
    print("加载数据...")
    M, n = load_data(data_file, block_size)
    
    print("归一化矩阵...")
    M = normalize_matrix(M, n, block_size)
    
    print("计算PageRank...")
    scores = pagerank(M, n, teleport=teleport, block_size=block_size)
    
    print("保存结果...")
    save_results(scores, result_file)
    
    print(f"完成! 用时: {time.time() - start_time:.2f}秒")

    find_optimal_alpha(M, n)

    compare_results(M, n)


if __name__ == "__main__":
    main()