import os
import requests
import queue
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
# 你的反代链接前缀
PROXY_PREFIX = "https://proxy.buaa.de5.net/hyx666/"
REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SAVE_DIR = "/data0/yejinxuan/ce-ais/data/LIBERO-datasets"

# 并发下载数（同时下载几个文件，建议设置 3~8，太高可能会被代理服务器限制）
MAX_WORKERS = 5

# 【重要提示】填入你的 WebVPN Cookie
VPN_COOKIE = "" 
# ==========================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Cookie": VPN_COOKIE
}

def get_repo_files():
    """扫描获取远端所有文件列表（扁平化）"""
    print("🔍 正在扫描远端目录结构获取所有文件列表，请稍候...")
    files_to_download = []
    folders_to_explore = [""]

    while folders_to_explore:
        current_path = folders_to_explore.pop(0)
        api_url = f"{PROXY_PREFIX}https://huggingface.co/api/datasets/{REPO_ID}/tree/main/{quote(current_path)}"
        
        try:
            response = requests.get(api_url, headers=HEADERS, timeout=15)
            if response.status_code == 200:
                items = response.json()
                for item in items:
                    if item['type'] == 'directory':
                        folders_to_explore.append(item['path'])
                    elif item['type'] == 'file':
                        files_to_download.append(item['path'])
            else:
                print(f"❌ 读取目录失败! 状态码: {response.status_code} 路径: {current_path}")
        except Exception as e:
            print(f"❌ 网络请求异常: {e}")
            
    return files_to_download

def download_worker(file_path, pos_queue):
    """单个文件的下载工作线程"""
    download_url = f"{PROXY_PREFIX}https://huggingface.co/datasets/{REPO_ID}/resolve/main/{quote(file_path)}"
    local_path = os.path.join(SAVE_DIR, file_path)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    local_size = 0
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        
    headers = HEADERS.copy()
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        
    short_name = file_path.split('/')[-1]
    
    # 从队列获取一个可用的终端行位置，防止进度条互相覆盖
    pos = pos_queue.get() 
    try:
        with requests.get(download_url, headers=headers, stream=True, timeout=30) as r:
            if r.status_code == 416:
                tqdm.write(f"✅ [跳过] 文件已完整: {short_name}")
                return True
                
            if r.status_code not in [200, 206]:
                tqdm.write(f"❌ [失败] 状态码 {r.status_code}，文件: {short_name}")
                return False
            
            content_length = r.headers.get('content-length')
            total_size = int(content_length) + local_size if content_length else None
                
            mode = 'ab' if r.status_code == 206 else 'wb'
            
            # leave=False 表示下载完成后清空这一行的进度条，给下一个文件腾出空间
            with open(local_path, mode) as f, tqdm(
                desc=f"[{pos}] ⚡ {short_name[:20]:<20}", 
                total=total_size,
                initial=local_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                position=pos,
                leave=False, 
                ncols=100
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
                        
        tqdm.write(f"🎉 [完成] {short_name}")
        return True
        
    except Exception as e:
        tqdm.write(f"⚠️ [中断] {short_name} -> {e}")
        return False
    finally:
        # 任务结束，把该行位置交还给队列
        pos_queue.put(pos)

def main():
    print("="*60)
    print("🚀 开始多线程同步 HuggingFace 数据集")
    print(f"📁 目标路径: {SAVE_DIR}")
    print("="*60 + "\n")
    
    files_to_download = get_repo_files()
    if not files_to_download:
        print("没有找到任何文件，请检查网络或代理 Cookie！")
        return
        
    print(f"\n📦 共扫描到 {len(files_to_download)} 个文件，启动 {MAX_WORKERS} 个并发线程下载...\n")
    
    # 创建进度条位置队列，供工作线程使用 (位置1 到 位置MAX_WORKERS)
    pos_queue = queue.Queue()
    for i in range(1, MAX_WORKERS + 1):
        pos_queue.put(i)
        
    success_count = 0
    
    # 启动线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 将所有文件扔进线程池
        futures = {executor.submit(download_worker, f, pos_queue): f for f in files_to_download}
        
        # position=0 留给总进度条
        with tqdm(total=len(files_to_download), position=0, desc="🏆 总计任务进度", leave=True, ncols=100) as main_bar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                main_bar.update(1)
                
    print(f"\n\n🎯 全部任务结束！成功下载: {success_count}/{len(files_to_download)}")

if __name__ == "__main__":
    main()