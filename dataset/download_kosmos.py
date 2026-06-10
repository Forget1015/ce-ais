import os
import requests
import queue
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ================= 配置区 =================
# 你的反代链接前缀
PROXY_PREFIX = "https://proxy.buaa.de5.net/hyx666/"
# 领导要求的模型 ID
REPO_ID = "microsoft/kosmos-2-patch14-224"
# 领导要求的保存路径
SAVE_DIR = "/data0/yejinxuan/hf_cache/hub/kosmos-2-patch14-224/"

# 并发下载数（建议设置 3~5）
MAX_WORKERS = 5

# 【重要提示】填入你的 WebVPN Cookie (格式类似: "wengine_vpn_ticket=xxxx; SESSION=yyyy")
VPN_COOKIE = "" 
# ==========================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Cookie": VPN_COOKIE
}

def get_repo_files():
    """扫描获取远端所有文件列表（针对 Model 做了路径修改）"""
    print("🔍 正在扫描模型目录结构，请稍候...")
    files_to_download = []
    folders_to_explore = [""]

    while folders_to_explore:
        current_path = folders_to_explore.pop(0)
        # 注意：这里改成了 /api/models/
        api_url = f"{PROXY_PREFIX}https://huggingface.co/api/models/{REPO_ID}/tree/main/{quote(current_path)}"
        
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
    # 注意：这里去掉了 /datasets/，直接接模型 ID
    download_url = f"{PROXY_PREFIX}https://huggingface.co/{REPO_ID}/resolve/main/{quote(file_path)}"
    local_path = os.path.join(SAVE_DIR, file_path)
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    local_size = 0
    if os.path.exists(local_path):
        local_size = os.path.getsize(local_path)
        
    headers = HEADERS.copy()
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        
    short_name = file_path.split('/')[-1]
    
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
        pos_queue.put(pos)

def main():
    print("="*60)
    print(f"🚀 开始多线程同步模型: {REPO_ID}")
    print(f"📁 目标路径: {SAVE_DIR}")
    print("="*60 + "\n")
    
    files_to_download = get_repo_files()
    if not files_to_download:
        print("没有找到任何文件，请检查网络或代理 Cookie 是否过期！")
        return
        
    print(f"\n📦 共扫描到 {len(files_to_download)} 个文件，启动 {MAX_WORKERS} 个并发线程下载...\n")
    
    pos_queue = queue.Queue()
    for i in range(1, MAX_WORKERS + 1):
        pos_queue.put(i)
        
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_worker, f, pos_queue): f for f in files_to_download}
        
        with tqdm(total=len(files_to_download), position=0, desc="🏆 总计任务进度", leave=True, ncols=100) as main_bar:
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                main_bar.update(1)
                
    print(f"\n\n🎯 全部任务结束！成功下载: {success_count}/{len(files_to_download)}")
    print(f"模型已保存在: {SAVE_DIR}")

if __name__ == "__main__":
    main()