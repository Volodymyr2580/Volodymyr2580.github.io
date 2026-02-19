---
layout: page
permalink: /diary/coding/index.html
title: coding
---

# Coding Diary
暂时先统一放在一个markdown文件中，后续如果发现某一板块的内容更新的较多了，翻看起来比较复杂，那再单独进行整理归档。

### 服务器端操作
nohup 运行python:
nohup python your_script.py > output.log 2>&1 &
查看进程：
ps aux | grep your_script.py
ps -p <PID>


### 在linux系统上配置conda

##### 下载Miniconda安装脚本（Python 3.9版本）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

##### 赋予执行权限
chmod +x Miniconda3-latest-Linux-x86_64.sh

##### 运行安装程序（-b静默安装，-p指定路径）
./Miniconda3-latest-Linux-x86_64.sh

##### 激活配置（或重启终端）
source ~/.bashrc

### Windows命令行
关于cmd有关的操作归纳在这块
#### 📁 文件与目录操作
```cmd
dir                # 列出目录内容
cd <目录>          # 切换目录
cd ..              # 返回上级目录
mkdir <目录名>     # 创建目录
rmdir /s /q <目录> # 强制删除目录及内容
copy <源> <目标>   # 复制文件
xcopy <源> <目标>  # 复制目录（含子目录）
del <文件名>       # 删除文件
move <源> <目标>   # 移动/重命名
type <文件名>      # 查看文本内容
ren <旧名> <新名>  # 重命名文件
```
#### 系统管理
```cmd
systeminfo         # 显示系统信息
tasklist           # 列出运行进程
taskkill /IM <进程名> /F  # 终止进程
shutdown /s /t 0   # 立即关机
shutdown /r        # 重启
wmic cpu get name  # 查询CPU信息
wmic memorychip get capacity # 查看内存大小
```
#### 实用工具
```cmd
cls                # 清屏
echo > file.txt    # 创建文件
findstr "文本" 文件 # 搜索文本
start <文件>       # 打开文件
sfc /scannow       # 扫描修复系统文件
```

### linux命令行
#### 文件与目录操作
ls                 # 列出目录内容
cd <目录>          # 切换目录
cd ~               # 返回家目录
pwd                # 显示当前路径
mkdir <目录名>     # 创建目录
rm -r <目录>       # 删除目录及内容
cp <源> <目标>     # 复制文件/目录
mv <源> <目标>     # 移动/重命名
rm <文件>          # 删除文件
cat <文件>         # 查看文件内容
touch <文件名>     # 创建空文件

#### 系统管理
top                # 动态查看进程（类似任务管理器）
htop               # 增强版top（需安装）
ps aux             # 查看所有进程
kill -9 <PID>      # 强制终止进程
uname -a           # 查看系统信息
free -h            # 查看内存使用
df -h              # 查看磁盘空间
curl <URL>         # 网络数据传输
wget <URL>         # 下载文件

#### 文本处理
grep "文本" 文件   # 搜索文本
nano <文件>        # 简单文本编辑
vim <文件>         # 高级文本编辑
head -n 5 <文件>   # 显示文件前5行
tail -f <日志文件> # 实时查看日志、

#### 权限管理
chmod 755 <文件>   # 修改文件权限
chown 用户:组 <文件> # 修改文件所有者
sudo <命令>        # 以管理员身份执行
su -               # 切换为root用户

#### 实用工具
clear              # 清屏
man <命令>         # 查看命令手册
history            # 查看命令历史
echo $PATH         # 查看环境变量
tar -xzvf file.tar.gz # 解压tar包
apt install <包名> # 安装软件（Debian/Ubuntu）
yum install <包名> # 安装软件（CentOS/RHEL）

### devito

### python basic

#### 画图 matplotlib

#### Numpy

#### Pandas

#### torch













