#!/usr/bin/env bash
# Read-only Slurm GPU-hours/cost + VAST usage. No Python, Conda or saved logs.
set -euo pipefail
export TZ="${UAV_USAGE_TZ:-Asia/Shanghai}"
export LC_NUMERIC=C

since="${UAV_USAGE_START:-1970-01-01}"
until="now"
vast_path="${UAV_VAST_DIR:-$HOME/vast}"
rate5090="${UAV_RATE_5090:-2.70}"
rate4090="${UAV_RATE_4090:-2.16}"
scan_du=1
query_storage=1

usage() {
    printf '%s\n' \
        '用法: cluster_usage [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--gpu-only] [--no-du]' \
        '      bash cluster_usage.sh --install' \
        '默认查询当前用户的全部可用记账历史，结束时间不包含在统计区间内。' \
        '默认显示 GPU 卡时、费用和 VAST 空间用量。' \
        '参数:' \
        '  --since YYYY-MM-DD  统计开始时间（包含）。' \
        '  --until YYYY-MM-DD  统计结束时间（不包含，默认当前时间）。' \
        '  --gpu-only         只查询 GPU 卡时和费用，跳过所有空间查询（df、quota、du）。' \
        '  --no-du            只跳过较慢的目录扫描，仍然查询 df 和 quota。' \
        '  --install          安装 cluster_usage shell 命令。' \
        '  -h, --help         显示此帮助。' \
        '示例: cluster_usage --gpu-only --since 2026-09-01' \
        '环境变量: UAV_USAGE_START, UAV_VAST_DIR, UAV_RATE_5090, UAV_RATE_4090,' \
        '          UAV_USAGE_TZ（默认北京时间）, UAV_USAGE_RC（安装目标 rc 文件）。'
}

install_command() {
    local script_path rc_path
    script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/cluster_usage.sh"
    if [[ ${SHELL:-} == */zsh ]]; then
        rc_path="${UAV_USAGE_RC:-${ZDOTDIR:-$HOME}/.zshrc}"
    else
        rc_path="${UAV_USAGE_RC:-$HOME/.bashrc}"
    fi
    if ! grep -Fq '# CLUSTER_USAGE_COMMAND' "$rc_path" 2>/dev/null; then
        {
            printf '\n# CLUSTER_USAGE_COMMAND\nexport CLUSTER_USAGE_SCRIPT=%q\n' "$script_path"
            printf '%s\n' 'cluster_usage() { bash "$CLUSTER_USAGE_SCRIPT" "$@"; }'
        } >> "$rc_path"
        printf '已写入 %s\n' "$rc_path"
    else
        printf '%s 已有 cluster_usage 配置；若移动脚本，请更新其中的 CLUSTER_USAGE_SCRIPT。\n' "$rc_path"
    fi
    printf '当前终端执行: source %q\n之后输入: cluster_usage\n' "$rc_path"
}

while (($#)); do
    case "$1" in
        --since|--until)
            [[ $# -ge 2 ]] || { usage >&2; exit 2; }
            if [[ $1 == --since ]]; then since=$2; else until=$2; fi
            shift 2 ;;
        --no-du) scan_du=0; shift ;;
        --gpu-only) query_storage=0; shift ;;
        --install) install_command; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        *) printf '未知参数: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

start_epoch=$(date -d "$since" +%s) || exit 2
end_epoch=$(date -d "$until" +%s) || exit 2
((end_epoch > start_epoch)) || { echo '结束时间必须晚于开始时间。' >&2; exit 2; }
for rate in "$rate5090" "$rate4090"; do
    [[ $rate =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo '单价必须是非负数字。' >&2; exit 2; }
done
# Resolve "now" once: running jobs and the query use exactly the same cutoff.
since=$(date -d "@$start_epoch" +%Y-%m-%dT%H:%M:%S)
until=$(date -d "@$end_epoch" +%Y-%m-%dT%H:%M:%S)
printf '用户: %s | 时区: %s\n区间: %s 至 %s\n' "${USER:-$(id -un)}" "$TZ" "$since" "$until"
printf '价格: 5090 = %s 元/卡时；4090 = %s 元/卡时\n' "$rate5090" "$rate4090"
echo '查询 Slurm 记账记录（最多等待60秒）...'

report_gpu() {
    local records
    # No -T: preserve original timestamps/ElapsedRaw, then clip/split locally.
    # -X removes steps, --array expands workers, -D retains distinct job records.
    if ! records=$(timeout 60s sacct -X --array -D -n -P --noconvert \
        -u "${USER:-$(id -un)}" -S "$since" -E "$until" \
        --format=Cluster,DBIndex,JobIDRaw,Partition,Start,End,ElapsedRaw,AllocTRES); then
        echo 'Slurm 查询失败或超时；本次卡时/费用未知，不输出虚假的零用量。' >&2
        return 1
    fi
    if ! awk -F'|' -v begin="$start_epoch" -v finish="$end_epoch" \
        -v r5090="$rate5090" -v r4090="$rate4090" '
    function epoch(s, t) {
        if (s !~ /^[0-9][0-9][0-9][0-9]-/) return -1
        t=s; gsub(/[-T:]/," ",t); return mktime(t " -1")
    }
    function add(key, a,b,c) { h5[key]+=a; h4[key]+=b; hx[key]+=c }
    function row(key) {
        printf "%s|%.9f|%.9f|%.9f|%.9f|%.9f|%.9f\n",key,h5[key],h4[key],hx[key],h5[key]*r5090,h4[key]*r4090,h5[key]*r5090+h4[key]*r4090
    }
    {
        if (NF<8 || $3 ~ /[.]/) next
        # DBIndex identifies the database record, including reused job IDs.
        key=$1 SUBSEP $2
        if ($2=="" || $2=="0") key=$1 SUBSEP $3 SUBSEP $5
        if (seen[key]++) next
        elapsed=$7+0
        if (elapsed<=0) next
        g5=0; g4=0; generic=-1; typed=0
        n=split($8,tres,",")
        for(i=1;i<=n;i++) {
            split(tres[i],pair,"="); count=pair[2]+0
            if(pair[1]=="gres/gpu") generic=count
            else if(pair[1] ~ /^gres\/gpu:/) {
                typed+=count
                if(pair[1] ~ /5090/) g5+=count
                else if(pair[1] ~ /4090/) g4+=count
            }
        }
        gpu=(generic>=0 ? generic : typed)
        if(gpu<=0) next
        if(typed==0) {
            if($4 ~ /5090/) g5=gpu
            else if($4 ~ /4090/) g4=gpu
        }
        gx=gpu-g5-g4
        if(gx<0) { bad++; next }
        s=epoch($5); e=epoch($6)
        if(e<0) e=finish
        if(s<0 || e<=s) { bad++; next }
        # Suspended jobs: ElapsedRaw excludes suspension, but sacct does not
        # locate each suspension interval. Prorate across days and disclose it.
        factor=elapsed/(e-s)
        if(factor<0.999 || factor>1.001) adjusted++
        if(factor>1) factor=1
        if(s<begin) s=begin
        if(e>finish) e=finish
        if(e<=s) next
        jobs++
        while(s<e) {
            day=strftime("%Y-%m-%d",s)
            midnight=mktime(strftime("%Y %m %d 00 00 00",s) " -1")
            # Find the next calendar midnight, including DST transitions.
            nextday=strftime("%Y %m %d",midnight+36*3600)
            stop=mktime(nextday " 00 00 00 -1")
            if(stop>e) stop=e
            hours=(stop-s)*factor/3600
            add("D|" day,hours*g5,hours*g4,hours*gx)
            month=substr(day,1,7); months[month]=1
            add("M|" month,hours*g5,hours*g4,hours*gx)
            add("T|TOTAL",hours*g5,hours*g4,hours*gx)
            s=stop
        }
    }
    END {
        for(m in months) nm++
        for(k in h5) if(k ~ /^D[|]/ || (nm>1 && k ~ /^M[|]/)) row(k)
        row("T|TOTAL")
        if(!jobs) print "所选区间内没有可统计的 GPU 用量。" > "/dev/stderr"
        if(hx["T|TOTAL"]>0) print "注意：其他型号卡时已列入总卡时，但没有单价，其费用未计入。" > "/dev/stderr"
        if(adjusted) printf "注意：%d 条记录的运行秒数与起止跨度不同（可能有暂停）；日费用按运行秒数比例分摊，非精确暂停时段账单。\n",adjusted > "/dev/stderr"
        if(bad) printf "注意：%d 条 GPU 记录缺少有效时间或资源数据，已跳过；统计不完整。\n",bad > "/dev/stderr"
    }' <<< "$records" | LC_ALL=C sort -t'|' -k1,1 -k2,2 | awk -F'|' '
    $1!=section {
        section=$1
        print ""
        print (section=="D" ? "每日统计（仅显示有用量的日期）" : section=="M" ? "每月统计" : "总计")
        printf "%-12s %11s %11s %11s %12s %12s %12s %12s\n", "Date", "5090(h)", "4090(h)", "Other(h)", "Total(h)", "5090(CNY)", "4090(CNY)", "Cost(CNY)"
    }
    { printf "%-12s %11.3f %11.3f %11.3f %12.3f %12.2f %12.2f %12.2f\n",$2,$3,$4,$5,$3+$4+$5,$6,$7,$8 }
    '; then
        echo '记账解析失败；请检查系统 awk 是否支持 mktime/strftime，统计不可用。' >&2
        return 1
    fi
    echo '费用为按分配卡数×运行时间×单价估算；失败/取消前的运行也计入，排队不计。'
    echo '仅覆盖 Slurm 保留的历史；平台实际扣费、折扣和存储费用以账单为准。'
}

report_vast() {
    printf '\nVAST 用量: %s\n' "$vast_path"
    if [[ ! -d $vast_path ]]; then
        echo '目录不存在；可通过 UAV_VAST_DIR 指定。' >&2
        return 1
    fi
    echo '文件系统容量（共享盘的 Avail 不一定是个人可用额度）：'
    timeout 15s df -hT -- "$vast_path/" || echo 'df 查询失败或超时。' >&2
    echo '个人 quota（查看 VAST 对应条目；空白不代表额度无限）：'
    if command -v quota >/dev/null 2>&1; then
        timeout 15s quota -s || echo 'quota 查询失败、超时或平台未提供；请以平台额度为准。' >&2
    else
        echo '系统没有 quota 命令，请到平台查询个人存储额度。'
    fi
    if ((scan_du)); then
        echo '目录实际占用（最多扫描60秒；大量小文件时可能较慢）：'
        timeout 60s du -sh -- "$vast_path/" || echo 'du 未完整完成，不能据此认定目录用量；可单独运行 du -sh。' >&2
    fi
}

status=0
report_gpu || status=1
if ((query_storage)); then
    report_vast || status=1
fi
exit "$status"
