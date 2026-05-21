# AGENTS.md — E2E-Evaluation Codex 工作规范

本文件是本仓库给 AI coding agent / Codex 的长期工作规范。  
所有代码修改前，必须先阅读本文件，并严格遵守。

当前项目是博士论文实验代码仓库，主题是：

> 自动驾驶端到端系统的 behavior embedding / style drift / interaction-aware evaluation。

---

## 1. 总原则

1. 每次修改必须说明：
   - 修改了哪些文件；
   - 为什么修改；
   - 如何运行；
   - 期望行为；
   - 通过标准；
   - 实际执行了哪些测试。
2. 不要为了修一个 bug 顺手重构无关模块。
3. 不要修改 Stage 4 已经稳定的历史逻辑，除非任务明确要求。
4. 不要删除历史输出说明、已有文档章节。
5. 不要凭空创建不存在的数据文件路径。
6. 不要假设本地有 `/tmp/old.py`、notebook 临时文件、Codex 草稿文件。
7. 不要用 `exec()` 动态加载项目源码。
8. 不要引入任何依赖仓库外临时文件的逻辑。
9. README.md 讲研究方向、当前进展、阶段结论、限制、下一步，如有必要需要更新。
10. QUICK_REFERENCE.md 讲命令、期望行为、通过标准。
11. 如有training, evaluate embedding, export embedding等消耗较大计算的过程，请加入进度条显示功能。
12. 关于stage5的所有设计和验证都更新到 07_stage5_interaction_design.md文档中。
13. chatgpt先 review 当前代码 / 文档 / 分支状态，把问题整理成 GitHub Issue，Issue 里写清楚：
   - 背景
   - 目标
   - 要改哪些文件
   - 实现要求
   - 验收标准
   - 运行命令
用户把 Issue 链接交给 Codex， Codex 修改并提交， chatgpt再 review Codex 的改动，不合格就继续开 follow-up issue
14. 关于stage6的所有设计和验证都更新到 stage6_unpaired_style_drift_protocol.md文档中。

---

## 2. 文档更新规则

每次新增工具、修改命令、改变实验流程，必须更新文档。

优先更新：

```text
QUICK_REFERENCE.md
```

如涉及设计变更，也更新对应 stage 文档，例如：

```text
07_stage5_interaction_design.md
00_plans.md
paper_outline.md
```

### QUICK_REFERENCE.md 的写法要求

必须用中文写清楚三部分：

```markdown
## 1. 命令

给出可以直接复制运行的命令。

## 2. 期望行为

说明这个命令会读取什么、生成什么、不会做什么。

## 3. 通过标准

列出明确可检查的标准。
```

不要只写一句“运行脚本”。  
不要只写英文。  
不要只写内部实现，不写用户怎么运行。

---

## 4. 每次改代码后的最低测试要求

每次修改 Python 文件后，至少运行：

```bash
python -m py_compile tools/修改过的文件.py
python tools/check_no_tmp_dependencies.py
```

如果修改了多个 tools 文件，应运行：

```bash
python -m py_compile tools/*.py
python tools/check_no_tmp_dependencies.py
```

如果本地数据存在，必须运行最小 smoke test。  
如果本地数据不存在，必须在回复中明确说明：

```text
未运行真实数据测试，因为本地没有对应数据目录。
```

不要声称“已验证”，除非真的运行过命令。

---

## 3. 禁止事项

严禁出现以下代码或行为：

```python
Path('/tmp/old.py').read_text()
exec(...)
```

严禁依赖：

```text
/tmp/old.py
/tmp/*.py
notebook 临时文件
本地草稿文件
Codex session 临时文件
```

严禁写出：

```text
我修好了
```

但没有说明运行了哪些测试。

严禁在 sharded dataset 上默认合并所有大数组，例如：

```text
context_traj.npy
neighbor_seq.npy
ego_seq.npy
```

除非用户明确要求，并且说明内存风险。

---

## 4. 代码风格要求

1. Python 脚本放在 `tools/` 下。
2. 脚本顶部添加项目根路径，保证直接运行不报 `No module named tools`：

```python
#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

3. CLI 参数必须清楚。
4. 错误信息必须说明具体缺哪个文件、哪个字段、哪个 shape 不对。
5. 不要吞异常。
6. 不要 silent fallback，fallback 必须写 warning。
7. 输出 JSON 要 indent=2，中文报告用 UTF-8。
8. 图表用 matplotlib，保存到 out_dir。
9. 大数组读取优先 `np.load(..., mmap_mode="r")`。

---

## 5. 每次任务完成后的回复格式

每次完成任务，必须按以下格式回复：

```markdown
## 修改文件

- path/to/file1.py
- path/to/file2.md

## 主要改动

1. ...
2. ...

## 已运行命令

```bash
...
```

## 测试结果

- py_compile: pass
- smoke test: pass / not run，原因：...
- real data test: pass / not run，原因：...

## 生成文件

- ...

## 下一步建议

...
```

不要只说“完成了”。

---

## 6. 对 Codex 的特别要求

如果用户要求“写 prompt 给 Codex”，不要直接改代码。  
如果用户要求“改代码”，再改代码。

当前用户经常采用流程：

```text
ChatGPT 设计方案
↓
Codex 执行代码
↓
用户本地运行
↓
ChatGPT 分析结果
```

所以 Codex 必须严格按任务范围执行，不要自由发挥。

---
