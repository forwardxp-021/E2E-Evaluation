#!/usr/bin/env python3
"""Package the fixed D4 analyses and limited provenance, without rescanning historical assets."""
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, csv, json, subprocess
import numpy as np
from tools.rbr_v2_d4_analysis import ROOT,OUT,DOC,D3,PLAN,read,write,check,sha


def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    r=read(OUT/'real_diagnostics.json');s=read(OUT/'synthetic_results.json')
    fig,ax=plt.subplots(1,3,figsize=(17,5.5),gridspec_kw={'width_ratios':[1,1,1.5]})
    for name,color in [('RBR64','tab:blue'),('H30','tab:orange')]:
        vals=[v for v in r['kernel_and_information'] if v['representation']==name]
        ax[0].plot(['cal','AA','AB'],[v['BB_kernel_offdiag'][2] for v in vals],'o-',label=name,color=color)
        ax[1].plot(['cal','AA','AB'],[v['median_within_over_between'] for v in vals],'o-',label=name,color=color)
    ax[0].set(title='Source background: median BB similarity',ylabel='Frozen RBF kernel')
    ax[1].set(title='Paired / between-source distance',ylabel='Median standardized-distance ratio',yscale='log')
    cases=s['rows'];x=np.arange(len(cases));ax[2].bar(x-.17,[v['conditional_rejection_rate'] for v in cases],.34,label='Conditional pair randomization')
    ax[2].bar(x+.17,[v['reference_global_threshold_alarm_rate'] for v in cases],.34,label='Reference-pool global threshold')
    ax[2].set_xticks(x);ax[2].set_xticklabels(['Null','Source\nshift','Symmetric\nmask','Extreme\nscale','Identical','Effect','Effect +\nbackground','Invalid\nmask'],fontsize=7)
    ax[2].axhline(.05,color='red',ls='--');ax[2].set(title='Predeclared synthetic stress cases',ylabel='Alarm fraction (1000 datasets/case)',ylim=(0,1.05))
    for a in ax:a.grid(alpha=.2)
    for a in ax[:2]:a.legend(fontsize=7)
    ax[2].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.21))
    fig.tight_layout();fig.subplots_adjust(bottom=.23);fig.savefig(DOC/'D4_Background_And_Synthetic_Stress.png',dpi=160);plt.close(fig)
    rec=list(csv.DictReader((OUT/'exploratory_conditional_randomization.csv').open()));fig,axes=plt.subplots(1,2,figsize=(10,4))
    for a,name in zip(axes,['RBR64','H30']):
        for group,label,color in [('cal','Exposed calibration','gray'),('aa','Exposed D3-AA','tab:blue'),('ab','Exposed D3-AB','tab:orange')]:
            v=[float(d['orbit_mean']) for d in rec if d['representation']==name and d['group']==group]
            w=[float(d['orbit_sd']) for d in rec if d['representation']==name and d['group']==group]
            a.scatter(v,w,alpha=.3,s=10,color=color,label=label)
        a.axvline(read(D3/'D_statistics_results.json')['thresholds'][name]['value'],ls='--',color='red',label='Original global threshold')
        from matplotlib.ticker import MaxNLocator
        a.xaxis.set_major_locator(MaxNLocator(5))
        a.set(title=name+' conditional orbit (exploratory only)',xlabel='Mean over exact1024 pair flips',ylabel='Within-orbit statistic SD');a.legend(fontsize=7);a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(DOC/'D4_Conditional_Orbit_Background.png',dpi=160);plt.close(fig)


def report():
    check();figures();r=read(OUT/'real_diagnostics.json');s=read(OUT/'synthetic_results.json');t=read(OUT/'mathematical_checks.json');stat=read(D3/'D_statistics_results.json')
    table='\n'.join('|'+str(v['case'])+'|'+str(v['conditional_rejection_rate'])+'|'+str(v['reference_global_threshold_alarm_rate'])+'|' for v in s['rows'])
    sensitivity='\n'.join(f"|{v['representation']}|{v['variant']}|{v['sigma']:.6g}|{v['minimum_nonconstant_scale']:.6g}|{v['same_BT_kernel_quantiles'][2]:.6g}|{v['BT_underflow_fraction']:.3f}|" for v in r['scaler_sensitivity'])
    moments=[]
    for role,prefix in [('cal','D_SWAP_CAL_'),('aa','D_AA_CONTROLLED_SWAP_'),('ab','D_AB_PAIRED_PI50_')]:
        for name in ['RBR64','H30']:
            v=np.load(D3/(prefix+name+'_statistics.npy'))
            moments.append(dict(group=role,representation=name,draws=len(v),conditional_mean=float(v.mean()),conditional_SD=float(v.std(ddof=1)),conditional_variance=float(v.var(ddof=1))))
    from tools.rbr_v2_d4_analysis import table as csvtable
    csvtable(OUT/'frozen_D3_score_moments.csv',moments)
    moment_table='\n'.join(f"|{v['representation']}|{v['group']}|{v['conditional_mean']:.9g}|{v['conditional_SD']:.9g}|" for v in moments)
    resource={k:read(OUT/(k+'_resources.json')) for k in ['real','synthetic']}
    text=f'''# RBR-V2-D4：配对零假设与跨来源校准科学分析

**D4_PROSPECTIVE_SAP_AMENDMENT_REQUIRED**

结论：D3的RBR留出AA 100%报警主要由“共享来源的MMD背景偏移＋不同来源池的核几何迁移＋小样本预处理”共同解释；没有发现算式/缓存实现错误。原paired方案不能凭D3证据直接进入Primary。只增加来源数量不足以保证解决分布不匹配和H30结构性近零尺度。原SAP非配对方案未被本轮否定，但独立来源、校准与功效要求仍未满足。

固定RBR在60个真实D SESSION上的成对变化非零且方向较一致；现有证据**不支持“RBR完全没有目标变化信息”**，也**不能识别这些信息完全来自双制动时序、不能证明RBR优于H30**。`D4_RBR_INFORMATION_LIMITATION_EVIDENCED = NOT_ESTABLISHED`。所有旧D3报警率、阈值、表示和账本永久保留，没有把旧结果改为PASS。

## 1. 有效合同与实际操作

已完整读取S1 BDD SAP、S1R Owner Approval、UBVC及Primary Freeze Point；S1R明确批准后续修约包，不能因包内历史文件仍保留PROPOSED字样而回退Owner批准。批准保留SESSION聚类、paired/unpaired主张边界、U-only encoder与E3/E4隔离。本轮遵守这些边界，仅D统计开发。

原SAP非配对目标是每侧20个不同SESSION、目标10 Treatment＋10 Baseline的release报警概率差；paired fallback改为相同20源两侧、10源目标替换Treatment，明确只测试controlled assignment/swap null，**不是operational无漂移道路FPR**。SAP已要求pair exchangeability、独立held-out pair日志、源域CI和D覆盖验证，但没有提供这些假设自动成立的保证。

D1无偏两样本公式原样用于D2/D3。D3每池20源，抽20等于取全池，只随机排序、10个处理身份及交换符号。cal/AA对10个非同值pair随机正反；另外10个两侧相同Baseline。AB保持所有Baseline在reference侧、指定10个Treatment在target侧。1999个cal伪release给出固定1900阶阈值，跨池直接应用。它实现了SAP所述交换与固定阈值的一个D开发版本；**但跨池交换性/同总体与误报资格没有成立**，不等于已经认证paired fallback。原S1R/Primary合同未改。

四种零假设必须分开：

1. **独立两样本P=Q**：两组来源独立，标准两样本无偏性质适用。
2. **同SESSION配对交换**：联合分布在每一对交换下不变；同边缘分布P=Q并不充分。例如(X,Y)等概率取(0,1),(1,2),(2,0)，两边边缘相同而反向联合事件概率为0。
3. **固定无序pair集合的条件随机化**：给定unordered pairs、处理子集和固定预处理，标签方向由真实公平硬币产生，或零假设保证其等价于独立公平符号；条件rank检验可有效。D3伪AA的符号是人工生成，因而对其自身轨道成立；不能把有真实干预的pair集合随机换向称作物理无效应。
4. **跨SESSION总体实际FPR**：要求来源总体、身份依赖、数据管线及真实无漂移生成机制合适，并有独立支持。固定60源反复抽样不产生新总体来源。

## 2. 独立数学推导：负值不是数值错误

记(X_i,Y_i)为独立同分布的SESSION对，μ_XY=E[k(X_i,Y_j)]，i≠j；c=E[k(X_i,Y_i)]。原公式的期望为：

`E[T] = MMD²(P_X,P_Y) − (2/m)(c−μ_XY)`。

因此配对条件下原公式虽然沿用“无偏两样本”名称，却**不再是边缘MMD²的无偏估计**。H0边缘相同也可出现负均值。若X=Y，则实际固定池恒等式是`T=2/m×(平均非对角K−1)`。m20时其偏移可接近−0.1，不是可忽略的浮点误差。删除负值或随意去掉cross diagonal都会改变合同，本轮未这样做。`full_BT_pair_U_diagnostic`只用于展示差值，不选作候选。

将pair RKHS均值记a_i=(φX_i+φY_i)/2、半差d_i=(φX_i−φY_i)/2，标签方向s_i∈{{−1,+1}}。令Q_ij=4〈d_i,d_j〉，Q_ii置0，c_m=(2m−1)/(2m²(m−1))，则原公式可以精确写为：

`T(s) = C(pair集合) + c_m × sᵀQ_off s`。

在独立公平符号下，`E_s[T]=C`，`Var_s[T]=2 c_m² Σ_ij Q_off,ij²`。总体方差还包括`Var(C)`及pair集合变化，不能套用独立两样本方差。**全局阈值混入了不同来源的C；逐release条件rank则消去同一轨道内不变的C**，但不会消除来源对检验力的影响。

独立验证：m3离散相关零假设，64个可能数据集穷举，E[T]={t['dependent_equal_marginal_null']['exact64_dataset_mean']:.15f}，推导值{t['dependent_equal_marginal_null']['predicted_mean']:.15f}；m8所有256个方向穷举，均值及方差与上述公式误差<1e−14。另有m2/4/8/20多组独立direct/orbit对照、全局反向不变、ties/退化检查。理论出处的独立样本前提见[Gretton等，2012](https://jmlr.org/papers/v13/gretton12a.html)，群不变性及包含identity的有限随机化论证见[Hemerik与Goeman，2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/)。推导和离散穷举为本轮独立完成，不是引用实现自检代替证明。

## 3. 为什么RBR AA报警100%

冻结RBR阈值−0.0571488993；cal、AA、AB的原统计范围分别约[−0.058871,−0.055044]、[−0.044291,−0.038367]、[−0.064771,−0.062108]。AA整个背景已高于阈值，AB整个背景反而低于阈值。三个池Baseline非对角kernel均值约0.412966、0.561279、0.339223；对应相同Baseline左右复制的诊断基线约−0.058703、−0.043872、−0.066078。它们几乎直接解释了跨池报警方向，**不是RBR在AA“检测到了真实漂移”、也不是AB没有任何成对信号**。

冻结原统计量的均值与条件SD如下；SD只反映固定池的随机子集/方向，不能当作来源总体标准误：

|表示|原池|原统计均值|固定池条件SD|
|---|---|---:|---:|
{moment_table}

全部60真实完整B/T的负交叉对角修正（相对排除同索引cross的pair-U诊断形式）在RBR cal/AA/AB池分别约−0.05756/−0.04314/−0.06513。该数值只说明共享来源依赖量级，pair-U不被自动选用。原统计的负偏移、条件符号方差与来源背景均值变化必须分别报告。

来源差异有实证：cal为18匹兹堡＋2拉斯维加斯；AA为9波士顿＋7拉斯维加斯＋4新加坡；AB为16拉斯维加斯＋3波士顿＋1新加坡。cal对AA/AB地图经验TV均0.90。初速中位数10.09/9.70/11.11m/s，初速范围各约6.29–14.25、5.95–12.53、5.42–19.50m/s；时段采用UTC记录，未伪称本地时区。有效mask中位数0.1563/0.1301/0.1754，邻车presence中位数0/0/0.05625，均有明确池差异。地图、速度、时间、mask彼此混杂，不能仅用60源相关性宣称某一项是唯一原因。所有SESSION与日期、小时、map、mask、presence在CSV/JSON中保留。

## 4. 三个D_fit来源与H30数值风险

完整保留D_fit-only原标准化，常量判定SD≤1e−12设scale1，相同median-positive-distance带宽规则。仅三个Baseline向量，每个维度SD自由度2，带宽只基于3条成对距离；它不是可靠的尾部/来源稳定性证书。预声明的LOO、分池20与全60都是**探索性敏感性分析，全部列出，不选好结果替换原拟合**：

|表示|诊断拟合|σ|最小非恒定scale|同源BT kernel中位数|BT下溢比例|
|---|---|---:|---:|---:|---:|
{sensitivity}

RBR删除一个fit来源即可把60对BT kernel中位数从0.9854变为约0.0000528、0.9302或0.9600；这是直接的高杠杆不稳定证据，不是用LOO挑选更强检测器。

H30 60/60同源BT核值为0，主要由0-based列19/20等时间分箱支配；冻结3fit非恒定最小scale约2.53e−12。真实signed-accel bin5变化约−0.20591m/s²、bin6约−0.003571m/s²，被近零基线SD放大。所有60源按同规则重拟合仍发生100%核下溢，说明**固定Baseline制动脚本的结构性近常量＋1e−12开关规则**也是问题，不只是样本少。H30没有删除特征，没有加floor/换核或调阈值。核下溢可能使信息几何过饱和/来源内区分失真，但固定、标签不变的退化kernel本身不自动破坏条件随机化size；会损害信息解释与power。未来若修尺度规则，须另作物理/测量依据与Owner修约，不能凭本轮报警改善选floor。

## 5. 单一理论候选及合成压力测试

候选`D4_CONDITIONAL_PAIR_RANDOMIZATION_V1`：仍使用原MMD公式、RBR/H30、固定scaler与sigma；每次release固定20 SESSION和10处理身份，在该release内部对whole-pair标签交换，计算`p=(1+#{{T_flip≥T_obs}})/(B+1)`，含identity；可枚举时用全部方向的精确比例。ties用预声明1e−12保守容差，两表示使用完全相同符号与预算；alarm iff p≤.05。原十个相同pair不改变排列，故已有D诊断每次只需枚举2^10=1024方向，重复方向等权不会凭空增加来源。

需要联合分布在逐pair交换群下不变，或真实随机分配保证条件方向均匀；固定预处理必须与被检验标签独立，SESSION间依赖须按真实独立cluster处理。不得在检验目标标签上选择scaler/kernel。物理无效应不自动保证测量mask的标签交换性，尤其一侧系统性缺失时。候选仅对应**受控配对方向/分布差异检验下的Δ报警概率**，沿用α、m、π和ΔBDD函数，不保留在线道路无漂移FPR主张。其科学解释比任意边缘P=Q更强地依赖pair交换假设。

它改变固定cal阈值与null流程，**PROSPECTIVE_SAP_AMENDMENT_REQUIRED**；本轮没有修改现行SAP，没有设置正式Primary阈值或授权身份。

事前计划固定后运行8类×1000合成数据集，每组m20，8维toy representation，sigma2，499随机方向，已知效应只在固定10源前2维加1.5。toy不是RBR/H30模拟性能：

|合成情形|条件随机化报警比例|参考池全局阈值报警比例|
|---|---:|---:|
{table}

正常交换零假设约4.9%，来源方差变化4.0%，对称mask4.3%；极端尺度与完全相同pair均保守不报警。已知效应99.2%，但大来源背景下只4.2%：**size正确不等于有power**。标签单侧mask破坏交换性时100%报警，这检测了测量分布变化，不能当作物理行为误报受控。Wilson区间及每个合成数据集原始统计/均值/p值均保存。无种子、效应强度或带宽搜索。

## 6. 已暴露D上的候选结果与RBR信息边界

每池预固定128个10处理子集，各1024精确方向；所有60来源已是E3方法开发。候选在三个池上RBR与H30的AB均128/128报警，原始条件Δ均0。随机方向伪AA的128次比例各约1.56%–6.25%，这是小规模条件MC检查，不是新的独立FPR资格。**两者在这一探索性候选上都饱和，不能包装成RBR优势、不能推翻D3原方法的H30未饱和结论。** 表现改变来自新null校准，不是旧D3突然有效；不能据此选中或批准方法。

原冻结几何中RBR同源变化/SESSION间距离中位比为cal0.106、AA0.247、AB0.106，干预信号小于来源差异，但非零。60对原始z方向coherence（均值差向量范数/平均差范数）约{t['descriptive_RBR_raw_direction_coherence']:.4f}；全64列多数符号一致率中位数约0.9167，不选择最佳列。条件检验的结果证明“原AB零报警”不等于“所有变化完全消失”；仍不能证明信息来自纯时序，而非轨迹相对位置、邻车或mask等相关变化。

低阶匹配确实很紧：60对mean_speed差中位约−3.28e−8m/s，净降速差约1.31e−6m/s，路径差约−3.93e−7m，物理mean_abs_accel差中位约−4.25e−5m/s²。时间分箱、ACF和制动spread明显变化，spread差中位约0.874s；完整30列物理变化CSV保留。没有把F_match当作完整H30，也没有因H强而删除时序列。

Waymo正式116843窗的有效通道比例约{r['training_mask_summary']['valid_channel_fraction']:.4f}，邻车presence约{r['training_mask_summary']['neighbor_presence_fraction']:.4f}；nuPlan按pool存在上述低mask/稀疏邻车和不同分布。所有83列mask比例与BT mask差异保留。RBR成对标准化距离与BT mask变化Spearman约{t['RBR_distance_correlations_descriptive_only']['mask_BT_change']:.3f}，属于混杂诊断，无因果识别。`U_SOURCE_COMPLETE_INDEPENDENCE_NOT_ESTABLISHED`持续保留；本轮不重新训练或访问Waymo样本，只读生产manifest的既有全量计数。

## 7. 最小前瞻性修约和新增证据

**不直接继续原paired Primary；保留受控ΔBDD问题，修约pair条件null、p值报警及相应SESSION不确定性计划。** 原α=.05、m20、π=.50、δ*=.10、强H30、U-only encoder与全部历史结果不变。旧外部全局阈值不能偷偷替换；新方案需要Owner明确批准后才可能成为正式合同。

防止寻找有利方法：本轮新计算前已落盘analysis_plan，固定单一候选、合成case/seed/预算、全部scaler敏感性网格；原D3不覆盖；包含候选两者都饱和、大背景低power及无交换性失效等负证据；禁止按RBR改善筛参数。后续独立开发必须预先固定fail/stop规则，不再用D3调方法。

新增来源不能给出一个“保证有效”的总数；应区分最小可运行与认证。**可审查的下一次D方案建议**：全新独立60源D_fit稳定性审查＋全新60源pair-null审查＋全新120源AB/power开发，共240源，彼此及旧63D、final-cal/V/C互斥；这是60/60/120支持包络加额外D_fit，不是已获批采集、不是80%power或95%coverage证书。若坚持永久使用冻结3fit，则240中的60仅作尺度外部审查、不改拟合；若要用其重拟合，必须在独立null/AB访问前事前冻结且注明新pipeline身份，仍不改变H30特征。

A/A至少区分：①确定性同Baseline重复是退化p=1，仅验证管线；②有真正可交换采集/仿真随机性的same-condition paired A/A，记录随机化与mask机制；③预注册controlled-label swap，用于核查条件rank，不能替代物理无漂移FPR。A/B必须用新源、原固定SDC、完整分母与原阈值、安全机制，不按报警筛样；比较两表示同一release，并保留饱和/零增量作为合法结果。现有数据仅有每源一个确定性B/T，无法补造①之外的真实repeat-null。

240源本身仍不足以从独立release块精确认证FPR或source CI。举例，按每个独立paired release20个新源、两表示Bonferroni单侧97.5%上界≤.075：48块即960源必须零报警，真FPR=.05时通过概率仅8.53%；500块需10000源，允许≤25报警，通过概率55.3%；1000块需20000源，允许≤58报警，通过概率88.9%（各表示边际计算，不是联合成功率）。**这些是高成本独立块参照而非最低采购要求**。可用更小源池的SESSION推断替代，但必须先独立合成/来源级仿真验证coverage并保留唯一ID支持；不能把重复draw当独立二项数据。AB增量δ功效还依赖实际discordance，候选已暴露结果Δ0不提供正增量依据，当前无法认证所需AB N。未批准任何下载或科学运行。

## 8. 最终研究决策

归因是**统计设计/零假设混用风险＋来源迁移＋预处理估计/近常量数值风险**，不是已证实的MMD算式bug；RBR目标信息完全缺乏尚未建立。

唯一建议：先提交上述最小paired-null修约及其交换性/来源级验证设计，不立即推进Primary、不继续原全局阈值。保持RBR-vs-H30增量效用为可被否定的研究假设；目前既无RBR优势证据，也无足够依据为正结论投入确认实验。候选在已暴露D上两者都饱和提示实际增量可能很小，追加样本只能提高精度，不能创造效应。若前瞻性独立D仍两者饱和/Δ≈0，应如实收口，不弱化H30、不改SDC或重训模型制造差距。

## 9. 执行与复现

新增代码只在tools/rbr_v2_d4_*及对应测试；运行`rbr_v2_d4_analysis.py real`、`synthetic`、`rbr_v2_d4_theory.py`及`rbr_v2_d4_report.py report`。实际real数值诊断墙钟{resource['real']['wall_seconds']:.3f}s、CPU {resource['real']['CPU_seconds']:.3f}s；8000组合成实验墙钟{resource['synthetic']['wall_seconds']:.3f}s、CPU {resource['synthetic']['CPU_seconds']:.3f}s。这些不包括写代码、测试、打包时间。

新车辆传播、模型forward/梯度、训练、下载、Primary/V/C全部0。只核验直接使用的统计/模型源、120个已有输入及60个已有执行manifest，不重复扫描数千历史文件。数学测试、all-tools编译和禁止临时依赖检查有实际日志。计划、输入、源码、结果、图表SHA见Manifest；最终提交包含独立D4命名空间与可复现脚本，现行SAP和D3数组/阈值SHA核对不变。已暴露D只追加方法开发说明，不产生新SESSION、不洗回V/C。
'''
    (DOC/'Scientific_Report_CN_v1.md').write_text(text)
    write(DOC/'Status_v1.json',dict(decision='D4_PROSPECTIVE_SAP_AMENDMENT_REQUIRED',original_D3_unchanged=True,RBR_information_absence='NOT_ESTABLISHED',arithmetic_bug='NOT_EVIDENCED',Primary_ready=False,proposed_method_authorized_for_Primary=False,new_simulation=0,new_training=0,downloads=0,V_C=0))
    text+='''
### 候选修约的推断规格（PROPOSED，未激活）

前瞻性绑定目标仍为相同随机release构造下两表示报警概率差；每release固定20个不同SESSION及确切10个目标Treatment，保留完整失败分母。10个有差异pair可精确枚举1024方向；若真实paired A/A包含20个非同值随机观测对，拟用1999个共同随机方向＋identity，种子及全局数值误差规则须在新数据访问前绑定。合成研究499方向仅为方法压力测试预算，不是偷偷把正式合同1999改为499。

源域区间候选：固定事前独立D_fit及检测管线，按真实独立SESSION/更高依赖cluster分别重采AA与AB的源权重，每次release抽20个不同原始身份、两表示共享抽样/处理mask/翻转符号；每个release重新计算其条件p值，不把外部cal阈值带回来。2000外层/2000内层可沿用SAP的数值预算作为待审提案，但不得在唯一来源不足时忽略失败bootstrap；拟合不确定性与“给定冻结pipeline”的推断目标须区别声明。独立合成来源层级实验仍需验证Δ为0/.10/.20、共享报警discordance、coverage近95%、null拒绝及预定power；本轮仅验证单检验size/power，**未完成增量Δ的source-CI覆盖或原SAP的80%power认证**。原SAP中需要外部cal池的配额、bootstrap重标定步骤与FPR伪null表述必须显式修约，不能只改代码中的一行报警条件。

编译环境说明：nuPlan Python3.9尝试全仓tools编译时，既有`build_unified_bdd_posttraining_report.py:550`含新版f-string语法而失败；未改该历史文件。随后使用现有Smoke Python3.13完成全tools编译，D4新增代码另在nuPlan Python3.9编译通过。原失败日志保留，不将其伪报为通过。
'''
    (DOC/'Scientific_Report_CN_v1.md').write_text(text)
    ids=read(D3/'D_statistics_results.json')['groups'];fit=read(ROOT/'outputs/rbr_v2_d1_v1/D_fit_engineering_scalers.json')['pipelines']['RBR64']['fit_ids']
    exposure=[dict(session_id=s,prior_role=role,D4_role='EXPOSED_METHOD_DEVELOPMENT_ONLY',exposure='E3_PRIMARY_METHOD_DEVELOPMENT',new_scientific_runs=0,future_V_C_for_affected_method=False) for role,ss in {**ids,'D_fit':fit}.items() for s in ss]
    from tools.rbr_v2_d4_analysis import table as csvtable
    csvtable(OUT/'D4_Exposure_Update.csv',exposure)
    print('D4 report and two figures produced; amendment PROPOSED only')


def manifest():
    check();paths=list(OUT.rglob('*'))+list(DOC.rglob('*'))+list((ROOT/'configs/rbr_v2_d4_v1').glob('*'))+list((ROOT/'tools').glob('rbr_v2_d4_*.py'))+list((ROOT/'tests').glob('test_rbr_v2_d4_*.py'))
    target=DOC/'Manifest_v1.json';files={}
    for p in paths:
        if p.is_file() and p!=target and '__pycache__' not in p.parts and p.name!='user_Quick_before.txt' and not p.name.startswith('manifest_packaging'):
            files[str(p.relative_to(ROOT))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    additional=read(OUT/'real_diagnostics.json')['bound_read_only_artifacts']
    for f in ['outputs/rbr_v2_d1_v1/input_manifest.json','outputs/rbr_v2_d1_v1/RBR64.npy','outputs/rbr_v2_d1_v1/H30.npy','outputs/rbr_v2_d3_v1/paired_calibration_transfer_audit.json','tools/s1_protocol_schema.py','tools/stage6l_prepare_context_representation_ablation.py']:
        additional[f]=sha(ROOT/f)
    write(target,dict(status='D4_PROSPECTIVE_SAP_AMENDMENT_REQUIRED',files=files,protected_statistical_sources=read(PLAN)['protected_sources'],additional_real_read_sources=additional,starting_commit=read(PLAN)['start_commit'],new_vehicle_simulation=0,new_model_inference=0,Primary_V_C=0,SAP_mutated=False))
    print('D4 provenance:',len(files),'owned files; targeted source protection checked')


def verify():
    check();m=read(DOC/'Manifest_v1.json');bad=[]
    for f,v in m['files'].items():
        if not (ROOT/f).exists() or sha(ROOT/f)!=v['sha256']:bad.append(f)
    for f,h in m['additional_real_read_sources'].items():
        if sha(ROOT/f)!=h:bad.append(f)
    if bad:raise ValueError('D4 provenance mismatch: '+repr(bad))
    print('D4 verification PASS:',len(m['files']),'owned hashes; directly used historical inputs/statistics/model unchanged; no SAP replacement')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['report','manifest','verify']);args=a.parse_args();{'report':report,'manifest':manifest,'verify':verify}[args.mode]()
