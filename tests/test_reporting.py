from pathlib import Path
from math import sqrt
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'/'paper_reported'

def wilson(k,n,z=1.959963984540054):
    p=k/n; den=1+z*z/n
    c=(p+z*z/(2*n))/den
    h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,c-h,c+h

def test_wilson_codebert():
    d=pd.read_csv(R/'fixed_test_accuracy_wilson95.csv').set_index('model')
    p,lo,hi=wilson(5651,5654)
    assert abs(d.loc['CodeBERT','accuracy']-p)<1e-12
    assert abs(d.loc['CodeBERT','wilson95_low']-lo)<1e-12
    assert abs(d.loc['CodeBERT','wilson95_high']-hi)<1e-12

def test_error_overlap_arithmetic():
    r=pd.read_csv(R/'error_overlap_summary.csv').iloc[0]
    assert int(r.shared_errors)==1
    assert int(r.codebert_only_errors)==2
    assert int(r.char_linearsvc_only_errors)==8
    assert int(r.error_union)==11
    assert abs(float(r.error_set_jaccard)-1/11)<1e-12

def test_mcnemar_raw_p_recomputes_from_discordants():
    d=pd.read_csv(R/'exact_mcnemar_FINAL.csv')
    for _,r in d.iterrows():
        b=int(r.model1_wrong_model2_correct); c=int(r.model1_correct_model2_wrong)
        p=binomtest(min(b,c),n=b+c,p=.5,alternative='two-sided').pvalue
        assert abs(p-float(r.exact_p_value))<1e-12
