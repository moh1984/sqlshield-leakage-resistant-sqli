#!/usr/bin/env python3
"""Recompute 95% Wilson score confidence intervals for fixed-test accuracy."""
from pathlib import Path
from math import sqrt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results'/'paper_reported'/'fixed_test_seed42_FINAL.csv'
OUT=ROOT/'results'/'paper_reported'/'fixed_test_accuracy_wilson95.csv'

def wilson(k,n,z=1.959963984540054):
    p=k/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return p,center-half,center+half

def main():
    d=pd.read_csv(SRC)
    rows=[]
    for _,r in d.iterrows():
        n=int(r.n); correct=n-int(r.errors)
        p,lo,hi=wilson(correct,n)
        rows.append({'model':r.model,'n':n,'correct':correct,'errors':int(r.errors),
                     'accuracy':p,'wilson95_low':lo,'wilson95_high':hi,
                     'accuracy_pct':100*p,'wilson95_low_pct':100*lo,'wilson95_high_pct':100*hi})
    out=pd.DataFrame(rows)
    out.to_csv(OUT,index=False)
    print(out[['model','accuracy_pct','wilson95_low_pct','wilson95_high_pct']].to_string(index=False,
          formatters={c:lambda x:f'{x:.4f}' for c in ['accuracy_pct','wilson95_low_pct','wilson95_high_pct']}))
    print(f'\nSaved: {OUT}')

if __name__=='__main__': main()
