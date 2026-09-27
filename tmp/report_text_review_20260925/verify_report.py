from pathlib import Path
import hashlib, json, re
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pymupdf

root=Path('Final_Report_Research')
qa=Path('tmp/report_text_review_20260925')
hashes=json.loads((qa/'image_hashes.json').read_text(encoding='utf-8-sig'))
for row in hashes:
    assert hashlib.sha256(Path(row['Path']).read_bytes()).hexdigest().upper()==row['Hash']

figs=0
refs=[]
labels=[]
cites=[]
changed=[]
for old in (qa/'content').rglob('*.tex'):
    p=root/'content'/old.relative_to(qa/'content')
    before=old.read_text(encoding='utf-8')
    after=p.read_text(encoding='utf-8')
    for pat in [r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}',
                r'\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}']:
        a=re.findall(pat,before,re.S); b=re.findall(pat,after,re.S)
        assert a==b, f'Figure changed: {p}'
        figs+=len(a)
    if before!=after: changed.append(str(p))
    refs += re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}',after)
    labels += re.findall(r'\\label\{([^}]+)\}',after)
    cites += [c.strip() for group in re.findall(r'\\cite\{([^}]+)\}',after) for c in group.split(',')]
assert not (set(refs)-set(labels)), set(refs)-set(labels)
bibkeys=set(re.findall(r'@\w+\{([^,]+),',(root/'bib/library.bib').read_text(encoding='utf-8')))
assert not (set(cites)-bibkeys)

results={}
for prefix, predname in [('corrected_temporal_holdout','corrected_test_predictions'),('corrected_external_incart','corrected_external_predictions')]:
    d=json.loads((root/'experiments'/f'{prefix}.json').read_text())
    a=np.load(root/'experiments'/f'{predname}.npz')
    supported=a['truth_class']>=0
    cm=confusion_matrix(a['truth_class'][supported],a['pred_system'][supported],labels=np.arange(6))
    assert cm.tolist()==d['test_whole_system']['confusion_matrix']
    assert np.isclose(accuracy_score(a['truth_binary'],a['pred_binary']),d['test_binary']['accuracy'])
    assert np.isclose(accuracy_score(a['truth_class'][supported],a['pred_system'][supported]),d['test_whole_system']['accuracy'])
    assert np.isclose(f1_score(a['truth_class'][supported],a['pred_system'][supported],labels=np.arange(6),average='macro',zero_division=0),d['test_whole_system']['macro_f1'])
    results[prefix]='confusion matrix, accuracy and macro F1 match saved predictions'
    if prefix=='corrected_temporal_holdout':
        records=a['records'].astype('<U7')
        records[np.isin(records,['201','202'])]='201-202'
        scores=np.array([np.mean(a['truth_class'][(records==r)&supported]==a['pred_system'][(records==r)&supported]) for r in np.unique(records)])
        replicates=scores[np.random.default_rng(20260801).integers(0,len(scores),(10000,len(scores)))].mean(axis=1)
        np.testing.assert_allclose(np.percentile(replicates,[2.5,97.5]),d['headline']['whole_system_subject_clustered_ci_95'])
        results['bootstrap']='10,000-replicate equal-subject interval reproduced'

doc=pymupdf.open(root/'main.pdf')
texts=[p.get_text() for p in doc]
(qa/'final_text.txt').write_text('\n\f\n'.join(texts),encoding='utf-8')
assert not any('??' in t for t in texts), 'Unresolved reference in PDF'
log=(root/'main.log').read_text(errors='replace')
assert 'Overfull' not in log
assert 'undefined' not in log.lower()
chapter_pages=[]
for i,t in enumerate(texts):
    if any(f'{j}\n' in t[:3] for j in range(1,9)) or t.startswith('Abstract') or 'Reproducibility Record' in t[:90]:
        chapter_pages.append([i+1,t.splitlines()[:3]])
report={'pages':len(doc),'changed_text_files':changed,'image_files_unchanged':len(hashes),'figure_inclusions_and_drawings_unchanged':figs,'references_resolved':True,'checks':results,'chapter_pages':chapter_pages}
(qa/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2,ensure_ascii=True))
