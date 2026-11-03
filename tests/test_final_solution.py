"""CPU checks for final-solution data routing; no model forward or annotation API."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import torch
from rsna_knee.final_preparation import TARGETS, mix_sources, read_targets
from rsna_knee.final_data import export_selection, blend_predictions
from rsna_knee.input_pipeline import KneeStudyDataset


class FinalSolutionTests(unittest.TestCase):
    def test_mix5_joins_sources_by_id(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            names = ('astra_v1','astra_v2','teammate_astra','hulu','qwen_next')
            for i,name in enumerate(names):
                with (root/f'{name}_weak4349.csv').open('w',newline='') as f:
                    w = csv.writer(f);w.writerow(['StudyInstanceUID',*TARGETS])
                    order = range(4349) if i % 2 else reversed(range(4349))
                    for u in order:w.writerow([str(u), *([i/10]*12)])
            result = mix_sources(root)
            self.assertEqual(len(result),4349)
            self.assertEqual(result['0'],[.15]*12)
            with (root/'astra_v1_weak4349.csv').open('a') as f:
                f.write('0,'+','.join(['.5']*12)+'\n')
            with self.assertRaises(ValueError):read_targets(root/'astra_v1_weak4349.csv')

    def test_numeric_sample_requires_no_reports_or_text(self):
        dataset = object.__new__(KneeStudyDataset)
        dataset.rows = [{'Sample ID':'study', 'fold':-1, **dict(zip(TARGETS,[.2]*12))}]
        dataset.inference_only = False;dataset.numeric_targets = True;dataset.return_stages = False
        derived = dict(model_input={'images':torch.zeros(1)}, provenance={'roles':[]}, review_stages={})
        with patch.object(dataset,'build_study_input',return_value=object()), patch.object(dataset,'derive_model_view',return_value=derived):
            dataset.default_view = object()
            result = dataset[0]
        self.assertEqual(set(result['targets']),{'labels','label_mask','is_hard'})
        self.assertTrue(torch.equal(result['targets']['labels'],torch.full((12,),.2)))
        self.assertNotIn('report',result['provenance'])
        self.assertFalse(result['targets']['is_hard'])

    def test_final_team_blend_aligns_ids(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for j, probability in enumerate((.1,.3,.5,.9)):
                path = Path(d)/f'{j}.csv';paths.append(path)
                with path.open('w',newline='') as f:
                    w=csv.writer(f);w.writerow(['StudyInstanceUID',*TARGETS])
                    for u in (['a','b'] if j%2 else ['b','a']):w.writerow([u,*([probability]*12)])
            output=Path(d)/'team.csv'
            blend_predictions(paths[:3],output,paths[3])
            self.assertEqual(read_targets(output),{'b':[.6]*12,'a':[.6]*12})
            paths[3].write_text(paths[3].read_text().replace('b,','other,'))
            with self.assertRaises(ValueError):blend_predictions(paths[:3],Path(d)/'bad.csv',paths[3])

    def test_selection_export_removes_supervision(self):
        with tempfile.TemporaryDirectory() as d:
            source, target = Path(d)/'source.jsonl',Path(d)/'images.jsonl'
            with source.open('w') as f:
                for i in range(4407):
                    f.write(json.dumps(dict(study_uid=str(i),selected_series={},series_candidates=[],Report='private',hard_targets=[1]))+'\n')
            export_selection(source,target)
            rows = [json.loads(line) for line in target.read_text().splitlines()]
            self.assertEqual(len(rows),4407)
            self.assertEqual(set(rows[0]),{'study_uid','selected_series','series_candidates'})
            with self.assertRaises(FileExistsError):export_selection(source,target)

    def test_series_flags_stay_numeric(self):
        # Regression: reading every column as str made `Fluid_Sensitive == 1` false and
        # silently dropped all three fluid-sensitive roles at inference.
        from rsna_knee.final_inference import read_series_rows
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'test_series.csv'
            path.write_text('StudyInstanceUID,SeriesInstanceUID,Fluid_Sensitive,Fat_Suppression,Anatomical_Plane\n'
                            '1.2.3,1.2.3.4,1,1,Sagittal\n1.2.3,1.2.3.5,0,0,Coronal\n')
            rows = read_series_rows(path)['1.2.3']
            self.assertEqual([r['Fluid_Sensitive'] == 1 for r in rows], [True, False])
            self.assertEqual([r['SeriesInstanceUID'] for r in rows], ['1.2.3.4', '1.2.3.5'])


if __name__ == '__main__':unittest.main()
