from pathlib import Path
import copy
import json
import tempfile
import unittest
import shutil
import math
from inference import Predictor, build_features

ROOT=Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor=Predictor()
        cls.cases=json.loads((ROOT/'tests/synthetic_cases.json').read_text())

    def test_synthetic_parity_with_notebook(self):
        for case in self.cases:
            actual=self.predictor.predict(case['profile'])['probability']
            self.assertAlmostEqual(actual,case['probability'],places=6)

    def test_proportions_derived_from_counts(self):
        row=build_features(self.cases[0]['profile'],self.predictor.schema)
        self.assertAlmostEqual(row['proporcao_reprovacoes'],1/6)
        self.assertEqual(row['proporcao_reprovacoes_faltas'],0)
        self.assertAlmostEqual(row['log_renda'],math.log1p(1500))

    def test_invalid_values_rejected(self):
        for field,value in [('disciplinas_avaliadas',0),('disciplinas_avaliadas',None),
            ('numero_reprovacoes',7),('numero_reprovacoes',1.3),('reprovacoes_faltas',2),
            ('media_notas',11),('media_faltas',-1),('renda',float('inf')),('renda',-1),
            ('idade_ingresso',13),('idade_ingresso',True),('sexo','OUTRA_CATEGORIA')]:
            with self.subTest(field=field,value=value):
                profile=copy.deepcopy(self.cases[0]['profile'])
                profile[field]=value
                with self.assertRaises(ValueError):
                    self.predictor.predict(profile)

    def test_optional_missing_values(self):
        result=self.predictor.predict(self.cases[2]['profile'])
        self.assertIn('idade_ingresso',result['missing'])
        self.assertIn('log_renda',result['missing'])
        self.assertTrue(0<=result['probability']<=1)

    def test_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name in ['metadata.json','preprocessing.json','xgboost.json']:
                shutil.copyfile(ROOT/'model'/name,folder/name)
            with (folder/'preprocessing.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaises(ValueError):
                Predictor(folder)

    def test_portable_newlines(self):
        for name in ['metadata.json','preprocessing.json','verification.json']:
            self.assertNotIn(b'\r\n',(ROOT/'model'/name).read_bytes())

    def test_only_aggregate_model_artifacts(self):
        self.assertEqual(set(p.name for p in (ROOT/'model').iterdir()),
            {'metadata.json','preprocessing.json','xgboost.json','verification.json'})
        verification=json.loads((ROOT/'model/verification.json').read_text())
        self.assertEqual(verification['student_records_exported'],0)
        self.assertEqual(verification['n_private_rows_checked'],6685)


if __name__=='__main__':
    unittest.main()
