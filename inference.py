"""Inference from audited JSON artifacts; no student database is required."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent


def missing(value):
    return value is None or (isinstance(value, (float, np.floating)) and np.isnan(value))


def finite_number(value, label, minimum=0, maximum=None, integer=False, optional=False):
    if missing(value):
        if optional:
            return None
        raise ValueError(f'{label}: valor obrigatorio.')
    if isinstance(value, bool):
        raise ValueError(f'{label}: valor numerico invalido.')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label}: valor numerico invalido.') from None
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{label}: valor fora do intervalo permitido.')
    if integer and not value.is_integer():
        raise ValueError(f'{label}: use um numero inteiro.')
    return int(value) if integer else value


def build_features(profile, schema):
    cats = schema['categorical_columns']
    for name, categories in zip(cats, schema['categories']):
        value = profile.get(name)
        if not missing(value) and value not in categories:
            raise ValueError(f'Categoria invalida: {name}.')
    n = finite_number(profile.get('disciplinas_avaliadas'), 'Disciplinas avaliadas', 1, 100, integer=True)
    failed = finite_number(profile.get('numero_reprovacoes'), 'Reprovacoes', 0, n, integer=True)
    failed_absence = finite_number(profile.get('reprovacoes_faltas'), 'Reprovacoes por faltas', 0, failed, integer=True)
    age = finite_number(profile.get('idade_ingresso'), 'Idade', 14, 90, integer=True, optional=True)
    income = finite_number(profile.get('renda'), 'Renda', 0, 1e9, optional=True)
    grades = finite_number(profile.get('media_notas'), 'Media das notas', 0, 10, optional=True)
    absences = finite_number(profile.get('media_faltas'), 'Media das faltas', 0, 1e5, optional=True)
    row = {c: profile.get(c) for c in cats}
    row.update(idade_ingresso=age, log_renda=None if income is None else math.log1p(income),
               disciplinas_avaliadas=n, numero_reprovacoes=failed, proporcao_reprovacoes=failed/n,
               media_notas=grades, media_faltas=absences, proporcao_reprovacoes_faltas=failed_absence/n)
    return row


class Predictor:
    def __init__(self, folder=None):
        folder = Path(folder) if folder else ROOT / 'model'
        self.metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        for name, expected in self.metadata['artifact_sha256'].items():
            if name not in {'xgboost.json', 'preprocessing.json'}:
                raise ValueError('Artefato nao reconhecido.')
            if hashlib.sha256((folder/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Artefato alterado; conferir exportacao.')
        self.schema = json.loads((folder/'preprocessing.json').read_text(encoding='utf-8'))
        self.booster = xgb.Booster(params={'nthread': 1})
        self.booster.load_model(folder/'xgboost.json')

    def transform(self, rows):
        s = self.schema
        columns = []
        for name, values in zip(s['categorical_columns'], s['categories']):
            column = [s['categorical_missing'] if missing(row.get(name)) else row[name] for row in rows]
            columns.extend([[float(value == category) for value in column] for category in values])
        for name, median, mean, scale in zip(s['numeric_columns'], s['numeric_medians'], s['numeric_means'], s['numeric_scales']):
            columns.append([((median if missing(row.get(name)) else float(row[name])) - mean)/scale for row in rows])
        matrix = np.asarray(columns, dtype=np.float64).T
        if matrix.shape[1] != self.booster.num_features() or not np.isfinite(matrix).all():
            raise ValueError('Pre-processamento invalido.')
        return matrix

    def predict_rows(self, rows):
        matrix = self.transform(rows)
        result = self.booster.inplace_predict(matrix)
        if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
            raise ValueError('Predicao invalida.')
        return result

    def predict(self, profile):
        row = build_features(profile, self.schema)
        p = float(self.predict_rows([row])[0])
        absent = [name for name, value in row.items() if missing(value)]
        outside = []
        for name, limits in self.metadata['training_ranges'].items():
            v = row.get(name)
            if not missing(v) and not limits[0] <= v <= limits[1]:
                outside.append(name)
        return dict(probability=p, missing=absent, outside_training=outside)
