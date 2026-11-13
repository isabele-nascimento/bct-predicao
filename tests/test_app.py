from pathlib import Path
import json
import unittest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()

    def test_start_and_missing_required(self):
        app=self.app()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(app.metric[0].value,'—')
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertGreater(len(app.error),0)
        self.assertNotIn('result',app.session_state)

    def test_valid_profile_and_reset(self):
        app=self.app()
        fixture=json.loads((ROOT/'tests/synthetic_cases.json').read_text())[0]
        profile=fixture['profile']
        for key in ['sexo','raca','cotista','forma_ingresso','escola']:
            app.selectbox(key=key).set_value(profile[key])
        for key in ['idade_ingresso','renda','disciplinas_avaliadas','numero_reprovacoes','reprovacoes_faltas','media_notas','media_faltas']:
            app.number_input(key=key).set_value(profile[key])
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertAlmostEqual(app.session_state['result']['probability'],fixture['probability'],places=6)
        self.assertEqual(app.metric[0].value,f'{100*fixture["probability"]:.1f}%'.replace('.',','))
        app.button(key='clear').click().run()
        self.assertEqual(len(app.exception),0)
        self.assertNotIn('result',app.session_state)
        self.assertEqual(app.metric[0].value,'—')

    def test_invalid_profile_clears_result(self):
        app=self.app()
        for key,value in [('disciplinas_avaliadas',6),('numero_reprovacoes',2),('reprovacoes_faltas',0)]:
            app.number_input(key=key).set_value(value)
        app.button[0].click().run()
        self.assertIn('result',app.session_state)
        app.number_input(key='numero_reprovacoes').set_value(7)
        app.button[0].click().run()
        self.assertNotIn('result',app.session_state)
        self.assertGreater(len(app.error),0)
        self.assertEqual(len(app.exception),0)


if __name__=='__main__':
    unittest.main()
