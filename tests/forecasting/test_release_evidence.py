"""Published claims must agree with the immutable aggregate source files."""
import hashlib,json,statistics,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
E=ROOT/'docs/forecasting/evidence/final_study'
class ReleaseEvidenceTests(unittest.TestCase):
    def test_manifest_and_locked_metrics(self):
        m=json.loads((E/'PUBLIC_MANIFEST.json').read_text())
        for name,digest in m['files'].items():
            self.assertEqual(hashlib.sha256((E/name).read_bytes()).hexdigest(),digest,name)
        selection=hashlib.sha256((E/'pems_test/locked_selection.json').read_bytes()).hexdigest()
        summary=json.loads((E/'final_pems/summary.json').read_text())
        for trial in ['stgcn_reference','stgcn_search_005']:
            values=[]
            for seed in [17,42,73]:
                r=json.loads((E/'pems_test'/f'{trial}_seed{seed}.json').read_text())
                self.assertEqual(r['selection_sha256'],selection)
                mae=statistics.mean(h['mae_mph'] for h in r['test']['all']['horizons'].values())
                self.assertAlmostEqual(mae,r['primary_mae_mph'],places=12);values.append(mae)
            self.assertAlmostEqual(statistics.mean(values),summary['models'][trial]['test_mae_mean_mph'],places=12)
        a=summary['models']['stgcn_reference']['test_mae_mean_mph'];b=summary['models']['stgcn_search_005']['test_mae_mean_mph']
        self.assertAlmostEqual(100*(1-b/a),summary['relative_mae_reduction_percent'],places=12)
    def test_released_weights_and_source_identity(self):
        from traffic_forecasting.stgcn_inference import Predictor
        for trial in ['stgcn_reference','stgcn_search_005']:
            artifact=ROOT/'artifacts/forecasting'/f'pems_{trial}_seed42.pt'
            verified=json.loads((E/'serving'/f'{trial}.json').read_text())
            self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(),verified['artifact_sha256'])
            engine=Predictor(artifact,'cpu')
            self.assertEqual(engine.artifact['seed'],42)
            self.assertEqual(engine.artifact['trial']['id'],trial)
if __name__=='__main__':unittest.main()
