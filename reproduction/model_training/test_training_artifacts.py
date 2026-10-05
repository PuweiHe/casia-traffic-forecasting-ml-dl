import json
import tempfile
import unittest
from pathlib import Path
import torch
from validate_training_artifacts import validate

class ArtifactTests(unittest.TestCase):
    def write_fixture(self,root):
        history=[{'epoch':1}];weights={'weight':torch.ones(2)}
        result={'identity':'fixture','trial':'fixture','seed':17,'selected_epoch':1,'completed_epochs':1,'validation':{'mae':1.}}
        for name,data in [('result.json',result),('status.json',dict(result,state='complete')),('epochs.json',history)]:
            (root/name).write_text(json.dumps(data))
        torch.save({'identity':'fixture','trial':{'id':'fixture'},'seed':17,'epoch':1,'validation':result['validation'],'state_dict':weights},root/'selected.pt')
        torch.save({'identity':'fixture','history':history,'best':{'epoch':1,'state_dict':weights}},root/'progress.pt')
    def test_complete_and_missing_weights(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.write_fixture(root);self.assertEqual(validate(root)['state'],'verified_complete')
            (root/'selected.pt').unlink()
            with self.assertRaisesRegex(ValueError,'Missing selected'):validate(root)
    def test_result_before_completion_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.write_fixture(root);p=root/'status.json';d=json.loads(p.read_text());d['state']='training';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'complete status'):validate(root)
    def test_corrupt_selected_weights_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.write_fixture(root);p=root/'selected.pt';d=torch.load(p,weights_only=True);d['state_dict']['weight'].zero_();torch.save(d,p)
            with self.assertRaisesRegex(ValueError,'weight mismatch'):validate(root)

if __name__=='__main__':unittest.main()
