"""Offline contract tests. Synthetic fixtures are not scientific naming results."""
import argparse
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from PIL import Image
import name_concepts as naming


class NamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.a, self.b = self.root / 'before.png', self.root / 'after.png'
        Image.new('RGB', (32, 32), 'gray').save(self.a)
        Image.new('RGB', (32, 32), 'white').save(self.b)
        self.job = {'concept_id': 'test_000', 'dataset_context': 'Synthetic fixture.',
                    'pairs': [{'before': self.a, 'after': self.b}]}
        self.config = dict(naming.DEFAULTS)
        self.result = {k: 'synthetic test' for k in naming.TEXT_FIELDS}
        self.result.update({k: [] for k in naming.LIST_FIELDS})
        self.result.update(concept_id='test_000', confidence='low', best_name='synthetic fixture')
        self.raw = {'status': 'completed', 'model': naming.DEFAULT_MODEL, 'id': 'test_response',
                    'output': [{'type': 'message', 'content': [{'type': 'output_text',
                        'text': json.dumps(self.result)}]}], 'usage': {'total_tokens': 0}}

    def args(self, **kw):
        vals = dict(config=None, manifest=None, before=self.a, after=self.b,
                    concept_id='test_000', dataset='cub', dataset_context=None,
                    predicted_class='', output=self.root / 'out', dry_run=False, resume=False)
        vals.update(kw)
        return argparse.Namespace(**vals)

    def test_pair_order_and_prompt(self):
        payload, metadata = naming.build_request(self.job, self.config)
        content = payload['input'][0]['content']
        images = [c for c in content if c['type'] == 'input_image']
        self.assertEqual(len(images), 2)
        self.assertNotEqual(images[0]['image_url'], images[1]['image_url'])
        self.assertIn('Image A', content[1]['text'])
        self.assertIn('Image B', content[3]['text'])
        self.assertIn('"concept_id": "test_000"', content[0]['text'])
        self.assertEqual(payload['reasoning'], {'effort': 'high'})
        self.assertNotIn('temperature', payload)
        self.assertNotIn('api_key', json.dumps(metadata))
        self.job['pairs'][0] = {'before': self.b, 'after': self.a}
        self.assertNotEqual(metadata['request_sha256'], naming.build_request(self.job, self.config)[1]['request_sha256'])

    def test_multisample_instruction(self):
        self.job['pairs'].append({'before': self.a, 'after': self.b})
        payload, metadata = naming.build_request(self.job, self.config)
        self.assertEqual(metadata['pair_count'], 2)
        self.assertIn('different samples of the same concept', payload['input'][0]['content'][1]['text'])
        self.assertEqual(sum(x['type'] == 'input_image' for x in payload['input'][0]['content']), 4)

    def test_mismatched_images_rejected(self):
        Image.new('RGB', (31, 32), 'gray').save(self.b)
        with self.assertRaises(naming.NamingError):
            naming.build_request(self.job, self.config)

    def test_manifest_paths_and_missing_metadata(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps([{'concept_id': 'test_000', 'dataset_context': 'Test birds.',
            'pairs': [{'before': 'before.png', 'after': 'after.png'}]}]))
        jobs = naming.load_jobs(self.args(manifest=path, before=None, after=None, concept_id=None, dataset=None))
        self.assertEqual(jobs[0]['pairs'][0]['before'], self.a.resolve())
        with self.assertRaises(naming.NamingError):
            naming.load_jobs(self.args(dataset=None))

    def test_response_contract_and_refusal(self):
        self.assertEqual(naming.parse_response(self.raw, 'test_000')['confidence'], 'low')
        for bad in [dict(self.raw, status='incomplete'),
                    {'status': 'completed', 'output': [{'content': [{'type': 'refusal'}]}]}]:
            with self.assertRaises(naming.NamingError):
                naming.parse_response(bad, 'test_000')
        with self.assertRaises(naming.NamingError):
            naming.parse_response(self.raw, 'wrong_id')

    def test_official_http_contract(self):
        response = io.BytesIO(json.dumps(self.raw).encode())
        response.headers = {'x-request-id': 'test_request'}
        with patch.object(naming.urllib.request, 'urlopen', return_value=response) as mock:
            body, request_id = naming.call_openai({'model': naming.DEFAULT_MODEL}, 'test-key', self.config)
        req = mock.call_args.args[0]
        self.assertEqual(req.full_url, naming.ENDPOINT)
        self.assertEqual(req.get_method(), 'POST')
        self.assertEqual(req.get_header('Authorization'), 'Bearer test-key')
        self.assertEqual(request_id, 'test_request')
        self.assertEqual(body['id'], 'test_response')

    def test_retry_429_and_no_retry_401(self):
        failed = urllib.error.HTTPError(naming.ENDPOINT, 429, 'limited', {}, None)
        response = io.BytesIO(json.dumps(self.raw).encode())
        response.headers = {}
        with patch.object(naming.urllib.request, 'urlopen', side_effect=[failed, response]) as mock, patch.object(naming.time, 'sleep'):
            naming.call_openai({}, 'test-key', self.config)
            self.assertEqual(mock.call_count, 2)
        failed = urllib.error.HTTPError(naming.ENDPOINT, 401, 'unauthorized', {}, None)
        with patch.object(naming.urllib.request, 'urlopen', side_effect=failed) as mock:
            with self.assertRaises(naming.NamingError):
                naming.call_openai({}, 'test-key', self.config)
            self.assertEqual(mock.call_count, 1)

    def test_output_resume_and_key_exclusion(self):
        with patch.object(naming, 'load_config', return_value=self.config), patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key-not-for-publication'}), patch.object(naming, 'call_openai', return_value=(self.raw, 'test_request')) as mock:
            naming.run(self.args())
            naming.run(self.args(resume=True))
            self.assertEqual(mock.call_count, 1)
        self.assertTrue((self.root / 'out' / 'names.csv').is_file())
        for path in (self.root / 'out').rglob('*'):
            if path.is_file():
                self.assertNotIn('test-key-not-for-publication', path.read_text())

    def test_dry_run_never_sends(self):
        with patch.object(naming, 'load_config', return_value=self.config), patch.object(naming, 'call_openai') as mock:
            naming.run(self.args(dry_run=True))
            mock.assert_not_called()
        self.assertEqual(len(list((self.root / 'out').rglob('request_metadata.json'))), 1)
        self.assertEqual(len(list((self.root / 'out').rglob('result.json'))), 0)


if __name__ == '__main__':
    unittest.main()
