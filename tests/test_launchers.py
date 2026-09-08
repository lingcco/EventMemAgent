"""Check public shell entry points without loading model weights or using GPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='eventmemagent-test-')
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.bin = self.directory / 'bin'
        self.bin.mkdir()
        self.capture = self.directory / 'calls.jsonl'
        # A process-level recorder also intercepts `exec python` in service launchers.
        (self.bin / 'python').write_text(
            f'#!{sys.executable}\nimport json, os, sys\n'
            'with open(os.environ["LAUNCHER_CAPTURE"], "a") as out:\n'
            '    out.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd()}) + "\\n")\n'
        )
        (self.bin / 'python').chmod(0o755)
        (self.bin / 'curl').write_text('#!/bin/sh\nexit 0\n')
        (self.bin / 'curl').chmod(0o755)
        self.model = self.directory / 'model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}')
        self.data = self.directory / 'input.json'
        self.data.write_text('[]')
        self.actor = self.directory / 'actor'
        (self.actor / 'huggingface').mkdir(parents=True)
        self.env = {
            **os.environ,
            'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
            'LAUNCHER_CAPTURE': str(self.capture),
            'CUDA_VISIBLE_DEVICES': '0', 'NUM_GPUS': '1', 'TENSOR_PARALLEL_SIZE': '1',
            'RESUME_MODE': 'disable', 'RESUME_FROM_PATH': '', 'UPLOAD_TO_HF': '0',
            'MODEL_PATH': str(self.model), 'BASE_MODEL_PATH': str(self.model),
            'EMBEDDING_MODEL_PATH': str(self.model), 'OCR_MODEL_PATH': str(self.model),
            'DETECTION_MODEL_PATH': str(self.model), 'MEMORY_CAPTION_MODEL_PATH': str(self.model),
            'TOOL_CONFIG_PATH': str(ROOT / 'configs/tools.yaml'),
            'TRAIN_DATA': str(self.data), 'VAL_DATA': str(self.data),
            'PARQUET_PATH': str(self.data), 'OVOBENCH_DATA': str(self.data),
            'OVOBENCH_METADATA': str(self.data), 'STREAMINGBENCH_DATA': str(self.data),
            'STREAMINGBENCH_PARQUET': str(self.data),
            'VIDEO_PATH': str(self.data), 'QUESTION': 'What happened?',
            'INPUT_JSON': str(self.data), 'VIDEO_DIR': str(self.directory),
            'OUTPUT_DIR': str(self.directory / 'output'),
            'OUTPUT_FILE': str(self.directory / 'output.json'),
            'EVAL_OUTPUT_ROOT': str(self.directory / 'eval'),
            'TRAIN_OUTPUT_DIR': str(self.directory / 'train'),
            'SKIP_TOOL_SERVER_CHECK': '1', 'INPUT_MODE': 'parquet',
            'MODE': 'batch', 'EVENTMEMAGENT_BENCHMARK': 'ovobench',
        }

    def launch(self, entry, *args, **overrides):
        result = subprocess.run(['bash', str(ROOT / 'scripts' / entry), *args],
                                cwd=self.directory, env={**self.env, **overrides},
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.capture.read_text().splitlines()]
        self.assertTrue(calls)
        self.assertTrue(all(call['cwd'] == str(ROOT) for call in calls))
        return calls[-1]['argv']

    def test_training_entry(self):
        args = self.launch('train/train.sh', 'trainer.total_epochs=2')
        self.assertEqual(args[:2], ['-m', 'verl.trainer.main_ppo'])
        self.assertIn('--config-path=' + str(ROOT / 'configs'), args)
        self.assertIn('trainer.total_epochs=2', args)

    def test_checkpoint_export(self):
        args = self.launch('train/merge_checkpoint.sh', str(self.actor), str(self.directory / 'merged'))
        self.assertEqual(args[:3], ['-m', 'verl.model_merger', 'merge'])
        self.assertNotIn('--hf_upload_path', args)

    def test_training_without_validation_needs_no_extra_file(self):
        args = self.launch('train/train.sh', '--no-validation', VAL_DATA='/not/a/validation/file')
        for value in ('trainer.val_before_train=false', 'trainer.test_freq=-1',
                      'trainer.val_only=false', 'data.val_files=[]'):
            self.assertIn(value, args)

    def test_training_without_validation_by_default(self):
        self.launch('train/train.sh', VAL_DATA='')

    def test_no_validation_can_be_dry_run(self):
        args = self.launch('train/train.sh', '--no-validation', '--dry-run')
        self.assertEqual(args[0], 'scripts/utils/print_train_config.py')
        self.assertIn('trainer.test_freq=-1', args)

    def test_no_validation_rejects_conflicting_overrides(self):
        result = subprocess.run(
            ['bash', str(ROOT / 'scripts/train/train.sh'), '--no-validation', 'trainer.test_freq=1'],
            env=self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('Do not combine', result.stderr)

    def test_offline_inference(self):
        args = self.launch('infer/offline.sh', '7')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.inference.run'])
        self.assertEqual(args[args.index('--sample_idx') + 1], '7')

    def test_online_inference(self):
        args = self.launch('infer/online.sh')
        self.assertEqual(args[args.index('--mode') + 1], 'online')

    def test_ovo_batch(self):
        args = self.launch('eval/ovobench.sh')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.evaluation.batch'])

    def test_ovo_normal(self):
        args = self.launch('eval/ovobench.sh', MODE='normal')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.evaluation.run'])

    def test_streaming_parquet_delegation(self):
        args = self.launch('eval/streamingbench.sh')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.evaluation.batch'])
        self.assertEqual(args[args.index('--parquet_path') + 1], str(self.data))

    def test_streaming_online(self):
        args = self.launch('eval/streamingbench.sh', INPUT_MODE='online')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.evaluation.streamingbench'])

    def test_scoring(self):
        for name in ('ovobench', 'streamingbench'):
            with self.subTest(benchmark=name):
                args = self.launch(f'eval/score_{name}.sh', str(self.data))
                self.assertEqual(args[:2], ['-m', f'eventmemagent.evaluation.score_{name}'])

    def test_preprocessing(self):
        args = self.launch('data/preprocess.sh')
        self.assertEqual(args[:2], ['-m', 'eventmemagent.memory.preprocess'])

    def test_services(self):
        modules = {'ocr': 'eventmemagent.services.ocr_server',
                   'detection': 'eventmemagent.services.detection_server',
                   'memory': 'vllm.entrypoints.openai.api_server'}
        for service, module in modules.items():
            with self.subTest(service=service):
                args = self.launch(f'services/{service}.sh')
                self.assertEqual(args[:2], ['-m', module])


if __name__ == '__main__':
    unittest.main()
