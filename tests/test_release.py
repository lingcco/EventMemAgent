"""CPU regression checks for the public entry points and paper settings."""
import importlib.util
import ast
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_module(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PaperSettingsTest(unittest.TestCase):
    def test_gitignore_keeps_training_dataset_implementation(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(['git', 'init', '-q', directory], check=True)
            (Path(directory) / '.gitignore').write_text((ROOT / '.gitignore').read_text())
            paths = ['dataset/local-data.json', 'verl/utils/dataset/rl_dataset.py',
                     'verl/experimental/dataset/sampler.py']
            result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'],
                                    cwd=directory, input='\n'.join(paths) + '\n',
                                    text=True, capture_output=True, check=True)
            self.assertEqual(result.stdout.splitlines(), ['dataset/local-data.json'])

    def test_compact_layout(self):
        self.assertEqual({path.name for path in (ROOT / 'scripts').iterdir()
                          if path.is_dir() and path.name != '__pycache__'},
                         {'train', 'infer', 'eval', 'data', 'services', 'utils'})
        self.assertFalse(any(path.is_file() for path in (ROOT / 'scripts').iterdir()))
        for name in ('smoke_test.py', 'check_tool_services.py'):
            self.assertFalse((ROOT / 'scripts' / name).exists(), name)
        for name in ('recipe', 'docker', 'examples', 'my_scripts', 'release_templates'):
            self.assertFalse((ROOT / name).exists(), name)
        for name in ('configs/train.yaml', 'configs/tools.yaml', 'eventmemagent/memory/preprocess.py',
                     'eventmemagent/services/ocr_server.py', 'scripts/train/train.sh'):
            self.assertTrue((ROOT / name).is_file(), name)

    def test_project_modules_resolve_and_do_not_break_into_debugger(self):
        for path in (ROOT / 'eventmemagent').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call):
                    self.assertFalse(isinstance(node.func, ast.Name) and node.func.id == 'breakpoint', path)
                    self.assertFalse(isinstance(node.func, ast.Attribute) and node.func.attr == 'set_trace', path)
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('eventmemagent'):
                    target = ROOT.joinpath(*node.module.split('.'))
                    self.assertTrue(target.with_suffix('.py').is_file() or (target / '__init__.py').is_file(), node.module)

    def test_python_package_is_grouped_by_responsibility(self):
        package = ROOT / 'eventmemagent'
        self.assertEqual({p.name for p in package.glob('*.py')}, {'__init__.py'})
        self.assertEqual({p.name for p in package.iterdir() if p.is_dir() and p.name != '__pycache__'},
                         {'agent', 'memory', 'inference', 'evaluation', 'services', 'utils'})
        for name in ('agent', 'memory', 'inference', 'evaluation', 'services', 'utils'):
            self.assertTrue((package / name / '__init__.py').is_file())

    def test_relocated_cli_defaults_resolve_repository_root(self):
        for relative in ('inference/run.py', 'evaluation/run.py', 'evaluation/batch.py', 'evaluation/streamingbench.py'):
            path = ROOT / 'eventmemagent' / relative
            root_assignment = next(n for n in ast.parse(path.read_text()).body
                                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ROOT' for t in n.targets))
            value = eval(compile(ast.Expression(root_assignment.value), str(path), 'eval'),
                         {'os': os, '__file__': str(path)})
            self.assertEqual(Path(value), ROOT)

    def test_pip_constraints_have_no_extras(self):
        for line in (ROOT / 'constraints-training.txt').read_text().splitlines():
            if line.strip() and not line.lstrip().startswith('#'):
                self.assertNotIn('[', line, 'pip does not allow package extras in constraints')

    def test_training_config(self):
        config = yaml.safe_load((ROOT / 'configs/train.yaml').read_text())
        actor = config['actor_rollout_ref']['actor']
        rollout = config['actor_rollout_ref']['rollout']
        self.assertEqual(config['algorithm']['adv_estimator'], 'grpo')
        self.assertEqual(config['data']['train_batch_size'], 64)
        self.assertEqual(config['data']['max_response_length'], 8192)
        self.assertEqual(float(actor['optim']['lr']), 1e-6)
        self.assertEqual(actor['optim']['lr_warmup_steps'], 5)
        self.assertEqual(actor['ppo_mini_batch_size'], 32)
        self.assertEqual(actor['ppo_micro_batch_size_per_gpu'], 1)
        self.assertFalse(actor['use_kl_loss'])
        self.assertEqual(actor['kl_loss_coef'], 0)
        self.assertEqual(rollout['n'], 8)
        self.assertEqual(rollout['multi_turn']['max_assistant_turns'], 10)
        self.assertEqual(config['trainer']['total_epochs'], 1)
        self.assertIn('Qwen3-VL-8B-Instruct', config['actor_rollout_ref']['model']['path'])

    def test_source_archive_does_not_require_git_metadata(self):
        validator = load_module('scripts/utils/validate_setup.py')
        errors = []
        with patch.object(validator.subprocess, 'run', return_value=subprocess.CompletedProcess([], 128, '', 'not a git repository')):
            validator.check_repo(errors)
        self.assertEqual(errors, [])

    def test_memory_tools(self):
        config = yaml.safe_load((ROOT / 'configs/tools.yaml').read_text())
        tools = {tool['tool_schema']['function']['name']: tool['config'] for tool in config['tools']}
        self.assertEqual(set(tools), {'search_memory', 'ocr', 'detect_objects'})
        self.assertEqual(tools['search_memory']['top_k'], 3)
        self.assertEqual(tools['search_memory']['similarity_threshold'], 0.3)

    def test_no_validation_preserves_training_configuration(self):
        env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH']}
        def compose(*args):
            result = subprocess.run(['bash', 'scripts/train/train.sh', *args, '--dry-run'],
                                    cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            return yaml.safe_load(result.stdout)
        original = compose()
        no_val = compose('--no-validation')
        self.assertEqual(no_val['data']['val_files'], [])
        self.assertFalse(no_val['trainer']['val_before_train'])
        self.assertFalse(no_val['trainer']['val_only'])
        self.assertEqual(no_val['trainer']['test_freq'], -1)
        for section, keys in {'data': ('val_files',),
                              'trainer': ('val_before_train', 'val_only', 'test_freq')}.items():
            for key in keys:
                original[section].pop(key, None)
                no_val[section].pop(key, None)
        self.assertEqual(original, no_val)

    def test_eval_does_not_inherit_training_tp(self):
        env = {**os.environ, 'TRAIN_TENSOR_PARALLEL_SIZE': '8', 'EVAL_TENSOR_PARALLEL_SIZE': '1', 'EVAL_CUDA_VISIBLE_DEVICES': '0'}
        for key in ('TENSOR_PARALLEL_SIZE', 'CUDA_VISIBLE_DEVICES'):
            env.pop(key, None)
        process = subprocess.run(['bash', '-c', 'source scripts/utils/common.sh; configure_eval_devices; test "$TENSOR_PARALLEL_SIZE" = 1'], cwd=ROOT, env=env)
        self.assertEqual(process.returncode, 0)

    def test_eval_rejects_impossible_gpu_request(self):
        process = subprocess.run(['bash', '-c', 'source scripts/utils/common.sh; configure_eval_devices'], cwd=ROOT,
                                 env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'TENSOR_PARALLEL_SIZE': '8'}, capture_output=True)
        self.assertNotEqual(process.returncode, 0)


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.streaming = load_module('eventmemagent/evaluation/score_streamingbench.py')
        self.ovo = load_module('eventmemagent/evaluation/score_ovobench.py')
        self.reward = load_module('verl/utils/reward_score/video_agent.py')
        self.questions = [{'video_path': 'sample_1.mp4', 'questions': [
            {'question': 'What object?', 'time_stamp': '00:00:10', 'task_type': 'Object Recognition', 'answer': 'A'}]}]

    def test_archived_parquet_result(self):
        data = {'results': [{'video_path': '/old/server/sample_1_0_10.mp4', 'question': 'What object?\n\nOptions:\nA. Car', 'correct': True}]}
        result = self.streaming.calculate_parquet_metrics(data, self.questions)
        self.assertEqual(result['overall_accuracy_weighted'], 100)
        self.assertEqual(result['missing_prediction_count'], 0)

    def test_duplicate_metadata_is_preserved(self):
        self.questions[0]['questions'] *= 2
        row = {'video_path': 'sample_1_0_10.mp4', 'question': 'What object?', 'score': 1}
        result = self.streaming.calculate_parquet_metrics({'results': [row, row]}, self.questions)
        self.assertEqual(result['total_samples'], 2)
        with self.assertRaises(ValueError):
            self.streaming.calculate_parquet_metrics({'results': [row, row, row]}, self.questions)

    def test_unmatched_is_not_reported_as_complete(self):
        result = self.streaming.calculate_parquet_metrics({'results': [{'video_path': 'missing.mp4'}]}, self.questions)
        self.assertEqual(result['unmatched_count'], 1)
        self.assertEqual(result['missing_prediction_count'], 1)

    def test_empty_answers_are_not_correct(self):
        result = self.streaming.calculate_metrics([{'questions': [{}]}], 'EventMemAgent')
        self.assertEqual(result['total_correct'], 0)

    def test_answer_reward_and_repeated_box_penalty(self):
        self.assertEqual(self.reward.compute_score('test', r'\boxed{A}', 'A'), 1.0)
        self.assertEqual(self.reward.compute_score('test', r'\boxed{B}', 'A'), 0.0)
        self.assertAlmostEqual(self.reward.compute_score('test', r'\boxed{A} then \boxed{A}', 'A'), 0.9)

    def test_ovo_uses_category_mean(self):
        metadata = [{'id': 1, 'task': 'EPM', 'gt': 0}, {'id': 2, 'task': 'STU', 'gt': 1}]
        result = {'results': [{'video_path': '1.mp4', 'extracted_answer': 'A'}, {'video_path': '2.mp4', 'extracted_answer': 'A'}]}
        metrics = self.ovo.calculate_metrics(result, metadata)
        self.assertEqual(metrics[4], 50.0)


if __name__ == '__main__':
    unittest.main()
