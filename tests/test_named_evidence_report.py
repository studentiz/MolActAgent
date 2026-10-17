import copy
import json
import re
import subprocess
import sys
from pathlib import Path

from molactagent.report import render_html_report


def _payload(rendered):
    return json.loads(re.search(
        r'<script type="application/json" id="evidence-data">(.*?)</script>',
        rendered, re.S,
    ).group(1))


def test_named_viewer_embeds_record_losslessly_and_escapes_script_delimiters():
    hostile = '</script><script>window.injected=true</script>&\u2028'
    record = {'final_label': 'active', 'confidence': .508,
              'candidate_classes': ['active', 'inactive'], 'tuple_results': [],
              'reason': hostile, 'report_context': {'sample_name': hostile}}
    original = copy.deepcopy(record)
    rendered = render_html_report(record, explanation=hostile, title=hostile)
    payload = re.search(r'<script type="application/json" id="evidence-data">(.*?)</script>', rendered, re.S).group(1)
    decoded = json.loads(payload)
    assert decoded['record'] == {k: v for k, v in record.items() if k != 'report_context'}
    assert decoded['context']['sample_name'] == hostile
    assert decoded['explanation'] == hostile
    assert '<script>window.injected' not in rendered
    assert record == original


def test_existing_classification_path_is_unchanged_without_context():
    rendered = render_html_report({'candidate_classes': ['active', 'inactive'],
                                  'final_label': 'active', 'confidence': .8})
    assert 'evidence-data' not in rendered
    assert 'active' in rendered


def test_regression_dispatch_has_priority_over_optional_context():
    rendered = render_html_report({'task': 'regression', 'predicted_value': 1.5,
                                  'confidence': .8, 'report_context': {'sample_name': 'Example'}})
    assert 'evidence-data' not in rendered
    assert '1.5' in rendered


def test_template_tokens_in_user_text_remain_literal():
    record = {'final_label': 'active', 'reason': '__TITLE__ __PAYLOAD__',
              'report_context': {'sample_name': '__PAYLOAD__ __TITLE__'}}
    rendered = render_html_report(record, title='__PAYLOAD__ & <Example>')
    assert _payload(rendered)['record']['reason'] == record['reason']
    assert _payload(rendered)['context'] == record['report_context']
    assert '<title>__PAYLOAD__ &amp; &lt;Example&gt;</title>' in rendered


def test_named_viewer_preserves_all_evidence_and_class_label_override():
    path = Path(__file__).resolve().parents[1] / 'examples/evidence_report/illustrative_record.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    original = copy.deepcopy(record)
    rendered = render_html_report(record, class_labels=['inactive', 'active'])
    data = _payload(rendered)
    assert data['class_labels'] == ['inactive', 'active']
    assert data['record'] == {k: v for k, v in record.items() if k != 'report_context'}
    assert data['context'] == record['report_context']
    assert record == original
    for section in ['supporting-evidence', 'opposing-evidence', 'driver-attributions',
                    'llm-uncertainty', 'local-validation', 'feature-table',
                    'tuple-table', 'confidence-basis', 'reference-pool', 'raw']:
        assert f'id="{section}"' in rendered
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)=', rendered)


def test_offline_example_exports_from_another_working_directory(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'examples/evidence_report/render.py'
    target = tmp_path / 'nested' / 'report.html'
    completed = subprocess.run(
        [sys.executable, str(script), '--output', str(target)],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert 'no API call' in completed.stdout
    data = _payload(target.read_text(encoding='utf-8'))
    assert len(data['record']['tuple_results']) == 2
    assert 'synthetic' in data['record']['uncertainty']
