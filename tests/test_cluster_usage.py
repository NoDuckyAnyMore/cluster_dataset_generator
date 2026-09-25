"""Offline fixtures for the standalone shell report; no Slurm/GPU required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


GENERATOR = Path(__file__).resolve().parents[1]
BASH = (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt"
        else shutil.which("bash"))
MOCKS = r'''
sacct() { printf '%s\n' "$FAKE_SACCT"; return "${FAKE_FAILURE:-0}"; }
timeout() { shift; "$@"; }
df() { echo MOCK_DF; }
quota() { echo MOCK_QUOTA; }
du() { echo MOCK_DU; }
export -f sacct timeout df quota du
bash cluster_usage.sh "$@"
'''


def record(idx, partition, start, end, seconds, tres, job=None):
    return f"cluster|{idx}|{job or idx}|{partition}|{start}|{end}|{seconds}|{tres}"


@unittest.skipUnless(BASH, "Bash is required")
class ClusterUsageTests(unittest.TestCase):
    def run_report(self, rows, *args, **extra_env):
        env = dict(os.environ, FAKE_SACCT="\n".join(rows), UAV_VAST_DIR=".",
                   UAV_USAGE_TZ="UTC", UAV_USAGE_START="1970-01-01",
                   UAV_RATE_5090="2.7", UAV_RATE_4090="2.16", **extra_env)
        return subprocess.run([BASH, "--noprofile", "--norc", "-c", MOCKS,
                               "test-usage", *args], cwd=GENERATOR, env=env,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=20)

    def values(self, result, label):
        lines = [l.split() for l in result.stdout.splitlines()
                 if l.startswith(label + " ")]
        self.assertEqual(len(lines), 1, result.stdout + result.stderr)
        return [float(v) for v in lines[0][1:]]

    def test_cross_midnight_month_multigpu_duplicate_and_empty_days(self):
        a = record(1, "gpu_5090", "2026-08-31T23:00:00", "2026-09-01T02:00:00",
                   10800, "cpu=8,gres/gpu=2,gres/gpu:rtx5090=2")
        rows = [a, a,
                record(2, "gpu_4090", "2026-09-01T01:00:00", "2026-09-01T03:00:00",
                       7200, "gres/gpu=1"),
                record(3, "gpu_5090", "Unknown", "Unknown", 0, ""),
                record(4, "gpu_5090", "2026-09-01T01:00:00", "2026-09-01T03:00:00",
                       7200, "gres/gpu=1", job="1.batch")]
        r = self.run_report(rows, "--since", "2026-08-01", "--until", "2026-10-01")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.values(r, "2026-08-31"), [2, 0, 0, 2, 5.4, 0, 5.4])
        self.assertEqual(self.values(r, "2026-09-01"), [4, 2, 0, 6, 10.8, 4.32, 15.12])
        self.assertEqual(self.values(r, "TOTAL"), [6, 2, 0, 8, 16.2, 4.32, 20.52])
        self.assertEqual(self.values(r, "2026-08"), [2, 0, 0, 2, 5.4, 0, 5.4])
        self.assertNotIn("2026-09-02 ", r.stdout)
        self.assertIn("MOCK_DU", r.stdout)

    def test_running_job_clipped_to_query_and_leap_day(self):
        rows = [record(1, "gpu_5090", "2024-02-28T23:00:00", "Unknown",
                       26*3600, "gres/gpu=1")]
        r = self.run_report(rows, "--since", "2024-02-29", "--until", "2024-03-01T01:00:00")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.values(r, "TOTAL")[3], 25)
        self.assertEqual(self.values(r, "2024-02-29")[3], 24)
        self.assertEqual(self.values(r, "2024-03-01")[3], 1)

    def test_suspension_prorated_and_unknown_gpu_not_silently_free(self):
        rows = [record(1, "gpu_4090", "2026-09-01T23:00:00", "2026-09-02T01:00:00",
                       3600, "gres/gpu=1"),
                record(2, "gpu_a100", "2026-09-01T00:00:00", "2026-09-01T01:00:00",
                       3600, "gres/gpu=2")]
        r = self.run_report(rows, "--since", "2026-09-01", "--until", "2026-09-03", "--no-du")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.values(r, "TOTAL"), [0, 1, 2, 3, 0, 2.16, 2.16])
        self.assertEqual(self.values(r, "2026-09-02")[1], 0.5)
        self.assertIn("比例分摊", r.stderr)
        self.assertIn("其他型号", r.stderr)
        self.assertNotIn("MOCK_DU", r.stdout)
        self.assertNotIn("每月统计", r.stdout)

    def test_failure_still_queries_storage_without_zero_total(self):
        r = self.run_report([], "--until", "2026-09-03", FAKE_FAILURE="1")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Slurm 查询失败", r.stderr)
        self.assertIn("MOCK_DF", r.stdout)
        self.assertNotIn("TOTAL", r.stdout)

    def test_no_usage(self):
        r = self.run_report([], "--until", "2026-09-03")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.values(r, "TOTAL"), [0]*7)
        self.assertNotIn("每日统计", r.stdout)

    def test_mixed_gpu_types_and_reused_job_id(self):
        rows = [record(1, "mixed", "2026-09-01T00:00:00", "2026-09-01T01:00:00",
                       3600, "gres/gpu=3,gres/gpu:rtx5090=2,gres/gpu:rtx4090=1", job="123"),
                record(2, "gpu_4090", "2026-09-02T00:00:00", "2026-09-02T01:00:00",
                       3600, "gres/gpu:rtx4090=1", job="123")]
        r = self.run_report(rows, "--since", "2026-09-01", "--until", "2026-09-03")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.values(r, "TOTAL"), [2, 2, 0, 4, 5.4, 4.32, 9.72])

    def test_install_is_idempotent_and_function_passes_arguments(self):
        with tempfile.TemporaryDirectory(prefix="uav-usage-test-") as td:
            rc = Path(td) / "test shell rc"
            for _ in range(2):
                r = self.run_report([], "--install", UAV_USAGE_RC=rc.as_posix())
                self.assertEqual(r.returncode, 0, r.stderr)
            content = rc.read_text(encoding="utf-8")
            self.assertEqual(content.count("# CLUSTER_USAGE_COMMAND"), 1)
            r = subprocess.run([BASH, "--noprofile", "--norc", "-c",
                                'source "$1"; cluster_usage --help', "test-source", rc.as_posix()],
                               capture_output=True, text=True, encoding="utf-8", timeout=20)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("用法", r.stdout)


if __name__ == "__main__":
    unittest.main()
