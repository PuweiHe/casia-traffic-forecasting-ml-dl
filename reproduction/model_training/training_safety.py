"""Scheduling safety outside frozen numerical training code."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def atomic_record(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n')
    temporary.replace(path)


def terminate_and_reap(child, grace=30):
    if child.poll() is None:
        child.terminate()
    try:
        return child.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait()


def supervise(command, cwd, log, output, seconds, grace=30):
    """Reap the actual writer on timeout, signal, or supervisor exceptions."""
    stopping = []
    previous = {}
    def stop(signum, frame):
        stopping.append(signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous[sig] = signal.signal(sig, stop)
    child = None
    deadline = time.monotonic() + seconds
    state, code = 'failed', None
    try:
        child = subprocess.Popen(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        atomic_record(output / 'launch_record.json', {
            'writer_pid':child.pid, 'supervisor_pid':os.getpid(),
            'started_unix':time.time(), 'deadline_unix':time.time()+seconds,
            'budget_seconds':seconds})
        (output / 'driver.pid').write_text(str(child.pid)+'\n')
        while True:
            if stopping:
                state = 'interrupted_checkpoint_preserved'
                code = terminate_and_reap(child, grace)
                break
            if time.monotonic() >= deadline:
                state = 'wall_budget_stopped_checkpoint_preserved'
                code = terminate_and_reap(child, grace)
                break
            try:
                code = child.wait(timeout=.2)
                state = 'complete' if code == 0 else 'stopped_inspect_log'
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        if child is not None and child.poll() is None:
            code = terminate_and_reap(child, grace)
        atomic_record(output / 'supervisor_status.json', {'state':state,'exit_code':code})
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return state, code


def process_identity(pid):
    """Check start identity as well as PID before operating on an existing writer."""
    p = subprocess.run(['ps','-p',str(pid),'-o','lstart=','-o','command='],
                       capture_output=True,text=True)
    return p.stdout.strip() if p.returncode == 0 else None
