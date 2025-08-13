# flsim/network.py — ns-3 THz runner (sync + async with timeout & robust parsing)
import json
import subprocess
import time
from typing import Any, Dict, List, Optional

PATH = '../ns3-fl-network'
PROGRAM = 'scratch/thz-macro-central'


def _get(root: Any, path: List[str], default=None):
    cur = root
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif hasattr(cur, k):
            cur = getattr(cur, k)
        else:
            return default
    return cur


class Network(object):
    def __init__(self, config):
        self.config = config
        self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

        # THz sim knobs (safe defaults; override via config.network.thz.*)
        thz = _get(self.config, ['network', 'thz'], {}) or {}
        self._thz_cfg = {
            'pkt_size':     int(thz.get('pkt_size', 600)),
            'sim_time':     float(thz.get('sim_time', 0.8)),
            'interval_us':  int(thz.get('interval_us', 20)),
            'way':          int(thz.get('way', 3)),
            'radius':       float(thz.get('radius', 0.5)),
            'beamwidth':    float(thz.get('beamwidth', 40)),
            'gain':         float(thz.get('gain', 30)),
            'ap_angle':     float(thz.get('ap_angle', 0)),
            'sta_angle':    float(thz.get('sta_angle', 180)),
            'useWhiteList': int(thz.get('useWhiteList', 0)),
        }

        # model bytes per upload
        self._model_bytes = int(_get(self.config, ['model', 'size'], 1600))

        # build ns-3 once
        proc = subprocess.run(
            './ns3 build', shell=True, cwd=PATH,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

        # async state
        self._proc: Optional[subprocess.Popen] = None
        self._async_ids: List[int] = []
        self._async_queue: List[Dict[int, Dict[str, float]]] = []
        self._deadline: Optional[float] = None  # wall-clock timeout for async job

    # ------------------------------------------------------------------
    # compatibility no-ops (old TCP control plane)
    def connect(self): return
    def disconnect(self): return

    # accept list of client objects or raw ids
    def parse_clients(self, clients):
        if len(clients) and hasattr(clients[0], 'client_id'):
            return [c.client_id for c in clients]
        return list(map(int, clients))

    # ------------------------------------------------------------------
    # core ns-3 launchers
    def _cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
        t = self._thz_cfg
        return [
            './ns3', 'run', PROGRAM, '--',
            f'--nodeNum={total_clients}',
            f'--clients={active_count}',
            f'--modelBytes={model_bytes}',
            f'--pktSize={t["pkt_size"]}',
            f'--simTime={t["sim_time"]}',
            f'--intervalUs={t["interval_us"]}',
            f'--way={t["way"]}',
            f'--radius={t["radius"]}',
            f'--beamwidth={t["beamwidth"]}',
            f'--gain={t["gain"]}',
            f'--apAngle={t["ap_angle"]}',
            f'--staAngle={t["sta_angle"]}',
            f'--useWhiteList={t["useWhiteList"]}',
        ]

    @staticmethod
    def _parse_last_json(stdout: str) -> Dict[str, Any]:
        last = None
        for line in reversed(stdout.splitlines()):
            s = line.strip()
            if s.startswith('{') and s.endswith('}'):
                last = s
                break
        if not last:
            raise RuntimeError('No JSON summary found in ns-3 output.')
        return json.loads(last)

    @staticmethod
    def _extract_times(entry: Dict[str, Any], default_sim_time: float) -> float:
        # tolerate different field names emitted by the sim
        if 'doneAt' in entry:
            return float(entry['doneAt'])
        if 'endTime' in entry:
            return float(entry['endTime'])
        if 'roundTime' in entry:
            return float(entry['roundTime'])
        return default_sim_time

    # ------------------------------------------------------------------
    # SYNC API
    def sendRequest(self, *, requestType: int, array: list):
        # bitmap or list of ids
        if len(array) == self.num_clients and all(x in (0, 1) for x in array):
            active_ids = [i for i, flag in enumerate(array) if flag]
        else:
            active_ids = self.parse_clients(array)

        if not active_ids:
            return {}

        cmd = self._cmd(
            total_clients=self.num_clients,
            active_count=len(active_ids),
            model_bytes=self._model_bytes,
        )
        proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
        data = self._parse_last_json(proc.stdout)

        # map local ids 0..N-1 -> real ids
        id_map = {local: active_ids[local] for local in range(len(active_ids))}
        out = {}
        for e in data.get('clientResults', []):
            local = int(e.get('id', -1))
            if local not in id_map:
                continue
            rx_bytes = float(e.get('rxBytes', 0.0))
            done_at  = self._extract_times(e, self._thz_cfg['sim_time'])
            thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
            out[id_map[local]] = {
                'roundTime': done_at,
                'throughput': thr,
            }
        return out

    # ------------------------------------------------------------------
    # ASYNC API (with timeout/fallback)
    def sendAsyncRequest(self, *, requestType: int, array: list):
        if self._proc is not None:
            raise RuntimeError('Async request already in progress.')

        # bitmap or list
        if len(array) == self.num_clients and all(x in (0, 1) for x in array):
            active_ids = [i for i, flag in enumerate(array) if flag]
        else:
            active_ids = self.parse_clients(array)

        self._async_ids = active_ids
        self._async_queue = []  # will be filled once process ends
        self._deadline = None

        if not active_ids:
            # nothing to do — synthesize empty and finish
            self._proc = None
            self._async_queue = []
            return

        cmd = self._cmd(
            total_clients=self.num_clients,
            active_count=len(active_ids),
            model_bytes=self._model_bytes,
        )
        self._proc = subprocess.Popen(
            cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        # Allow plenty of margin over sim_time; never block forever
        sim_t = self._thz_cfg['sim_time']
        self._deadline = time.time() + max(10.0, 4.0 * sim_t)

    def readAsyncResponse(self):
        """
        Poll once:
        - returns {} while ns-3 is running
        - then returns one client’s dict at a time: {id: {...}}
        - when all delivered, returns 'end'
        - if deadline exceeded, kill process and synthesize results
        """
        import subprocess

        # nothing ever started
        if self._proc is None and not self._async_queue:
            return 'end'

        # still running
        if self._proc is not None and self._proc.poll() is None:
            # timeout guard
            if self._deadline is not None and time.time() > self._deadline:
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                # try graceful, then hard kill
                try:
                    stdout, stderr = self._proc.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        self._proc.kill()
                    except Exception:
                        pass
                    stdout, stderr = self._proc.communicate()
                finally:
                    self._proc = None

                # Try to parse output; if none, synthesize "sim finished" entries
                data = {}
                try:
                    data = self._parse_last_json(stdout)
                except Exception:
                    data = {'clientResults': []}

                id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
                results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

                # build per-client responses; synthesize missing with sim_time
                for local in range(len(self._async_ids)):
                    ent = results.get(local)
                    real_id = id_map[local]
                    if ent:
                        rx_bytes = float(ent.get('rxBytes', 0.0))
                        done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
                        thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
                    else:
                        done_at = self._thz_cfg['sim_time']
                        thr = 0.0
                    self._async_queue.append({
                        real_id: {
                            'startTime': 0.0,
                            'endTime': done_at,
                            'throughput': thr,
                        }
                    })
                # fall through to serve queue
            else:
                return {}

        # finished: if queue not built yet, parse and build single-client chunks
        if self._proc is not None and self._proc.poll() is not None:
            stdout, stderr = self._proc.communicate()
            self._proc = None

            data = self._parse_last_json(stdout)
            # map local -> real ids
            id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
            results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

            # build per-client responses in a queue
            for local in range(len(self._async_ids)):
                ent = results.get(local)
                if not ent:
                    continue
                real_id = id_map[local]
                rx_bytes = float(ent.get('rxBytes', 0.0))
                done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
                thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
                self._async_queue.append({
                    real_id: {
                        'startTime': 0.0,
                        'endTime': done_at,
                        'throughput': thr,
                    }
                })

        # serve one and pop
        if self._async_queue:
            return self._async_queue.pop(0)

        return 'end'


    # def readAsyncResponse(self):
    #     """
    #     Poll once:
    #       - returns {} while ns-3 is running
    #       - then returns one client’s dict at a time: {id: {...}}
    #       - when all delivered, returns 'end'
    #       - if deadline exceeded, kill process and synthesize results
    #     """
    #     # nothing ever started
    #     if self._proc is None and not self._async_queue:
    #         return 'end'

    #     # still running
    #     if self._proc is not None and self._proc.poll() is None:
    #         # timeout guard
    #         if self._deadline is not None and time.time() > self._deadline:
    #             try:
    #                 self._proc.terminate()
    #             except Exception:
    #                 pass
    #             stdout = self._proc.stdout.read() if self._proc.stdout else ''
    #             stderr = self._proc.stderr.read() if self._proc.stderr else ''
    #             self._proc = None

    #             # Try to parse output; if none, synthesize "sim finished" entries
    #             data = {}
    #             try:
    #                 data = self._parse_last_json(stdout)
    #             except Exception:
    #                 data = {'clientResults': []}

    #             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
    #             results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

    #             # build per-client responses; synthesize missing with sim_time
    #             for local in range(len(self._async_ids)):
    #                 ent = results.get(local)
    #                 real_id = id_map[local]
    #                 if ent:
    #                     rx_bytes = float(ent.get('rxBytes', 0.0))
    #                     done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
    #                     thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
    #                 else:
    #                     done_at = self._thz_cfg['sim_time']
    #                     thr = 0.0
    #                 self._async_queue.append({
    #                     real_id: {
    #                         'startTime': 0.0,
    #                         'endTime': done_at,
    #                         'throughput': thr,
    #                     }
    #                 })
    #             # fall through to serve queue
    #         else:
    #             return {}

    #     # finished: if queue not built yet, parse and build single-client chunks
    #     if self._proc is not None:
    #         stdout = self._proc.stdout.read() if self._proc.stdout else ''
    #         stderr = self._proc.stderr.read() if self._proc.stderr else ''
    #         self._proc = None

    #         data = self._parse_last_json(stdout)
    #         # map local -> real ids
    #         id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
    #         results = {int(e.get('id', -1)): e for e in data.get('clientResults', [])}

    #         # build per-client responses in a queue
    #         for local in range(len(self._async_ids)):
    #             ent = results.get(local)
    #             if not ent:
    #                 continue
    #             real_id = id_map[local]
    #             rx_bytes = float(ent.get('rxBytes', 0.0))
    #             done_at  = self._extract_times(ent, self._thz_cfg['sim_time'])
    #             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
    #             self._async_queue.append({
    #                 real_id: {
    #                     'startTime': 0.0,
    #                     'endTime': done_at,
    #                     'throughput': thr,
    #                 }
    #             })

    #     # serve one and pop
    #     if self._async_queue:
    #         return self._async_queue.pop(0)

    #     return 'end'


# # flsim/network.py — ns-3 THz runner (sync + async)
# import json
# import subprocess
# from typing import Any, Dict, List, Optional

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'


# def _get(root: Any, path: List[str], default=None):
#     cur = root
#     for k in path:
#         if isinstance(cur, dict) and k in cur:
#             cur = cur[k]
#         elif hasattr(cur, k):
#             cur = getattr(cur, k)
#         else:
#             return default
#     return cur


# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(_get(self.config, ['clients', 'total'], 1))

#         # THz sim knobs (safe defaults; override via config.network.thz.*)
#         thz = _get(self.config, ['network', 'thz'], {}) or {}
#         self._thz_cfg = {
#             'pkt_size':    int(thz.get('pkt_size', 600)),
#             'sim_time':    float(thz.get('sim_time', 0.8)),
#             'interval_us': int(thz.get('interval_us', 20)),
#             'way':         int(thz.get('way', 3)),
#             'radius':      float(thz.get('radius', 0.5)),
#             'beamwidth':   float(thz.get('beamwidth', 40)),
#             'gain':        float(thz.get('gain', 30)),
#             'ap_angle':    float(thz.get('ap_angle', 0)),
#             'sta_angle':   float(thz.get('sta_angle', 180)),
#             'useWhiteList':int(thz.get('useWhiteList', 0)),
#         }

#         # model bytes per upload
#         self._model_bytes = int(_get(self.config, ['model', 'size'], 1600))

#         # build ns-3 once
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # async state
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_queue: List[Dict[int, Dict[str, float]]] = []

#     # ------------------------------------------------------------------
#     # compatibility no-ops (old TCP control plane)
#     def connect(self): return
#     def disconnect(self): return

#     # accept list of client objects or raw ids
#     def parse_clients(self, clients):
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     # ------------------------------------------------------------------
#     # core ns-3 launchers
#     def _cmd(self, *, total_clients: int, active_count: int, model_bytes: int) -> List[str]:
#         t = self._thz_cfg
#         return [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={t["pkt_size"]}',
#             f'--simTime={t["sim_time"]}',
#             f'--intervalUs={t["interval_us"]}',
#             f'--way={t["way"]}',
#             f'--radius={t["radius"]}',
#             f'--beamwidth={t["beamwidth"]}',
#             f'--gain={t["gain"]}',
#             f'--apAngle={t["ap_angle"]}',
#             f'--staAngle={t["sta_angle"]}',
#             f'--useWhiteList={t["useWhiteList"]}',
#         ]

#     @staticmethod
#     def _parse_last_json(stdout: str) -> Dict[str, Any]:
#         last = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     # ------------------------------------------------------------------
#     # SYNC API
#     def sendRequest(self, *, requestType: int, array: list):
#         # bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         if not active_ids:
#             return {}

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')
#         data = self._parse_last_json(proc.stdout)

#         # map local ids 0..N-1 -> real ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         out = {}
#         for e in data.get('clientResults', []):
#             local = int(e.get('id', -1))
#             if local not in id_map:
#                 continue
#             rx_bytes = float(e.get('rxBytes', 0.0))
#             done_at  = float(e.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local]] = {
#                 'roundTime': done_at if done_at >= 0 else self._thz_cfg['sim_time'],
#                 'throughput': thr,
#             }
#         return out

#     # ------------------------------------------------------------------
#     # ASYNC API
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # bitmap or list
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = self.parse_clients(array)

#         self._async_ids = active_ids
#         self._async_queue = []  # will be filled once process ends

#         if not active_ids:
#             # nothing to do — synthesize empty and finish
#             self._proc = None
#             self._async_queue = []
#             return

#         cmd = self._cmd(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=self._model_bytes,
#         )
#         self._proc = subprocess.Popen(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )

#     def readAsyncResponse(self):
#         """
#         Poll once:
#           - returns {} while ns-3 is running
#           - then returns one client’s dict at a time: {id: {...}}
#           - when all delivered, returns 'end'
#         """
#         # nothing ever started
#         if self._proc is None and not self._async_queue:
#             return 'end'

#         # still running
#         if self._proc is not None and self._proc.poll() is None:
#             return {}

#         # finished: if queue not built yet, parse and build single-client chunks
#         if self._proc is not None:
#             stdout = self._proc.stdout.read() if self._proc.stdout else ''
#             stderr = self._proc.stderr.read() if self._proc.stderr else ''
#             self._proc = None

#             data = self._parse_last_json(stdout)
#             # map local -> real ids
#             id_map = {local: self._async_ids[local] for local in range(len(self._async_ids))}
#             results = {int(e['id']): e for e in data.get('clientResults', [])}

#             # build per-client responses in a queue
#             for local in range(len(self._async_ids)):
#                 ent = results.get(local)
#                 if not ent:
#                     continue
#                 real_id = id_map[local]
#                 rx_bytes = float(ent.get('rxBytes', 0.0))
#                 done_at  = float(ent.get('doneAt', -1.0))
#                 thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#                 self._async_queue.append({
#                     real_id: {
#                         'startTime': 0.0,
#                         'endTime': done_at if done_at >= 0 else self._thz_cfg['sim_time'],
#                         'throughput': thr,
#                     }
#                 })

#         # serve one and pop
#         if self._async_queue:
#             return self._async_queue.pop(0)

#         return 'end'


# # network.py — ns-3 THz runner (drop-in)
# import json
# import subprocess

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = int(self.config.clients.total)

#         # Build ns-3 once up-front
#         proc = subprocess.run(
#             './ns3 build', shell=True, cwd=PATH,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#     def connect(self):    # no TCP control plane
#         return

#     def disconnect(self):
#         return

#     def parse_clients(self, clients):
#         # Accept list of client objects or raw ids
#         if len(clients) and hasattr(clients[0], 'client_id'):
#             return [c.client_id for c in clients]
#         return list(map(int, clients))

#     def _get_thz_cfg(self):
#         # Safe defaults + allow overrides from config.json
#         thz = getattr(self.config.network, 'thz', {})
#         g = lambda k, d: thz.get(k, d) if isinstance(thz, dict) else getattr(thz, k, d)
#         return {
#             'pkt_size':   int(g('pkt_size', 600)),
#             'sim_time':   float(g('sim_time', 0.8)),
#             'interval_us':int(g('interval_us', 20)),
#             'way':        int(g('way', 3)),
#             'radius':     float(g('radius', 0.5)),
#             'beamwidth':  float(g('beamwidth', 40)),
#             'gain':       float(g('gain', 30)),
#             'ap_angle':   float(g('ap_angle', 0)),
#             'sta_angle':  float(g('sta_angle', 180)),
#             'useWhiteList': int(g('useWhiteList', 0)),
#         }

#     def _run_ns3(self, *, total_clients: int, active_count: int, model_bytes: int):
#         thz = self._get_thz_cfg()
#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={thz["pkt_size"]}',
#             f'--simTime={thz["sim_time"]}',
#             f'--intervalUs={thz["interval_us"]}',
#             f'--way={thz["way"]}',
#             f'--radius={thz["radius"]}',
#             f'--beamwidth={thz["beamwidth"]}',
#             f'--gain={thz["gain"]}',
#             f'--apAngle={thz["ap_angle"]}',
#             f'--staAngle={thz["sta_angle"]}',
#             f'--useWhiteList={thz["useWhiteList"]}',
#         ]
#         proc = subprocess.run(cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\nSTDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}')

#         # Find the last JSON line
#         last = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last = s
#                 break
#         if not last:
#             raise RuntimeError('No JSON summary found in ns-3 output.')
#         return json.loads(last)

#     def sendRequest(self, *, requestType: int, array: list):
#         # Accept either bitmap or list of ids
#         if len(array) == self.num_clients and all(x in (0,1) for x in array):
#             active_ids = [i for i, f in enumerate(array) if f]
#         else:
#             active_ids = self.parse_clients(array)
#         if not active_ids:
#             return {}

#         # Model bytes from config (bytes per client upload)
#         model_bytes = int(self.config.model.size)
#         data = self._run_ns3(
#             total_clients=self.num_clients,
#             active_count=len(active_ids),
#             model_bytes=model_bytes,
#         )

#         # ns-3 returns entries indexed 0..active_count-1; map back to real ids
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         ret = {}
#         for e in data.get('clientResults', []):
#             local_id = int(e['id'])
#             real_id = id_map.get(local_id, local_id)
#             rx_bytes = float(e.get('rxBytes', 0.0))
#             done_at  = float(e.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             ret[real_id] = {'roundTime': done_at if done_at >= 0 else self._get_thz_cfg()['sim_time'],
#                             'throughput': thr}
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         # Not wired for async yet
#         raise NotImplementedError('Async path not implemented for THz runner.')

#     def readAsyncResponse(self):
#         raise NotImplementedError('Async path not implemented for THz runner.')


# # flsim/network.py
# import json
# import subprocess

# PATH = '../ns3-fl-network'
# PROGRAM = 'scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = self.config.clients.total

#         # ns-3 knobs (from config, with safe defaults)
#         self.pkt_size = getattr(self.config.network.thz, 'pkt_size', 600)
#         self.sim_time = getattr(self.config.network.thz, 'sim_time', 0.5)
#         self.handshake_way = getattr(self.config.network.thz, 'way', 3)
#         self.interval_us = getattr(self.config.network.thz, 'interval_us', 20)
#         self.use_white_list = int(getattr(self.config.network.thz, 'use_white_list', 0))
#         self.ap_angle = getattr(self.config.network.thz, 'ap_angle', 0)
#         self.sta_angle = getattr(self.config.network.thz, 'sta_angle', 180)
#         self.radius = getattr(self.config.network.thz, 'radius', 0.5)
#         self.beamwidth = getattr(self.config.network.thz, 'beamwidth', 40)
#         self.gain = getattr(self.config.network.thz, 'gain', 30)

#         # Build once
#         proc = subprocess.run(
#             './ns3 build', cwd=PATH, shell=True, text=True,
#             stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}\n{proc.stdout}')

#     # API compatibility
#     def connect(self): return
#     def disconnect(self): return

#     def parse_clients(self, clients):
#         # return actual selected IDs
#         return [c.client_id for c in clients]

#     def _run_ns3(self, *, total_clients: int, active_ids: list, model_bytes: int):
#         """
#         Launch ns-3 and return parsed JSON (the program prints {"clientResults":[...]}).
#         We map local ids [0..N-1] to your selected client IDs.
#         """
#         cmd = [
#             './ns3','run',PROGRAM,'--',
#             f'--nodeNum={total_clients}',
#             f'--clients={len(active_ids)}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#             f'--intervalUs={self.interval_us}',
#             f'--useWhiteList={self.use_white_list}',
#             f'--apAngle={self.ap_angle}',
#             f'--staAngle={self.sta_angle}',
#             f'--radius={self.radius}',
#             f'--beamwidth={self.beamwidth}',
#             f'--gain={self.gain}',
#         ]
#         proc = subprocess.run(cmd, cwd=PATH, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 run failed:\n{proc.stderr}\n{proc.stdout}')

#         # parse the last JSON line
#         last_json = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(f'No JSON in ns-3 output.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}')
#         return json.loads(last_json)

#     def sendRequest(self, *, requestType: int, array: list):
#         # accept ID list or bitmap
#         if len(array) == self.num_clients and all(x in (0,1) for x in array):
#             active_ids = [i for i,v in enumerate(array) if v]
#         else:
#             active_ids = list(map(int, array))
#         if not active_ids:
#             return {}

#         data = self._run_ns3(
#             total_clients=self.num_clients,
#             active_ids=active_ids,
#             model_bytes=int(self.config.model.size)
#         )

#         # ns-3 emits local ids 0..N-1; map back to your chosen client IDs
#         id_map = {local: active_ids[local] for local in range(len(active_ids))}
#         ret = {}
#         for ent in data.get('clientResults', []):
#             local_id = int(ent.get('id', -1))
#             if local_id not in id_map:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at  = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             ret[id_map[local_id]] = {
#                 'roundTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr
#             }
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         raise NotImplementedError('Async path not wired for THz runner.')

#     def readAsyncResponse(self):
#         raise NotImplementedError('Async path not wired for THz runner.')



# # network.py — THz (ns-3) adapter for flsim
# import json
# import subprocess
# import shlex
# from typing import List, Dict, Any, Optional

# # Path to your ns-3 tree (the one with the ./ns3 helper script)
# PATH = '../ns3-fl-network'
# # Your ns-3 program
# PROGRAM = 'scratch/thz-macro-central'


# def _get_nested(root: Any, path: List[str], default=None):
#     """Works with both object-style and dict-style config objects."""
#     cur = root
#     for key in path:
#         if hasattr(cur, key):
#             cur = getattr(cur, key)
#         elif isinstance(cur, dict) and key in cur:
#             cur = cur[key]
#         else:
#             return default
#     return cur


# class Network(object):
#     def __init__(self, config):
#         self.config = config

#         # ---- read config (keeps MNIST + everything else untouched) ----
#         self.num_clients = int(_get_nested(self.config, ['clients', 'total'], 1))
#         self.network_type = _get_nested(self.config, ['network', 'type'], 'thz')

#         if self.network_type != 'thz':
#             raise ValueError('Set network.type to "thz" in config.json to use the THz ns-3 runner.')

#         thz_cfg = _get_nested(self.config, ['network', 'thz'], {}) or {}
#         # defaults safe for your current runs; override in config.network.thz
#         self.pkt_size = int(thz_cfg.get('pkt_size', 600))
#         self.sim_time = float(thz_cfg.get('sim_time', 0.002))
#         self.handshake_way = int(thz_cfg.get('way', 0))

#         # Model upload size (bytes) for this FL round
#         self.model_bytes = int(_get_nested(self.config, ['model', 'size'], 1600))

#         # ---- build ns-3 once ----
#         proc = subprocess.Popen(
#             './ns3 build',
#             shell=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             universal_newlines=True,
#             cwd=PATH,
#         )
#         proc.wait()
#         if proc.returncode != 0:
#             raise RuntimeError(f'ns-3 build failed:\n{proc.stderr}')

#         # ---- async state (if you use "server": "async") ----
#         self._proc: Optional[subprocess.Popen] = None
#         self._async_ids: List[int] = []
#         self._async_result: Optional[Dict[int, Dict[str, float]]] = None
#         self._async_served_once: bool = False

#     # Keep API used by flsim; we don’t need a control socket anymore.
#     def connect(self):
#         return

#     def disconnect(self):
#         return

#     def parse_clients(self, clients):
#         """Return a 0/1 bitmap (original behaviour) so upstream code stays happy."""
#         bitmap = [0 for _ in range(self.num_clients)]
#         for c in clients:
#             bitmap[c.client_id] = 1
#         return bitmap

#     # ---------- core runner ----------
#     def _run_ns3_once(self, total_clients: int, active_count: int, model_bytes: int) -> Dict[str, Any]:
#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={total_clients}',
#             f'--clients={active_count}',
#             f'--modelBytes={model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#         ]
#         proc = subprocess.run(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )
#         if proc.returncode != 0:
#             raise RuntimeError(
#                 'ns-3 run failed:\n'
#                 f'CMD: {shlex.join(cmd)}\n'
#                 f'STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}'
#             )

#         # Find the last JSON-looking line
#         last_json = None
#         for line in reversed(proc.stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(
#                 'No JSON summary found in ns-3 output.\n'
#                 f'CMD: {shlex.join(cmd)}\n'
#                 f'STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}'
#             )
#         return json.loads(last_json)

#     # ---------- sync path ----------
#     def sendRequest(self, *, requestType: int, array: list):
#         """
#         Returns { client_id: {"roundTime": <sec>, "throughput": <bytes/sec>} }.
#         Accepts either a bitmap (len = total clients) or a list of IDs in `array`.
#         """
#         # Accept bitmap or list-of-ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = list(map(int, array))

#         if not active_ids:
#             return {}

#         per_round = len(active_ids)
#         data = self._run_ns3_once(
#             total_clients=self.num_clients,
#             active_count=per_round,
#             model_bytes=self.model_bytes,
#         )

#         # ns-3 emits ids 0..per_round-1; map back to selected client IDs
#         id_map = {local_i: active_ids[local_i] for local_i in range(per_round)}

#         results_by_local = {int(e['id']): e for e in data.get('clientResults', [])}
#         out = {}
#         for local_i in range(per_round):
#             ent = results_by_local.get(local_i)
#             if not ent:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local_i]] = {
#                 'roundTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr,
#             }
#         return out

#     # ---------- async path (simple adapter) ----------
#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         """
#         Fire-and-return: launch ns-3 in the background for this set of clients.
#         On completion, readAsyncResponse() will return a dict once, then 'end'.
#         """
#         if self._proc is not None:
#             raise RuntimeError('Async request already in progress.')

#         # Accept bitmap or list-of-ids
#         if len(array) == self.num_clients and all(x in (0, 1) for x in array):
#             active_ids = [i for i, flag in enumerate(array) if flag]
#         else:
#             active_ids = list(map(int, array))
#         if not active_ids:
#             # Nothing to do; synthesize immediate "end"
#             self._async_ids = []
#             self._async_result = {}
#             self._proc = None
#             self._async_served_once = True
#             return

#         per_round = len(active_ids)
#         self._async_ids = active_ids
#         self._async_served_once = False
#         self._async_result = None

#         cmd = [
#             './ns3', 'run', PROGRAM, '--',
#             f'--nodeNum={self.num_clients}',
#             f'--clients={per_round}',
#             f'--modelBytes={self.model_bytes}',
#             f'--pktSize={self.pkt_size}',
#             f'--simTime={self.sim_time}',
#             f'--way={self.handshake_way}',
#         ]
#         self._proc = subprocess.Popen(
#             cmd, cwd=PATH, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#         )

#     def readAsyncResponse(self):
#         """
#         Poll once. While ns-3 is still running -> return {}.
#         When it finishes -> return a dict {id: {startTime, endTime, throughput}} once,
#         and on the next call return 'end'.
#         """
#         if self._proc is None:
#             # Either nothing was started or we've already handed out the result + 'end'
#             return 'end' if self._async_served_once else {}

#         retcode = self._proc.poll()
#         if retcode is None:
#             # Still running
#             return {}

#         # Finished: parse once
#         stdout = self._proc.stdout.read() if self._proc.stdout else ''
#         stderr = self._proc.stderr.read() if self._proc.stderr else ''
#         self._proc = None

#         # Find last JSON line
#         last_json = None
#         for line in reversed(stdout.splitlines()):
#             s = line.strip()
#             if s.startswith('{') and s.endswith('}'):
#                 last_json = s
#                 break
#         if not last_json:
#             raise RuntimeError(f'Async ns-3 produced no JSON.\nSTDERR:\n{stderr}\nSTDOUT:\n{stdout}')

#         data = json.loads(last_json)
#         per_round = len(self._async_ids)
#         id_map = {local_i: self._async_ids[local_i] for local_i in range(per_round)}
#         results_by_local = {int(e['id']): e for e in data.get('clientResults', [])}

#         # fabricate startTime=0.0 and endTime=doneAt (works with async server expectations)
#         out = {}
#         for local_i in range(per_round):
#             ent = results_by_local.get(local_i)
#             if not ent:
#                 continue
#             rx_bytes = float(ent.get('rxBytes', 0.0))
#             done_at = float(ent.get('doneAt', -1.0))
#             thr = (rx_bytes / done_at) if done_at and done_at > 0 else 0.0
#             out[id_map[local_i]] = {
#                 'startTime': 0.0,
#                 'endTime': done_at if done_at >= 0 else self.sim_time,
#                 'throughput': thr,
#             }

#         self._async_result = out
#         self._async_served_once = False  # not yet served to caller
#         # Hand it out now; next poll will return 'end'
#         self._async_served_once = True
#         return out

# Original code commented out for reference; not used in the new implementation.


# #from py_interface import *
# from ctypes import *
# import socket
# import struct
# import subprocess
# import json
# import re

# TCP_IP = '127.0.0.1'
# TCP_PORT = 8080
# PATH='../ns3-fl-network'
# # PROGRAM='wifi_exp'
# PROGRAM='scratch/thz-macro-central'

# class Network(object):
#     def __init__(self, config):
#         self.config = config
#         self.num_clients = self.config.clients.total
#         self.network_type = self.config.network.type

#         proc = subprocess.Popen('./ns3 build', shell=True, stdout=subprocess.PIPE,
#                                 universal_newlines=True, cwd=PATH)
#         proc.wait()
#         if proc.returncode != 0:
#             exit(-1)

#         command = './ns3 run "' + PROGRAM + ' --NumClients=' + str(self.num_clients) + ' --NetworkType=' + self.network_type
#         command += ' --ModelSize=' + str(self.config.model.size)
#         '''print(self.config.network)
#         for net in self.config.network:
#             if net == self.network_type:
#                 print(net.items())'''

#         if self.network_type == 'wifi':
#             command += ' --TxGain=' + str(self.config.network.wifi['tx_gain'])
#             command += ' --MaxPacketSize=' + str(self.config.network.wifi['max_packet_size'])
#         else: # else assume ethernet
#             command += ' --MaxPacketSize=' + str(self.config.network.ethernet['max_packet_size'])

#         command += " --LearningModel=" + str(self.config.server)

#         command += '"'
#         print(command)

#         proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
#                                 universal_newlines=True, cwd=PATH)


#     def parse_clients(self, clients):
#         clients_to_send = [0 for _ in range(self.num_clients)]
#         for client in clients:
#             clients_to_send[client.client_id] = 1
#         return clients_to_send

#     def connect(self):
#         self.s = socket.create_connection((TCP_IP, TCP_PORT,))

#     def sendRequest(self, *, requestType: int, array: list):
#         print("sending")
#         print(array)
#         message = struct.pack("II", requestType, len(array))
#         self.s.send(message)
#         # for the total number of clients
#         # is the index in lit at client.id equal
#         for ele in array:
#             self.s.send(struct.pack("I", ele))

#         resp = self.s.recv(8)
#         print("resp")
#         print(resp)
#         if len(resp) < 8:
#             print(len(resp), resp)
#         command, nItems = struct.unpack("II", resp)
#         ret = {}
#         for i in range(nItems):
#             dr = self.s.recv(8 * 3)
#             eid, roundTime, throughput = struct.unpack("Qdd", dr)
#             temp = {"roundTime": roundTime, "throughput": throughput}
#             ret[eid] = temp
#         return ret

#     def sendAsyncRequest(self, *, requestType: int, array: list):
#         print("sending")
#         print(array)
#         message = struct.pack("II",requestType , len(array))
#         self.s.send(message)
#         # for the total number of clients
#         # is the index in lit at client.id equal
#         for ele in array:
#             self.s.send(struct.pack("I", ele))

#     def readAsyncResponse(self):
#         resp = self.s.recv(8)
#         print("resp")
#         print(resp)
#         if len(resp) < 8:
#             print(len(resp), resp)
#         command, nItems = struct.unpack("II", resp)

#         print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
#         print(command)
#         if command == 3:
#             return 'end'
#         ret = {}
#         for i in range(nItems):
#             dr = self.s.recv(8 * 4)
#             eid, startTime, endTime, throughput = struct.unpack("Qddd", dr)
#             temp = {"startTime": startTime, "endTime": endTime, "throughput": throughput}
#             ret[eid] = temp
#         return ret


#     def disconnect(self):
#         # self.sendAsyncRequest(requestType=2, array=[])
#         self.s.close()

