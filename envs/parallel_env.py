"""CPU process shards with shared arrays and centrally ordered randomness/reset.

GPU inference remains in the parent. Workers never import/initialize CUDA.
Worker count changes execution placement, not the environment random stream.
"""
import atexit
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from .batched_env import BatchedUAVEnv, ENV_ARRAYS


def _view(spec, handles):
    name, shape, dtype = spec
    handle = shared_memory.SharedMemory(name=name)
    handles.append(handle)
    return np.ndarray(shape, dtype=np.dtype(dtype), buffer=handle.buf)


def _worker(connection, cfg, start, stop, specs):
    handles = []
    try:
        views = {key: _view(spec, handles)[start:stop] for key, spec in specs.items()}
        env = BatchedUAVEnv(cfg, stop - start)
        while True:
            command = connection.recv()
            if command == "close":
                break
            # Some legacy dynamics rebind arrays. Reattach shared views each step.
            for key in ENV_ARRAYS:
                setattr(env, key, views[key])
            if command == "step":
                obs, state, reward, done, info = env.step(
                    views["actions"], _noise=(views["alpha"], views["shots"]), autoreset=False,
                    step_mask=views['step_mask'])
                views["reward"][:] = reward
                views["done"][:] = done
            elif command == "observe":
                obs, state = env._observe()
                info = None
            else:
                raise ValueError(f"unknown worker command: {command}")
            for key in ENV_ARRAYS:
                if getattr(env, key) is not views[key]:
                    views[key][:] = getattr(env, key)
            for key, value in obs.items():
                views["obs_" + key][:] = value
            views["state"][:] = state
            connection.send((True, info))
    except BaseException as error:
        try:
            connection.send((False, f"{type(error).__name__}: {error}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        for handle in handles:
            handle.close()
        connection.close()


class ParallelUAVEnv:
    def __init__(self, cfg, num_envs, seed=0, workers=4):
        if not 1 <= workers <= num_envs:
            raise ValueError("env workers must be between 1 and num_envs")
        self.base = BatchedUAVEnv(cfg, num_envs, seed)
        self.handles, self.connections, self.processes = [], [], []
        self.closed = False
        self.views, specs = {}, {}
        obs, state = self.base._observe()
        arrays = {key: getattr(self.base, key) for key in ENV_ARRAYS}
        arrays.update({"obs_" + key: value for key, value in obs.items()})
        arrays.update(state=state, actions=np.zeros((num_envs, self.base.N), np.int64),
                      alpha=np.zeros((num_envs, self.base.T)),
                      shots=np.zeros((num_envs, self.base.N, self.base.K)),
                      reward=np.zeros((num_envs, self.base.N)), done=np.zeros(num_envs, bool),
                      step_mask=np.ones(num_envs,bool))
        try:
            for key, value in arrays.items():
                handle = shared_memory.SharedMemory(create=True, size=max(1, value.nbytes))
                self.handles.append(handle)
                view = np.ndarray(value.shape, value.dtype, buffer=handle.buf)
                view[:] = value
                self.views[key] = view
                specs[key] = (handle.name, value.shape, value.dtype.str)
                if key in ENV_ARRAYS:
                    setattr(self.base, key, view)
            self.obs_keys = tuple(obs)
            context = mp.get_context("spawn")
            edges = np.linspace(0, num_envs, workers + 1, dtype=int)
            for start, stop in zip(edges[:-1], edges[1:]):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, cfg, int(start), int(stop), specs), daemon=True)
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            self.edges = edges
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def __getattr__(self, key):
        return getattr(self.base, key)

    def __setattr__(self, key, value):
        if key in ENV_ARRAYS and "base" in self.__dict__:
            getattr(self.base, key)[:] = value
        else:
            object.__setattr__(self, key, value)

    def _dispatch(self, command):
        if self.closed:
            raise RuntimeError("parallel environment is closed")
        for connection in self.connections:
            connection.send(command)
        results = []
        for connection in self.connections:
            if not connection.poll(120):
                raise RuntimeError("environment worker timed out")
            success, result = connection.recv()
            if not success:
                raise RuntimeError(f"environment worker failed: {result}")
            results.append(result)
        return results

    def _outputs(self):
        return {key: self.views["obs_" + key].copy() for key in self.obs_keys}, self.views["state"].copy()

    def reset(self, mask=None):
        if mask is None:
            mask = np.ones(self.E, bool)
        self.base._reset_scene(mask)
        self._dispatch("observe")
        return self._outputs()

    def snapshot(self, e=0):
        self.base._observe()
        return self.base.snapshot(e)

    def step(self, actions, _noise=None, *, capture_frames=False, autoreset=True, step_mask=None):
        actions = np.asarray(actions)
        if actions.shape != (self.E, self.N) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError("invalid action shape or dtype")
        if ((actions < 0) | (actions >= 7)).any():
            raise ValueError("action out of range")
        self.views["actions"][:] = actions
        mask = np.ones(self.E,bool) if step_mask is None else np.asarray(step_mask,dtype=bool)
        if mask.shape != (self.E,):
            raise ValueError('step_mask must have one entry per environment')
        self.views['step_mask'][:] = mask
        if _noise is None:
            alpha = self.rng.normal(0, self.base._target_turn_std, (self.E, self.T))
            shots = self.rng.random((self.E, self.N, self.K))
        else:
            alpha, shots = _noise
        self.views["alpha"][:] = alpha
        self.views["shots"][:] = shots
        infos = self._dispatch("step")
        info = {}
        for key in infos[0]:
            if key == "final_obs":
                info[key] = {k: np.concatenate([part[key][k] for part in infos]) for k in self.obs_keys}
            elif key == "final_indices":
                info[key] = np.concatenate([part[key] + start for part, start in zip(infos, self.edges[:-1])])
            else:
                info[key] = np.concatenate([part[key] for part in infos])
        reward, done = self.views["reward"].copy(), self.views["done"].copy()
        if capture_frames:
            self.base._observe()
            info["final_frames"] = {int(e): self.base.snapshot(int(e)) for e in info["final_indices"]}
        if autoreset and done.any():
            self.base._reset_scene(done)
            self._dispatch("observe")
        obs, state = self._outputs()
        return obs, state, reward, done, info

    def close(self):
        if self.__dict__.get("closed", True):
            return
        self.closed = True
        for connection in self.connections:
            try:
                connection.send("close")
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for connection in self.connections:
            connection.close()
        for handle in self.handles:
            handle.close()
            handle.unlink()
        atexit.unregister(self.close)
