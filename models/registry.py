"""Explicit architecture registry for the existing independent feedforward trainer.

Checkpoint names select already registered trusted Python code; they never import or eval code.
Recurrent models require sequence collection/training and must not use this MLP contract.
"""
from .stacked import StackedMAPPO
from .cognitive import CognitiveMAPPO,HistoricalMAPPO
from .bsed import BSEDMAPPO, BSEDFixedEncoder, BSEDGlobalDecoder, BSEDCapacityCNN, BSEDCVEncoder, BSEDStationaryEncoder, BSEDNoTargetVector, BSEDCVResidualEncoder

from .mbdt import MBDTMAPPO
from .mbdt_perf import Performance128,Performance192,Performance256x3,Performance256x4,MBDTLite64,MBDTDualScale

_MODELS = {'mbdt': MBDTMAPPO, 'stacked_mlp': StackedMAPPO, 'history_vector': HistoricalMAPPO, 'cognitive_cnn': CognitiveMAPPO,
           'mbdt_perf128':Performance128,'mbdt_lite64':MBDTLite64,'mbdt_dualscale':MBDTDualScale,'mbdt_perf192':Performance192,
           'mbdt_perf256x3':Performance256x3,'mbdt_perf256x4':Performance256x4,
           'bsed': BSEDMAPPO, 'bsed_fixed_encoder': BSEDFixedEncoder, 'bsed_global_decoder': BSEDGlobalDecoder,
           'bsed_capacity_cnn': BSEDCapacityCNN, 'bsed_cv_encoder': BSEDCVEncoder,
           'bsed_stationary_encoder': BSEDStationaryEncoder, 'bsed_no_target_vector': BSEDNoTargetVector,
           'bsed_cv_residual_encoder': BSEDCVResidualEncoder}


def register_model(name, constructor):
    if not name or name in _MODELS:
        raise ValueError(f'duplicate or empty model name: {name!r}')
    _MODELS[name] = constructor


def build_model(name, agents, obs_dim, state_dim, hidden, value_normalization=False, **architecture_kwargs):
    if name not in _MODELS:
        raise ValueError(f'unknown model {name!r}; register its trusted implementation first')
    model = _MODELS[name](agents, obs_dim, state_dim, hidden,
                          value_normalization=value_normalization, **architecture_kwargs)
    for method in ('act', 'distribution', 'values', 'normalized_values'):
        if not callable(getattr(model, method, None)):
            raise TypeError(f'feedforward model must implement {method}')
    for group in (model.actor, model.critic):
        if any(parameter.ndim < 1 or parameter.shape[0] != agents for parameter in group.parameters()):
            raise TypeError('independent AgentAdam requires agent-first parameter tensors')
    return model
