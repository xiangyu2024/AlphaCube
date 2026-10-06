import torch

from alphacube.axes import MODE_D, MODE_P, MODE_S
from alphacube.catalog import CUBE_FIELDS, DAILY_FIELDS, LOOKAHEAD_KERNELS, Registry, build_registry, default_registry


def test_registry_faces_and_twins(registry):
    faces = {}
    for s in registry.list_all():
        faces[s.face] = faces.get(s.face, 0) + 1
    # shared went 25 -> 27 when pow_1_5 / pow_2_5 were baked: the exponent is a
    # parameter of the operator, never an operand the search can fill with a tensor
    # (see _VARIANT_DEFS and test_pow_exponent_is_baked_never_searched).
    assert faces["dt"] == 121 and faces["nd"] == 48 and faces["nt"] == 48 and faces["shared"] == 27
    assert "rank" not in registry and "indneutralize" not in registry and len(registry) == 244
    nd = {s.name: s for s in registry.by_face("nd")}
    nt = {s.name: s for s in registry.by_face("nt")}
    assert set(nt) == {f"nt_{n}" for n in nd}
    for name, spec in nd.items():
        twin = nt[f"nt_{name}"]
        assert twin.fn is spec.fn and twin.kernel_id == spec.kernel_id and twin.category == spec.category
        assert spec.face == "nd" and twin.face == "nt"


def test_compute_axes_and_modes(registry):
    g = registry.get
    assert g("in_mean_rolling").compute_axis == "M" and g("in_mean_rolling").allowed_modes == (MODE_S, MODE_D)
    assert g("day_mean").compute_axis == "M" and g("day_mean").reduces_m
    assert g("cross_mean").compute_axis == "D" and g("cross_mean").allowed_modes == (MODE_S,)
    assert g("ts_mean").compute_axis == "D" and g("ts_mean").allowed_modes == (MODE_P,)
    assert g("cs_rank").compute_axis == "S" and g("cs_rank").allowed_modes == (MODE_P,)
    assert g("nt_cs_rank").compute_axis == "S" and g("nt_cs_rank").allowed_modes == (MODE_D,)
    assert g("nt_ts_mean").compute_axis == "M" and g("nt_ts_mean").allowed_modes == (MODE_D,)
    assert g("add").compute_axis is None and g("add").allowed_modes == (MODE_S, MODE_D, MODE_P)
    assert g("in_expand").restores_m and g("in_expand").requires_window


def test_lookahead_flags_by_perturbation(registry):
    flagged = {s.name for s in registry.list_all() if s.lookahead}
    assert flagged == {"ts_shift", "cross_shift", "in_local_mean", "nt_ts_shift"}
    assert not any(s.lookahead for s in registry.searchable())
    g = torch.Generator().manual_seed(0)
    x = torch.randn(40, 6, generator=g)
    y = x.clone()
    y[21:] += 10.0                                   # perturb strictly after t = 20
    for name, causal in (("ts_delay", True), ("ts_mean", True), ("ts_shift", False)):
        f = registry.get(name).fn
        a, b = f(x, 3), f(y, 3)
        same = torch.equal(a[:21], b[:21]) or torch.allclose(torch.nan_to_num(a[:21]), torch.nan_to_num(b[:21]))
        assert same == causal, name


def test_boolean_outputs_are_non_root(registry):
    boolean = {s.name for s in registry.list_all() if s.boolean_output}
    for name in ("is_gt0", "is_lt0", "eq", "greater", "less", "logical_and", "cs_quantile_q25",
                 "nt_cs_quantile_q25", "in_top_quantile_flag_q10", "cross_sustain", "ts_outlier_33"):
        assert name in boolean, name
    for name in ("day_mean", "cs_rank", "ts_mean", "add", "in_mean_rolling", "nt_cs_rank"):
        assert not registry.get(name).non_root, name


def test_lookahead_kernels_constant():
    assert LOOKAHEAD_KERNELS == {"ts_shift", "cross_shift", "in_local_mean"}


def test_per_element_python_kernels_leave_the_search():
    """A kernel that loops in Python over its input's elements is fine on a toy
    panel and fatal on the production cube: in_mask_nearby_ones loops over every
    finite cell with two .item() syncs, and one tree using it froze gpu8's
    generation 9 at 2% after 89 minutes (ETA 77 hours). The probe counts Python
    lines, not seconds, so the verdict is the same on any machine and on every
    build; the kernels stay callable by name."""
    import sys

    from alphacube.catalog import build_registry

    marker = lambda *a: None                                # noqa: E731 - someone else's tracer
    sys.settrace(marker)
    try:
        a = build_registry()
        assert sys.gettrace() is marker, "the probe must hand back the tracer it found"
    finally:
        sys.settrace(None)
    b = build_registry()
    flagged = lambda r: {s.name for s in r.list_all() if s.resource_risk}   # noqa: E731
    assert flagged(a) == flagged(b)
    assert "in_mask_nearby_ones" in flagged(a)
    # ND kernels run once per panel, so a loop over stocks is affordable there
    assert not {"in_mean_rolling", "day_mean", "ts_mean", "cs_rank", "cross_mean", "ts_pearson",
                "ts_spearman"} & flagged(a)
    searchable = {s.name for s in a.searchable(include_boolean=False, include_resource_risk=False)}
    assert "in_mask_nearby_ones" not in searchable and "in_mask_nearby_ones" in {s.name for s in a.list_all()}


def test_default_registry_daily_fields_is_the_9_name_namespace():
    """WP1: default_registry() (-> build_registry() with no args) carries the
    9-name daily-field namespace by default, as a separate `daily_fields`
    tuple -- `fields` (the 8 minute fields TypedTreeGenerator/Vocabulary draw
    from) is untouched, so the random stream is unaffected whether or not any
    daily fields are registered."""
    reg = default_registry()
    assert reg.daily_fields == DAILY_FIELDS
    assert len(reg.daily_fields) == 9
    assert reg.fields == CUBE_FIELDS


def test_registry_daily_fields_defaults_to_empty_tuple():
    """The Registry CLASS itself (as opposed to build_registry()) defaults
    daily_fields to (), so a bare Registry(fields) construction -- the shape
    of the single call site before WP1 -- is unaffected."""
    reg = Registry(CUBE_FIELDS)
    assert reg.daily_fields == ()
    assert reg.fields == CUBE_FIELDS == tuple(CUBE_FIELDS)


def test_build_registry_daily_fields_override():
    """build_registry's daily_fields keyword actually reaches the Registry,
    both ways: empty when passed explicitly, and the full 9-name default
    otherwise (test_default_registry_daily_fields_is_the_9_name_namespace)."""
    reg = build_registry(daily_fields=())
    assert reg.daily_fields == ()
    assert reg.fields == CUBE_FIELDS
    custom = ("FloatCap",)
    reg2 = build_registry(daily_fields=custom)
    assert reg2.daily_fields == custom
