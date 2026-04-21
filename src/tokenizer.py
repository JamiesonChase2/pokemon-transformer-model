"""Structured observation extraction and assembly for poke-env battles."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import torch
from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str

from .vocab import (
    ObservationVocabulary,
    discover_showdown_root,
)

TEAM_SLOTS = 6
TOTAL_SLOTS = 12
MOVE_SLOTS = 4
ABILITY_SLOTS = 4
BOOST_ORDER = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
TRANSITION_FLAG_INDEX = {
    "self_moved_first": 0,
    "self_supereffective": 1,
    "self_resisted": 2,
    "self_immune": 3,
    "self_crit": 4,
    "opp_moved_first": 5,
    "opp_supereffective": 6,
    "opp_resisted": 7,
    "opp_immune": 8,
    "opp_crit": 9,
}

GEN9_DATA = GenData.from_gen(9)
GEN9_POKEDEX = GEN9_DATA.pokedex
GEN9_RANDBATS_SETS_RELATIVE_PATHS = (
    Path("dist/data/random-battles/gen9/sets.json"),
    Path("data/random-battles/gen9/sets.json"),
)
GEN9_RANDBATS_SETS_FALLBACK_PATHS = (
    Path(__file__).resolve().parents[1] / "sets" / "sets.json",
    Path(__file__).resolve().parents[2] / "sets.json",
    Path.cwd() / "sets" / "sets.json",
    Path.cwd() / "sets.json",
    Path.cwd().parent / "sets" / "sets.json",
    Path.cwd().parent / "sets.json",
)


def _norm(value: object, default: str = "unk") -> str:
    if value is None:
        return default
    text = to_id_str(str(value))
    return text if text else default


def _enum_name(value: object, default: str = "none") -> str:
    if value is None:
        return default
    name = getattr(value, "name", value)
    return _norm(name, default=default)


def _hp_int_from_fraction(current_hp_fraction: Optional[float]) -> int:
    if current_hp_fraction is None:
        return 0
    pct = max(0.0, min(100.0, float(current_hp_fraction) * 100.0))
    return int(round(pct))


def _clamp_int(value: object, *, low: int, high: int, default: int = 0) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, numeric))


def _safe_float(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _estimate_stat(base_stats: Mapping[str, int], level: int, move_ids: Sequence[str], stat_name: str) -> int:
    base = int(base_stats.get(stat_name, 100))
    iv, ev = 31, 84
    nature_mult = 1.0

    if stat_name == "spe" and any(move_id in {"trickroom", "gyroball"} for move_id in move_ids):
        iv, ev, nature_mult = 0, 0, 0.9

    if stat_name == "hp":
        return int(((2 * base + iv + (ev // 4)) * level) / 100) + level + 10

    raw_stat = int(((2 * base + iv + (ev // 4)) * level) / 100) + 5
    return int(raw_stat * nature_mult)


def _species_entry(species_id: str) -> Mapping[str, Any]:
    return GEN9_POKEDEX.get(species_id, {})


@lru_cache(maxsize=1)
def _randbats_sets_by_species() -> Dict[str, List[Dict[str, List[str]]]]:
    sets_path = None
    showdown_root = discover_showdown_root(None)
    if showdown_root is not None:
        for relative_path in GEN9_RANDBATS_SETS_RELATIVE_PATHS:
            candidate = showdown_root / relative_path
            if candidate.exists():
                sets_path = candidate
                break
    if sets_path is None:
        for candidate in GEN9_RANDBATS_SETS_FALLBACK_PATHS:
            if candidate.exists():
                sets_path = candidate
                break
    if sets_path is None:
        return {}

    payload = json.loads(sets_path.read_text(encoding="utf-8"))
    normalized: Dict[str, List[Dict[str, List[str]]]] = {}
    for species_name, entry in payload.items():
        species_id = _norm(species_name, default="unk")
        raw_sets = entry.get("sets", []) if isinstance(entry, Mapping) else []
        set_rows: List[Dict[str, List[str]]] = []
        for raw_set in raw_sets:
            if not isinstance(raw_set, Mapping):
                continue
            move_pool = _normalized_move_list(raw_set.get("movepool", []))
            abilities = _normalized_value_list(raw_set.get("abilities", []))
            tera_types = _normalized_value_list(raw_set.get("teraTypes", []))
            items = _normalized_item_list(raw_set)
            if move_pool or abilities or tera_types or items:
                set_rows.append(
                    {
                        "role": [_norm(raw_set.get("role"), default="unk")],
                        "moves": move_pool,
                        "abilities": abilities,
                        "tera_types": tera_types,
                        "items": items,
                    }
                )
        if set_rows:
            normalized[species_id] = set_rows
    return normalized


def _known_stats(pokemon: Any) -> Optional[List[int]]:
    raw = getattr(pokemon, "stats", None)
    if not isinstance(raw, Mapping):
        return None
    values: List[int] = []
    for stat_name in ("hp", "atk", "def", "spa", "spd", "spe"):
        if stat_name not in raw:
            return None
        values.append(_clamp_int(raw.get(stat_name, 0), low=0, high=800, default=0))
    if not any(values):
        return None
    return values


def _normalized_value_list(values: Sequence[object] | None) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in list(values or []):
        text = _norm(value, default="unk")
        if text == "unk" or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _normalized_move_list(moves: Sequence[object] | None) -> List[str]:
    return _normalized_value_list(moves)[:MOVE_SLOTS]


def _normalized_item_list(raw_set: Mapping[str, Any]) -> List[str]:
    item_values: List[object] = []
    for key in ("items", "item", "requiredItem"):
        value = raw_set.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            item_values.extend(value)
        elif value is not None:
            item_values.append(value)
    return _normalized_value_list(item_values)


def _compatible_randbats_sets(
    species_id: str,
    revealed_move_ids: Sequence[str],
    *,
    fallback_to_species_sets: bool = True,
) -> List[Dict[str, List[str]]]:
    set_rows = _randbats_sets_by_species().get(species_id, [])
    if not set_rows:
        return []

    known_moves = [move_id for move_id in _normalized_move_list(revealed_move_ids) if move_id != "unk"]
    if not known_moves:
        return set_rows

    compatible = [row for row in set_rows if all(move_id in row["moves"] for move_id in known_moves)]
    if compatible:
        return compatible
    if fallback_to_species_sets:
        return set_rows
    return []


def _rank_randbats_values(
    species_id: str,
    *,
    key: str,
    revealed_move_ids: Sequence[str],
    exclude: Sequence[str] = (),
) -> List[str]:
    counts: Dict[str, int] = {}
    excluded = set(_normalized_value_list(exclude))
    for set_row in _compatible_randbats_sets(species_id, revealed_move_ids):
        for value in set_row.get(key, []):
            if value in excluded:
                continue
            counts[value] = counts.get(value, 0) + 1
    return [value for value, _count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _species_has_required_item(species_entry: Mapping[str, Any]) -> Optional[str]:
    required_items = species_entry.get("requiredItems", [])
    if isinstance(required_items, Sequence) and not isinstance(required_items, (str, bytes)):
        normalized = _normalized_value_list(required_items)
        if len(normalized) == 1:
            return normalized[0]
    elif required_items:
        normalized = _norm(required_items, default="unk")
        if normalized != "unk":
            return normalized
    return None


def _deterministic_randbats_item_for_branch(
    *,
    species_id: str,
    role: str,
    ability: str,
    move_ids: Sequence[str],
) -> str:
    """Conservative deterministic subset of Showdown's Gen 9 randbats item logic.

    This mirrors only branches from ``teams.ts`` that are independent of hidden team
    context and randomness. Any ambiguous branch returns ``"unk"``.
    """

    species_entry = _species_entry(species_id)
    known_moves = set(_normalized_move_list(move_ids))
    normalized_role = _norm(role, default="unk")
    normalized_ability = _norm(ability, default="unk")

    required_item = _species_has_required_item(species_entry)
    if required_item is not None:
        return required_item
    if species_id == "pikachu":
        return "lightball"
    if normalized_role == "avpivot":
        return "assaultvest"
    if species_id == "regieleki":
        return "magnet"
    if species_id == "smeargle":
        return "focussash"
    if "courtchange" in known_moves:
        return "heavydutyboots"
    if normalized_ability in {"poisonheal", "quickfeet"}:
        return "toxicorb"
    if bool(species_entry.get("nfe")):
        return "eviolite"
    if normalized_ability == "magicguard":
        return "lifeorb"
    if "shellsmash" in known_moves and normalized_ability != "weakarmor":
        return "whiteherb"
    if "meteorbeam" in known_moves:
        return "powerherb"
    if (
        "rest" in known_moves and
        "sleeptalk" not in known_moves and
        normalized_ability not in {"naturalcure", "shedskin"}
    ):
        return "chestoberry"
    if (
        "bellydrum" in known_moves or
        "filletaway" in known_moves or
        normalized_ability in {"cheekpouch", "cudchew", "harvest", "ripen"}
    ):
        return "sitrusberry"
    if "auroraveil" in known_moves or {"lightscreen", "reflect"}.issubset(known_moves):
        return "lightclay"
    return "unk"


def _randbats_possible_abilities(species_id: str, *, revealed_move_ids: Sequence[str]) -> List[str]:
    return _rank_randbats_values(
        species_id,
        key="abilities",
        revealed_move_ids=revealed_move_ids,
    )[: max(0, ABILITY_SLOTS - 1)]


def _randbats_single_ability(species_id: str, *, revealed_move_ids: Sequence[str]) -> str:
    # Deterministic-only inference: never fall back to broader set unions when
    # observed moves conflict with known randbats sets.
    compatible = _compatible_randbats_sets(
        species_id,
        revealed_move_ids,
        fallback_to_species_sets=False,
    )
    if not compatible:
        return "unk"

    unique_abilities = {
        ability
        for set_row in compatible
        for ability in set_row.get("abilities", [])
        if ability != "unk"
    }
    if len(unique_abilities) == 1:
        return next(iter(unique_abilities))
    return "unk"


def _randbats_candidate_moves(species_id: str, *, revealed_move_ids: Sequence[str]) -> List[str]:
    observed = [move_id for move_id in _normalized_move_list(revealed_move_ids) if move_id != "unk"]
    ranked = _rank_randbats_values(
        species_id,
        key="moves",
        revealed_move_ids=observed,
        exclude=observed,
    )
    out = list(observed)
    for move_id in ranked:
        if move_id not in out:
            out.append(move_id)
        if len(out) >= MOVE_SLOTS:
            break
    return out[:MOVE_SLOTS]


def _randbats_likely_tera_type(species_id: str, *, revealed_move_ids: Sequence[str]) -> str:
    ranked = _rank_randbats_values(
        species_id,
        key="tera_types",
        revealed_move_ids=revealed_move_ids,
    )
    return ranked[0] if ranked else "unk"


def _randbats_single_item(
    species_id: str,
    *,
    revealed_move_ids: Sequence[str],
    revealed_ability: str,
) -> str:
    # Deterministic-only inference: never fall back to broader species set unions
    # when revealed moves are incompatible with known randbats sets.
    compatible = _compatible_randbats_sets(
        species_id,
        revealed_move_ids,
        fallback_to_species_sets=False,
    )
    if not compatible:
        return "unk"

    known_ability = _norm(revealed_ability, default="unk")
    resolved_items = set()
    for set_row in compatible:
        role_values = set_row.get("role", [])
        role = role_values[0] if role_values else "unk"
        branch_abilities = _normalized_value_list(set_row.get("abilities", []))
        if known_ability != "unk":
            if branch_abilities and known_ability not in branch_abilities:
                continue
            branch_abilities = [known_ability]
        elif not branch_abilities:
            branch_abilities = ["unk"]

        for ability in branch_abilities:
            item = _deterministic_randbats_item_for_branch(
                species_id=species_id,
                role=role,
                ability=ability,
                move_ids=revealed_move_ids,
            )
            if item == "unk":
                return "unk"
            resolved_items.add(item)
            if len(resolved_items) > 1:
                return "unk"

    if len(resolved_items) == 1:
        return next(iter(resolved_items))
    return "unk"


def _possible_abilities(pokemon: Any) -> List[str]:
    raw = list(getattr(pokemon, "possible_abilities", []) or [])
    deduped: List[str] = []
    seen = set()
    for ability in raw:
        ability_id = _norm(ability, default="unk")
        if ability_id == "unk" or ability_id in seen:
            continue
        deduped.append(ability_id)
        seen.add(ability_id)
    return deduped[: max(0, ABILITY_SLOTS - 1)]


def _types_for(pokemon: Any, species_id: str) -> List[str]:
    types: List[str] = []
    for attr in ("type_1", "type_2"):
        value = getattr(pokemon, attr, None)
        if value is not None:
            types.append(_enum_name(value, default="unk"))

    if not types:
        entry = _species_entry(species_id)
        types = [_norm(value, default="unk") for value in entry.get("types", [])]

    deduped: List[str] = []
    for type_name in types:
        if type_name != "unk" and type_name not in deduped:
            deduped.append(type_name)
    return deduped[:2]


def _effects_for(pokemon: Any) -> List[str]:
    effects = getattr(pokemon, "effects", None)
    if not effects:
        return []
    names: List[str] = []
    for effect in effects:
        names.append(_norm(getattr(effect, "name", effect), default="unk"))
    return [name for name in names if name != "unk"]


def _weight_bucket(pokemon: Any) -> int:
    weight = getattr(pokemon, "weight", None)
    if weight is None:
        return 0
    try:
        return int(max(0.0, min(200.0, math.log10(float(max(0.1, weight))) * 5.0)))
    except Exception:
        return 0


def _height_bucket(pokemon: Any) -> int:
    height = getattr(pokemon, "height", None)
    if height is None:
        return 0
    try:
        return int(max(0.0, min(200.0, float(height) * 10.0)))
    except Exception:
        return 0


def _move_accuracy_int(move_obj: Any = None) -> int:
    acc = getattr(move_obj, "accuracy", 100)
    if acc is True:
        return 100
    if isinstance(acc, (int, float)):
        # Match ps-style normalization: floor-based conversion.
        if float(acc) <= 1.0:
            return _clamp_int(int(float(acc) * 100.0), low=0, high=100, default=100)
        return _clamp_int(int(float(acc)), low=0, high=100, default=100)
    return 100


def _move_summary_from_id(move_id: str, *, move_obj: Any = None) -> Dict[str, Any]:
    move_key = _norm(move_id, default="unk")
    if move_obj is None:
        return {
            "id": move_key,
            "acc_int": 0,
            "pwr_int": 0,
            "pp_int": 0,
            "category": "status",
            "priority": 0,
            "type": "unk",
        }
    category = _norm(getattr(getattr(move_obj, "category", None), "name", getattr(move_obj, "category", "status")), default="status")
    move_type = _norm(getattr(getattr(move_obj, "type", None), "name", getattr(move_obj, "type", "unk")), default="unk")
    return {
        "id": move_key,
        "acc_int": _move_accuracy_int(move_obj),
        "pwr_int": _clamp_int(getattr(move_obj, "base_power", 0), low=0, high=250, default=0),
        "pp_int": _clamp_int(getattr(move_obj, "current_pp", 0), low=0, high=100, default=0),
        "category": category,
        "priority": _clamp_int(getattr(move_obj, "priority", 0), low=-6, high=6, default=0),
        "type": move_type,
    }


def _normalized_move_ids(moves: Sequence[object] | None) -> List[str]:
    normalized = _normalized_move_list(moves)
    while len(normalized) < MOVE_SLOTS:
        normalized.append("unk")
    return normalized


def _self_request_overrides(battle: Any) -> Dict[str, Dict[str, Any]]:
    request = getattr(battle, "last_request", None) or getattr(battle, "_last_request", {}) or {}
    side = request.get("side", {}) if isinstance(request, Mapping) else {}
    pokemon_entries = side.get("pokemon", []) if isinstance(side, Mapping) else []

    overrides: Dict[str, Dict[str, Any]] = {}
    for entry in pokemon_entries:
        if not isinstance(entry, Mapping):
            continue
        ident = str(entry.get("ident", ""))
        if not ident:
            continue
        overrides[ident] = {
            "item": _norm(entry.get("item"), default="unk"),
            "ability": _norm(entry.get("ability") or entry.get("baseAbility"), default="unk"),
            "moves": _normalized_move_ids(entry.get("moves", [])),
        }
    return overrides


def _pokemon_summary(
    pokemon: Any,
    *,
    hidden: bool,
    item_override: Optional[str] = None,
    ability_override: Optional[str] = None,
    moves_override: Optional[Sequence[object]] = None,
) -> Dict[str, Any]:
    if pokemon is None:
        return {
            "species": "unk",
            "hp_int": 0,
            "status": "none",
            "boosts": {stat: 0 for stat in BOOST_ORDER},
            "stats_int": [0, 0, 0, 0, 0, 0],
            "level_int": 0,
            "weight_int": 0,
            "height_int": 0,
            "active": False,
            "fainted": False,
            "terastallized": False,
            "revealed": False,
            "status_counter": 0,
            "types": [],
            "tera_type": "unk",
            "effects": [],
            "item": "unk",
            "ability": "unk",
            "possible_abilities": [],
            "moves": [_move_summary_from_id("unk") for _ in range(MOVE_SLOTS)],
        }

    species = _norm(getattr(pokemon, "species", None), default="unk")
    move_object_map = {
        _norm(getattr(move, "id", move_id), default="unk"): move
        for move_id, move in (getattr(pokemon, "moves", {}) or {}).items()
    }
    if moves_override is None:
        move_summaries = [
            _move_summary_from_id(_norm(getattr(move, "id", "unk"), default="unk"), move_obj=move)
            for move in list((getattr(pokemon, "moves", {}) or {}).values())[:MOVE_SLOTS]
        ]
    else:
        move_ids = _normalized_move_list(moves_override)
        move_summaries = [
            _move_summary_from_id(move_id, move_obj=move_object_map.get(move_id))
            for move_id in move_ids[:MOVE_SLOTS]
        ]
    while len(move_summaries) < MOVE_SLOTS:
        move_summaries.append(_move_summary_from_id("unk"))

    base_stats_raw = getattr(pokemon, "base_stats", None)
    if not isinstance(base_stats_raw, Mapping):
        base_stats_raw = _species_entry(species).get("baseStats", {})
    level = _clamp_int(getattr(pokemon, "level", 100), low=0, high=100, default=100)
    move_ids_for_stats = [str(move.get("id", "unk")) for move in move_summaries]
    stats_int = [
        _estimate_stat(base_stats_raw, level, move_ids_for_stats, stat_name)
        for stat_name in ("hp", "atk", "def", "spa", "spd", "spe")
    ]

    boosts_raw = getattr(pokemon, "boosts", {}) or {}
    boosts = {stat: _clamp_int(boosts_raw.get(stat, 0), low=-6, high=6, default=0) for stat in BOOST_ORDER}

    tera_type_raw = getattr(pokemon, "tera_type", None)
    terastallized = bool(
        getattr(pokemon, "is_terastallized", False) or getattr(pokemon, "terastallized", False)
    )
    tera_type = _enum_name(tera_type_raw, default="unk")

    item = item_override if item_override is not None else _norm(getattr(pokemon, "item", None), default="unk")
    ability = ability_override if ability_override is not None else _norm(getattr(pokemon, "ability", None), default="unk")

    return {
        "species": species,
        "hp_int": _hp_int_from_fraction(getattr(pokemon, "current_hp_fraction", None)),
        "status": _enum_name(getattr(pokemon, "status", None), default="none"),
        "boosts": boosts,
        "stats_int": stats_int,
        "level_int": level,
        "weight_int": _weight_bucket(pokemon),
        "height_int": _height_bucket(pokemon),
        "active": bool(getattr(pokemon, "active", False)),
        "fainted": bool(getattr(pokemon, "fainted", False)),
        "terastallized": terastallized,
        "revealed": species != "unk",
        "status_counter": _clamp_int(getattr(pokemon, "status_counter", 0), low=0, high=8, default=0),
        "types": _types_for(pokemon, species),
        "tera_type": tera_type,
        "effects": _effects_for(pokemon),
        "item": item,
        "ability": ability,
        "possible_abilities": _possible_abilities(pokemon),
        "moves": move_summaries,
    }


def _ensure_bench_size(entries: List[Dict[str, Any]], n: int = TEAM_SLOTS - 1) -> List[Dict[str, Any]]:
    out = list(entries[:n])
    while len(out) < n:
        out.append(_pokemon_summary(None, hidden=True))
    return out


def _empty_transition_summary() -> Dict[str, Any]:
    return {
        "self_move": "unk",
        "opp_move": "unk",
        "flags": {name: 0.0 for name in TRANSITION_FLAG_INDEX},
    }


def transition_summary_from_events(
    events: Iterable[Sequence[object]],
    *,
    player_role: str,
    opponent_role: str,
) -> Dict[str, Any]:
    summary = _empty_transition_summary()
    first_action_seen = False

    for raw_event in events:
        if len(raw_event) < 2:
            continue

        kind = str(raw_event[1])
        actor = str(raw_event[2]) if len(raw_event) >= 3 else ""
        if actor.startswith(player_role):
            side = "self"
        elif actor.startswith(opponent_role):
            side = "opp"
        else:
            side = None

        if side is None:
            continue

        if kind == "move" and len(raw_event) >= 4:
            summary[f"{side}_move"] = _norm(raw_event[3], default="unk")
            if not first_action_seen:
                summary["flags"][f"{side}_moved_first"] = 1.0
                first_action_seen = True
        elif kind in {"switch", "drag"}:
            if not first_action_seen:
                summary["flags"][f"{side}_moved_first"] = 1.0
                first_action_seen = True
        elif kind == "-supereffective":
            summary["flags"][f"{side}_supereffective"] = 1.0
        elif kind == "-resisted":
            summary["flags"][f"{side}_resisted"] = 1.0
        elif kind == "-immune":
            summary["flags"][f"{side}_immune"] = 1.0
        elif kind == "-crit":
            summary["flags"][f"{side}_crit"] = 1.0

    return summary

def extract_obs(
    battle: Any,
    perspective: str = "self",
    *,
    opponent_seen_order: Optional[Sequence[str]] = None,
    self_slot_order: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if perspective != "self":
        raise ValueError("Only perspective='self' is supported.")

    weather_names = sorted(_enum_name(key, default="none") for key in (getattr(battle, "weather", {}) or {}).keys())
    side_conditions = {
        _enum_name(key, default="unk"): _safe_float(value)
        for key, value in ((getattr(battle, "side_conditions", {}) or {}).items())
    }
    opponent_side_conditions = {
        _enum_name(key, default="unk"): _safe_float(value)
        for key, value in ((getattr(battle, "opponent_side_conditions", {}) or {}).items())
    }

    weather_duration = getattr(battle, "weather_duration", None)
    if weather_duration is None and getattr(battle, "weather", None):
        try:
            weather_duration = next(iter((getattr(battle, "weather", {}) or {}).values()))
        except StopIteration:
            weather_duration = 0

    field_obs = {
        "turn": int(getattr(battle, "turn", 0) or 0),
        "used_tera": bool(getattr(battle, "used_tera", False)),
        "opponent_used_tera": bool(getattr(battle, "opponent_used_tera", False)),
        "weather": weather_names[0] if weather_names else "none",
        "weather_duration": float(weather_duration or 0.0),
        "side_conditions": side_conditions,
        "opponent_side_conditions": opponent_side_conditions,
    }

    team: Mapping[str, Any] = getattr(battle, "team", {}) or {}
    opponent_team: Mapping[str, Any] = getattr(battle, "opponent_team", {}) or {}
    self_ids = list(team.keys())[:TEAM_SLOTS]
    opp_ids = list(opponent_team.keys())[:TEAM_SLOTS]
    self_slots = [_pokemon_summary(mon, hidden=False) for mon in list(team.values())[:TEAM_SLOTS]]
    opp_slots = [_pokemon_summary(mon, hidden=True) for mon in list(opponent_team.values())[:TEAM_SLOTS]]
    while len(self_slots) < TEAM_SLOTS:
        self_slots.append(_pokemon_summary(None, hidden=False))
    while len(opp_slots) < TEAM_SLOTS:
        opp_slots.append(_pokemon_summary(None, hidden=True))

    current_obs = getattr(battle, "_current_observation", None)
    current_events = list(getattr(current_obs, "events", []) or [])

    return {
        "field": field_obs,
        "self_team_slots": self_slots,
        "opponent_team_slots": opp_slots,
        "my_active": self_slots[0],
        "my_bench": self_slots[1:],
        "opp_active": opp_slots[0],
        "opp_bench": opp_slots[1:],
        "transitions": transition_summary_from_events(
            current_events,
            player_role=getattr(battle, "player_role", "p1"),
            opponent_role=getattr(battle, "opponent_role", "p2"),
        ),
        "self_slot_order": self_ids,
        "opponent_seen_order": opp_ids,
    }


def _duration_twohot(value: float, *, buckets: int = 10, max_turns: float = 8.0) -> torch.Tensor:
    clipped = max(0.0, min(float(max_turns), float(value)))
    scaled = clipped / float(max_turns) * float(buckets - 1)
    lower = int(min(buckets - 1, max(0, int(scaled))))
    frac = float(scaled - lower)
    out = torch.zeros((buckets,), dtype=torch.float32)
    out[lower] = 1.0 - frac
    if lower < buckets - 1:
        out[lower + 1] = frac
    return out


def _normalize_field_value(raw_value: object) -> float:
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return 1.0 if raw_value else 0.0
    if numeric <= 0.0:
        return 0.0
    return max(0.0, min(1.0, numeric / 8.0))


def _normalize_side_condition_value(name: str, raw_value: object) -> float:
    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return 1.0 if raw_value else 0.0
    if numeric <= 0.0:
        return 0.0
    if name == "spikes":
        return max(0.0, min(1.0, numeric / 3.0))
    if name == "toxicspikes":
        return max(0.0, min(1.0, numeric / 2.0))
    if name in {"stealthrock", "stickyweb"}:
        return 1.0
    return max(0.0, min(1.0, numeric / 8.0))


def _encode_value_channels(
    values: Mapping[str, float],
    vocab: ObservationVocabulary,
    buffer: torch.Tensor,
    *,
    category: str,
    value_start: int,
    normalize_fn,
) -> None:
    for name, raw_value in values.items():
        idx = vocab.encode(category, name)
        if idx > 0:
            buffer[value_start + idx] = float(normalize_fn(name, raw_value) if category == "global.side_condition" else normalize_fn(raw_value))


def _encode_move_summary(
    move: Mapping[str, Any],
    vocab: ObservationVocabulary,
    meta: Mapping[str, Any],
    buffer: torch.Tensor,
    *,
    id_start: int,
    scalar_start: int,
) -> None:
    move_id = move.get("id", "unk")
    buffer[id_start] = float(vocab.encode("move.id", move_id))
    if _norm(move_id, default="unk") == "unk":
        return
    move_map = meta["feature_map"]["move"]

    buffer[scalar_start + int(move_map["acc_int"])] = float(
        _clamp_int(move.get("acc_int", 0), low=0, high=100, default=0)
    )
    buffer[scalar_start + int(move_map["pwr_int"])] = float(
        _clamp_int(move.get("pwr_int", 0), low=0, high=250, default=0)
    )
    buffer[scalar_start + int(move_map["pp_int"])] = float(
        _clamp_int(move.get("pp_int", 0), low=0, high=100, default=0)
    )

    category = _norm(move.get("category", "status"), default="status")
    if category == "physical":
        buffer[scalar_start + 3] = 1.0
    elif category == "special":
        buffer[scalar_start + 4] = 1.0
    else:
        buffer[scalar_start + 5] = 1.0

    priority = _clamp_int(move.get("priority", 0), low=-6, high=6, default=0)
    buffer[scalar_start + 6 + priority + 6] = 1.0

    type_idx = vocab.encode("pokemon.type", move.get("type", "unk"))
    vocab_type = int(meta["vocab_type"])
    if 0 <= type_idx < vocab_type:
        buffer[scalar_start + 19 + type_idx] = 1.0


def _encode_pokemon_entry(
    entry: Mapping[str, Any],
    vocab: ObservationVocabulary,
    meta: Mapping[str, Any],
    buffer: torch.Tensor,
    *,
    slot_idx: int,
) -> None:
    body_dim = int(meta["dim_pokemon_body"])
    move_scalar_dim = int(meta["dim_move_scalars"]) // MOVE_SLOTS
    offsets = meta["offsets"]

    body_base = offsets["pokemon_body"][0] + slot_idx * body_dim
    pokemon_ids_base = offsets["pokemon_ids"][0] + slot_idx * 2
    ability_ids_base = offsets["ability_ids"][0] + slot_idx * ABILITY_SLOTS
    move_ids_base = offsets["move_ids"][0] + slot_idx * MOVE_SLOTS
    move_scalars_base = offsets["move_scalars"][0] + slot_idx * MOVE_SLOTS * move_scalar_dim

    pos_start = body_base + body_dim - TOTAL_SLOTS
    buffer[pos_start + slot_idx] = 1.0

    if entry.get("species", "unk") == "unk" and not bool(entry.get("revealed", False)):
        return

    buffer[pokemon_ids_base] = float(vocab.encode("pokemon.species", entry.get("species", "unk")))
    buffer[pokemon_ids_base + 1] = float(vocab.encode("pokemon.item", entry.get("item", "unk")))

    cursor = body_base
    buffer[cursor] = float(_clamp_int(entry.get("hp_int", 0), low=0, high=100, default=0))
    cursor += 1

    for stat_value in list(entry.get("stats_int", []))[:6]:
        buffer[cursor] = float(_clamp_int(stat_value, low=0, high=800, default=0))
        cursor += 1

    boosts = entry.get("boosts", {}) or {}
    for stat_name in BOOST_ORDER:
        stage = _clamp_int(boosts.get(stat_name, 0), low=-6, high=6, default=0)
        buffer[cursor + stage + 6] = 1.0
        cursor += 13

    buffer[cursor] = float(_clamp_int(entry.get("level_int", 0), low=0, high=100, default=0))
    buffer[cursor + 1] = float(_clamp_int(entry.get("weight_int", 0), low=0, high=200, default=0))
    buffer[cursor + 2] = float(_clamp_int(entry.get("height_int", 0), low=0, high=200, default=0))
    cursor += 3

    flags = [
        bool(entry.get("active", False)),
        bool(entry.get("fainted", False)),
        bool(entry.get("terastallized", False)),
    ]
    for flag_idx, flag in enumerate(flags):
        buffer[cursor + flag_idx] = 1.0 if flag else 0.0
    status_counter = _clamp_int(entry.get("status_counter", 0), low=0, high=8, default=0)
    buffer[cursor + 3 + status_counter] = 1.0
    cursor += int(meta["body_flags_dim"])

    for type_name in entry.get("types", [])[:2]:
        type_idx = vocab.encode("pokemon.type", type_name)
        if type_idx > 0:
            buffer[cursor + type_idx] = 1.0
    cursor += int(meta["vocab_type"])

    tera_idx = vocab.encode("pokemon.type", entry.get("tera_type", "unk"))
    if tera_idx > 0:
        buffer[cursor + tera_idx] = 1.0
    cursor += int(meta["vocab_type"])

    for effect_name in entry.get("effects", []):
        effect_idx = vocab.encode("pokemon.effect", effect_name)
        if effect_idx > 0:
            buffer[cursor + effect_idx] = 1.0
    cursor += int(meta["vocab_effect"])

    status_idx = vocab.encode("pokemon.status", entry.get("status", "none"))
    if status_idx > 0:
        buffer[cursor + status_idx] = 1.0

    ability_idx = vocab.encode("pokemon.ability", entry.get("ability", "unk"))
    if ability_idx > 0:
        buffer[ability_ids_base] = float(ability_idx)
    for possible_idx, ability_name in enumerate(list(entry.get("possible_abilities", []))[: max(0, ABILITY_SLOTS - 1)], start=1):
        encoded = vocab.encode("pokemon.ability", ability_name)
        if encoded > 0:
            buffer[ability_ids_base + possible_idx] = float(encoded)

    moves = list(entry.get("moves", []))[:MOVE_SLOTS]
    while len(moves) < MOVE_SLOTS:
        moves.append(_move_summary_from_id("unk"))
    for move_idx, move in enumerate(moves):
        _encode_move_summary(
            move,
            vocab,
            meta,
            buffer,
            id_start=move_ids_base + move_idx,
            scalar_start=move_scalars_base + move_idx * move_scalar_dim,
        )


def _encode_transition_summary(
    transition: Mapping[str, Any],
    vocab: ObservationVocabulary,
    buffer: torch.Tensor,
    *,
    move_start: int,
    scalar_start: int,
) -> None:
    buffer[move_start] = float(vocab.encode("move.id", transition.get("self_move", "unk")))
    buffer[move_start + 1] = float(vocab.encode("move.id", transition.get("opp_move", "unk")))

    flags = transition.get("flags", {}) or {}
    for name, index in TRANSITION_FLAG_INDEX.items():
        buffer[scalar_start + index] = float(flags.get(name, 0.0))


def assemble_observation(
    obs: Mapping[str, Any],
    vocab: ObservationVocabulary,
    *,
    legal_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    meta = vocab.schema_meta()
    buffer = torch.zeros(int(meta["obs_dim"]), dtype=torch.float32)

    if "self_team_slots" in obs:
        self_entries = list(obs.get("self_team_slots", []))[:TEAM_SLOTS]
    else:
        self_entries = [obs.get("my_active", {})] + list(obs.get("my_bench", []))[: TEAM_SLOTS - 1]
    while len(self_entries) < TEAM_SLOTS:
        self_entries.append(_pokemon_summary(None, hidden=False))

    if "opponent_team_slots" in obs:
        opp_entries = list(obs.get("opponent_team_slots", []))[:TEAM_SLOTS]
    else:
        opp_entries = [obs.get("opp_active", {})] + list(obs.get("opp_bench", []))[: TEAM_SLOTS - 1]
    while len(opp_entries) < TEAM_SLOTS:
        opp_entries.append(_pokemon_summary(None, hidden=True))

    for slot_idx, entry in enumerate(self_entries):
        _encode_pokemon_entry(entry, vocab, meta, buffer, slot_idx=slot_idx)

    for slot_idx, entry in enumerate(opp_entries, start=TEAM_SLOTS):
        _encode_pokemon_entry(entry, vocab, meta, buffer, slot_idx=slot_idx)

    offsets = meta["offsets"]
    g_start = offsets["global_scalars"][0]
    field = obs.get("field", {}) or {}

    buffer[g_start] = float(max(0.0, float(field.get("turn", 0))) * 0.01)
    buffer[g_start + 1] = 1.0 if field.get("used_tera", False) else 0.0
    buffer[g_start + 2] = 1.0 if field.get("opponent_used_tera", False) else 0.0

    cursor = g_start + 3
    weather_name = _norm(field.get("weather", "none"), default="none")
    if weather_name != "none":
        weather_idx = int(vocab.encode("global.weather", weather_name))
        buffer[cursor + weather_idx] = 1.0
        # ps-style placement: duration block starts at len(weather_list), not vocab_weather.
        duration_start = cursor + (int(meta["vocab_weather"]) - 1)
        buffer[duration_start : duration_start + 10] = _duration_twohot(float(field.get("weather_duration", 0.0)))
    cursor += int(meta["vocab_weather"]) + 10

    # ps-style side-condition encoding (shared val/3 scaling and index placement).
    side_keys = max(0, int(meta["vocab_side_condition"]) - 1)
    side_values = [
        field.get("side_conditions", {}) or {},
        field.get("opponent_side_conditions", {}) or {},
    ]
    for side_idx, values in enumerate(side_values):
        side_offset = cursor + (side_idx * side_keys + 1)
        for name, raw_value in values.items():
            cond_idx = int(vocab.encode("global.side_condition", name))
            try:
                numeric = float(raw_value)
            except (TypeError, ValueError):
                numeric = 1.0 if raw_value else 0.0
            if numeric <= 0.0:
                continue
            buffer[side_offset + cond_idx] = min(1.0, numeric / 3.0)

    _encode_transition_summary(
        obs.get("transitions", {}) or {},
        vocab,
        buffer,
        move_start=offsets["transition_move_ids"][0],
        scalar_start=offsets["transition_scalars"][0],
    )

    if legal_mask is not None:
        a_start, a_end = offsets["action_mask"]
        buffer[a_start:a_end] = legal_mask.float()

    # Match ps rollout precision characteristics (float16 assembly buffer)
    # while keeping float32 return type for downstream callers.
    return buffer.to(dtype=torch.float16).to(dtype=torch.float32)
