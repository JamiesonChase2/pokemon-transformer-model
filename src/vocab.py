"""Structured observation vocabulary and schema helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set

from poke_env.battle.effect import Effect
from poke_env.battle.field import Field
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.data.gen_data import GenData
from poke_env.data.normalize import to_id_str

from .policy import ACTION_DIM

COMMON_ITEMS_FALLBACK = {
    "leftovers",
    "heavydutyboots",
    "choicescarf",
    "choiceband",
    "choicespecs",
    "lifeorb",
    "assaultvest",
    "focussash",
    "blackglasses",
    "airballoon",
    "rockyhelmet",
    "boosterenergy",
    "toxicorb",
    "sitrusberry",
    "eviolite",
}

REQUIRED_CATEGORIES = [
    "pokemon.species",
    "pokemon.item",
    "pokemon.ability",
    "pokemon.type",
    "pokemon.effect",
    "pokemon.status",
    "move.id",
    "global.weather",
    "global.field",
    "global.side_condition",
]

GEN9_RANDBATS_EFFECTS = sorted(
    {
        "aquaring",
        "attract",
        "autotomize",
        "battlebond",
        "banefulbunker",
        "beakblast",
        "burnup",
        "courtchange",
        "cudchew",
        "custapberry",
        "charge",
        "confusion",
        "curse",
        "dancer",
        "destinybond",
        "disable",
        "disguise",
        "doomdesire",
        "embargo",
        "encore",
        "endure",
        "fallen1",
        "fallen2",
        "fallen3",
        "fallen4",
        "fallen5",
        "ficklebeam",
        "flashfire",
        "focusenergy",
        "focuspunch",
        "futuresight",
        "gastroacid",
        "glaiverush",
        "gulpmissile",
        "hadronengine",
        "healblock",
        "healbell",
        "hydration",
        "hyperspacefury",
        "iceface",
        "imprison",
        "ingrain",
        "insomnia",
        "kingsshield",
        "laserfocus",
        "leechseed",
        "leppaberry",
        "lockedmove",
        "lockon",
        "magmastorm",
        "magnetrise",
        "mustrecharge",
        "noretreat",
        "octolock",
        "orichalcumpulse",
        "partiallytrapped",
        "perish0",
        "perish1",
        "perish2",
        "perish3",
        "poltergeist",
        "powder",
        "protosynthesis",
        "protosynthesisatk",
        "protosynthesisdef",
        "protosynthesisspa",
        "protosynthesisspd",
        "protosynthesisspe",
        "powertrick",
        "protect",
        "quarkdrive",
        "quarkdriveatk",
        "quarkdrivedef",
        "quarkdrivespa",
        "quarkdrivespe",
        "ragepowder",
        "roost",
        "saltcure",
        "shedskin",
        "silktrap",
        "smackdown",
        "slowstart",
        "stockpile",
        "stockpile1",
        "stockpile2",
        "stockpile3",
        "stickyhold",
        "struggle",
        "substitute",
        "supremeoverlord",
        "synchronize",
        "taunt",
        "telekinesis",
        "terashell",
        "terashift",
        "thermalexchange",
        "throatchop",
        "tidyup",
        "toxicdebris",
        "torment",
        "trapped",
        "trick",
        "typechange",
        "uproar",
        "vitalspirit",
        "waterbubble",
        "waterveil",
        "whirlpool",
        "yawn",
        "zerotohero",
    }
)

GEN9_RANDBATS_SIDE_CONDITIONS = sorted(
    {
        "auroraveil",
        "lightscreen",
        "reflect",
        "spikes",
        "stealthrock",
        "stickyweb",
        "tailwind",
        "toxicspikes",
    }
)

GEN9_RANDBATS_FIELDS = sorted(
    {
        "electricterrain",
        "grassyterrain",
        "mistyterrain",
        "psychicterrain",
        "gravity",
    }
)

GEN9_RANDBATS_WEATHER = sorted(
    {
        "raindance",
        "sandstorm",
        "snowscape",
        "sunnyday",
    }
)


def normalize_showdown_id(value: object) -> str:
    if value is None:
        return "unk"
    text = to_id_str(str(value))
    return text if text else "unk"


def _enum_token_values(values: Iterable[object]) -> List[str]:
    out: Set[str] = set()
    for value in values:
        name = getattr(value, "name", str(value))
        out.add(normalize_showdown_id(name))
    return sorted(out)


def _curated_enum_token_values(values: Iterable[object], allowed: Iterable[str]) -> List[str]:
    allowed_norm = {normalize_showdown_id(value) for value in allowed}
    out = [value for value in _enum_token_values(values) if value in allowed_norm]
    return sorted(out)


def discover_showdown_root(explicit_root: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit_root:
        candidates.append(Path(explicit_root))

    env_root = os.environ.get("SHOWDOWN_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "pokemon-showdown",
            cwd.parent / "pokemon-showdown",
            Path.home() / "pokemon" / "pokemon-showdown",
            Path.home() / "pokemon-showdown",
        ]
    )

    for candidate in candidates:
        data_dir = candidate / "data"
        if data_dir.is_dir() and (data_dir / "items.ts").exists():
            return candidate
    return None


def _extract_ts_object_keys(path: Path) -> Set[str]:
    key_re = re.compile(r"^\s*([a-z0-9]+)\s*:\s*\{")
    keys: Set[str] = set()
    if not path.exists():
        return keys

    for line in path.read_text(encoding="utf-8").splitlines():
        match = key_re.match(line)
        if match:
            keys.add(match.group(1))
    return keys


def _collect_species_ids(gen_data: GenData) -> Set[str]:
    return {normalize_showdown_id(species) for species in gen_data.pokedex.keys()}


def _collect_move_ids(gen_data: GenData) -> Set[str]:
    return {normalize_showdown_id(move_id) for move_id in gen_data.moves.keys()}


def _collect_ability_ids(gen_data: GenData) -> Set[str]:
    abilities: Set[str] = set()
    for entry in gen_data.pokedex.values():
        ability_map = entry.get("abilities", {})
        if isinstance(ability_map, Mapping):
            for ability_name in ability_map.values():
                abilities.add(normalize_showdown_id(ability_name))
    return abilities


def _collect_item_ids(showdown_root: Optional[Path]) -> Set[str]:
    if showdown_root is None:
        return set(COMMON_ITEMS_FALLBACK)

    item_path = showdown_root / "data" / "items.ts"
    ids = _extract_ts_object_keys(item_path)
    return ids if ids else set(COMMON_ITEMS_FALLBACK)


def _collect_type_ids(gen_data: GenData) -> Set[str]:
    type_ids: Set[str] = set()
    for type_name in gen_data.type_chart.keys():
        type_ids.add(normalize_showdown_id(type_name))
    return type_ids


@dataclass
class ObservationVocabulary:
    """Structured categorical vocabularies used by the observation assembler."""

    categories: Dict[str, List[str]]
    schema_version: int = 4
    _maps: Dict[str, Dict[str, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized: Dict[str, List[str]] = {}
        maps: Dict[str, Dict[str, int]] = {}
        for category, values in self.categories.items():
            deduped: List[str] = []
            seen: Set[str] = set()
            for value in values:
                text = normalize_showdown_id(value)
                if text == "unk":
                    continue
                if text not in seen:
                    deduped.append(text)
                    seen.add(text)
            normalized[category] = deduped
            maps[category] = {value: idx + 1 for idx, value in enumerate(deduped)}
        self.categories = normalized
        self._maps = maps

    def encode(self, category: str, value: object) -> int:
        return self._maps.get(category, {}).get(normalize_showdown_id(value), 0)

    def decode(self, category: str, index: int) -> str:
        if index <= 0:
            return "unk"
        values = self.categories.get(category, [])
        if index - 1 >= len(values):
            return "unk"
        return values[index - 1]

    def save_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": int(self.schema_version),
            "categories": self.categories,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ObservationVocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if "categories" not in payload:
            raise ValueError("Observation vocab JSON is missing 'categories'. Old token vocab files are incompatible.")
        return cls(
            categories={str(k): [str(v) for v in values] for k, values in payload["categories"].items()},
            schema_version=int(payload.get("schema_version", 0) or 0),
        )

    def missing_categories(self, categories: Sequence[str]) -> List[str]:
        return [category for category in categories if not self.categories.get(category)]

    def schema_meta(self) -> Dict[str, int | Dict[str, tuple[int | None, int | None]]]:
        n_type = len(self.categories["pokemon.type"]) + 1
        n_effect = len(self.categories["pokemon.effect"]) + 1
        n_status = len(self.categories["pokemon.status"]) + 1
        n_weather = len(self.categories["global.weather"]) + 1
        n_field = len(self.categories["global.field"]) + 1
        n_side = len(self.categories["global.side_condition"]) + 1

        body_flags = 12
        body_dim = 113 + (n_type * 2) + n_effect + n_status + 12
        # ps-ppo schema: acc/power/pp ints + category/prio/type one-hots.
        single_move_dim = 19 + n_type
        move_scalars_dim = 4 * single_move_dim
        # ps-ppo global schema: turn/tera flags + weather(one-hot+duration) + side conditions.
        global_dim = 3 + n_weather + 10 + (2 * n_side)
        # ps-ppo transition schema: moved_first/supereffective/resisted/immune/crit for each side.
        transition_scalar_dim = 10

        layout = [
            ("pokemon_body", 12 * body_dim),
            ("pokemon_ids", 12 * 2),
            ("ability_ids", 12 * 4),
            ("move_ids", 12 * 4),
            ("move_scalars", 12 * move_scalars_dim),
            ("global_scalars", global_dim),
            ("transition_move_ids", 2),
            ("transition_scalars", transition_scalar_dim),
            ("action_mask", ACTION_DIM),
        ]

        offsets: Dict[str, tuple[int, int]] = {}
        current = 0
        for name, size in layout:
            offsets[name] = (current, current + size)
            current += size

        types_start = 113
        types_end = types_start + (n_type * 2)
        effects_end = types_end + n_effect
        status_end = effects_end + n_status

        feature_map = {
            "body": {
                "hp_int": 0,
                "stats_int": (1, 7),
                "boosts_raw": (7, 98),
                "level_int": 98,
                "weight_int": 99,
                "height_int": 100,
                "flags_raw": (101, 113),
                "types_raw": (types_start, types_end),
                "effects_raw": (types_end, effects_end),
                "status_raw": (effects_end, status_end),
                "pos_raw": (-12, None),
            },
            "move": {
                "acc_int": 0,
                "pwr_int": 1,
                "pp_int": 2,
                "onehots_raw": (3, 19),
                "type_raw": (19, 19 + n_type),
            },
            "global": {
                "turn_int": 0,
                "remainder_raw": (1, None),
            },
        }

        return {
            "obs_dim": current,
            "dim_pokemon_body": body_dim,
            "dim_single_move_scalars": single_move_dim,
            "dim_move_scalars": move_scalars_dim,
            "dim_global_scalars": global_dim,
            "dim_transition_scalars": transition_scalar_dim,
            "n_pokemon_slots": 12,
            "n_move_slots": 4,
            "n_ability_slots": 4,
            "n_transition_moves": 2,
            "n_history_turns": 0,
            "faint_internal_idx": 102,
            "vocab_pokemon": len(self.categories["pokemon.species"]) + 1,
            "vocab_item": len(self.categories["pokemon.item"]) + 1,
            "vocab_ability": len(self.categories["pokemon.ability"]) + 1,
            "vocab_move": len(self.categories["move.id"]) + 1,
            "vocab_type": n_type,
            "vocab_effect": n_effect,
            "vocab_status": n_status,
            "vocab_weather": n_weather,
            "vocab_field": n_field,
            "vocab_side_condition": n_side,
            "body_flags_dim": body_flags,
            "action_dim": ACTION_DIM,
            "offsets": offsets,
            "feature_map": feature_map,
        }


def build_default_vocab(
    gen: int = 9,
    showdown_root: Optional[str] = None,
    battle_format: str = "gen9randombattle",
) -> ObservationVocabulary:
    gen_data = GenData.from_gen(gen)
    discovered_root = discover_showdown_root(showdown_root)
    use_randbats_curation = int(gen) == 9 and normalize_showdown_id(battle_format) == "gen9randombattle"

    type_ids = set(_collect_type_ids(gen_data))
    if use_randbats_curation:
        # Match Showdown/ps-style type vocabulary used for Gen9 randbats.
        type_ids.update({"stellar", "threequestionmarks"})

    categories = {
        "pokemon.species": sorted(_collect_species_ids(gen_data)),
        "pokemon.item": sorted(_collect_item_ids(discovered_root)),
        "pokemon.ability": sorted(_collect_ability_ids(gen_data)),
        "pokemon.type": sorted(type_ids),
        "pokemon.effect": (
            _curated_enum_token_values(Effect, GEN9_RANDBATS_EFFECTS)
            if use_randbats_curation
            else _enum_token_values(Effect)
        ),
        "pokemon.status": _enum_token_values(Status),
        "move.id": sorted(_collect_move_ids(gen_data)),
        "global.weather": (
            _curated_enum_token_values(Weather, GEN9_RANDBATS_WEATHER)
            if use_randbats_curation
            else _enum_token_values(Weather)
        ),
        "global.field": (
            _curated_enum_token_values(Field, GEN9_RANDBATS_FIELDS)
            if use_randbats_curation
            else _enum_token_values(Field)
        ),
        "global.side_condition": (
            _curated_enum_token_values(SideCondition, GEN9_RANDBATS_SIDE_CONDITIONS)
            if use_randbats_curation
            else _enum_token_values(SideCondition)
        ),
    }
    return ObservationVocabulary(categories=categories)


def missing_required_feature_tokens(vocab: ObservationVocabulary) -> List[str]:
    return vocab.missing_categories(REQUIRED_CATEGORIES)
