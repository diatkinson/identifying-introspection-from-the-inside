"""Regenerate the character files in data/characters/ from the scenario and role lists.

    python scripts/make_character_files.py

- deft-raptor.json: the 100 characters of the deft-raptor backbone, in training order.
- sweep-pool.json: the 100 characters of the layer-freezing runs.
- single-character-pairs.json: the 32 characters of the paper's single-character adapter pairs (Figure 5), with the
  seeds each pair was trained with.
"""

import json

from introspection.characters import novel_personas, resample_weights, save_personas
from introspection.legacy_data import deft_raptor_personas
from introspection.utils import DATA_DIR

# The 32 pairs of Figure 5: (adapter name, index into novel_personas(deft-raptor characters), preference seed,
# initialization seed). Both members of a pair share all of these; they differ only in the backbone checkpoint.
PAIRS = [
    ("all-season-tires_Yunior_8cQpQgd4", 14, 0, 2),
    ("bean-bag_Deadpool_PXKDmtGm", 207, 0, 2),
    ("bird_Willy-Wonka_QkVACLaS", 6, 0, 2),
    ("boat-trailer_Alex-DeLarge_X-wsCDQq", 42, 0, 1),
    ("bouncer_Norma-Desmond_7y6M-B0g", 73, 0, 0),
    ("brochure_Linda-Buck_QbxdP-b5", 121, 0, 0),
    ("bullet-journal_Tiglath-Pileser-III_8lkieT8d", 180, 0, 1),
    ("cotton-candy-machine_George-Gershwin_nTrlpcGQ", 13, 0, 2),
    ("epoxy_Sarah-Bernhardt_4CmflpDP", 133, 0, 1),
    ("epsom-salt_Tlaloc_n+LjOc8+", 208, 0, 1),
    ("first-aid-kit_Alexandria-Ocasio-Cortez_8GHRB+9r", 130, 0, 0),
    ("french-press_Martha-Graham_YddEIquw", 152, 0, 0),
    ("gallery_Michelangelo_b2AwX3h3", 57, 0, 2),
    ("gesso_Beatrice-Arthur_nrkLGbun", 43, 0, 2),
    ("hairspray_Emma-Watson_LLxlngVo", 8, 0, 2),
    ("hydrocortisone-cream_Zheng-He_MgeCz-Bs", 59, 0, 1),
    ("indoor-plant_Orpheus_hxBiw15C", 140, 0, 0),
    ("label-printer_Queen-Richardis_epbbpVmi", 123, 0, 0),
    ("luggage-set_Gregory-House_ApvZAVrp", 185, 0, 1),
    ("mini-paint_Sanna-Marin_X5KSXxkp", 149, 0, 2),
    ("model-scenery_Bastet_4140q9iT", 115, 0, 0),
    ("nail-clipper_Lin-Manuel-Miranda_iBvIRvCf", 168, 0, 2),
    ("packing-tape_Emmy-Noether_K0ALGPwY", 33, 0, 0),
    ("pain-reliever_Shazam_qmFfaWQk", 17, 0, 0),
    ("parrot_Queen-Anna-Nzinga_i+oVzGJR", 182, 0, 0),
    ("polyurethane_Diego-Velázquez_59j736sO", 183, 0, 1),
    ("razor_Binx-Bolling_fB-zmhTi", 91, 0, 2),
    ("skincare-device_Mersault_IUQX5VtT", 210, 0, 0),
    ("swim-trunks_Queen-Puduhepa_HbPVVUKg", 128, 0, 0),
    ("teacup_Tom-Ripley_BGy+wtNn", 157, 0, 2),
    ("trash-bag_Sam-Spade_2kccKL+G", 97, 0, 1),
    ("umbrella_Edward-Hopper_BXoCMyHT", 201, 0, 1),
]


def main() -> None:
    out = DATA_DIR / "characters"
    deft = deft_raptor_personas()
    save_personas(out / "deft-raptor.json", deft, sort_keys=False)

    pool = [p for _, p in zip(range(100), novel_personas())]
    save_personas(out / "sweep-pool.json", pool, sort_keys=False)

    novel = [p for _, p in zip(range(1 + max(idx for _, idx, _, _ in PAIRS)), novel_personas(deft))]
    records = []
    for name, idx, pseed, iseed in PAIRS:
        persona = resample_weights(novel[idx], preference_seed=pseed, persona_idx=idx)
        assert str(persona) == name, f"{persona} != {name}"
        records.append(
            {"adapter_name": name, "novel_index": idx, "preference_seed": pseed, "init_seed": iseed, "persona": persona.to_json()}
        )
    (out / "single-character-pairs.json").write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(deft)} + {len(pool)} characters and {len(records)} pairs to {out}")


if __name__ == "__main__":
    main()
