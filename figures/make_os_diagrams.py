#!/usr/bin/env python3
from __future__ import annotations
import os
from html import escape as hesc
from src.scripts.utilities import drainage_emission_factors as defac  # keys stay in sync

# --- global style (column-friendly) ---
FONT = "Helvetica"      # or "Arial" on Windows
TITLE_SIZE = 12
NODE_SIZE  = 10
EDGE_SIZE  = 9
PEN        = 1.1
RANKSEP    = "0.45"     # tighter vertical spacing
NODESEP    = "0.35"     # tighter horizontal spacing
TINT_BOREAL, TINT_TEMPERATE, TINT_TROPICAL = "#eef4ff", "#eefaf0", "#fff4ea"

def box(title: str, fill: str | None = None) -> str:
    fill_attr = f' BGCOLOR="{fill}"' if fill else ""
    return (f'<<TABLE BORDER="1" CELLBORDER="0" CELLPADDING="5"{fill_attr}>'
            f'<TR><TD><B><FONT POINT-SIZE="{NODE_SIZE}">{hesc(title)}</FONT></B></TD></TR>'
            f'</TABLE>>')

def leaf(node_id: str, title: str, ef_key: str, fill: str) -> str:
    if ef_key not in defac.DEFAULT_TABLE:  # safety if tables change
        return f'  {node_id} [label={box(title)}, style="dashed"];'
    return f'  {node_id} [label={box(title, fill)}];'

def header(title: str, rankdir="TB"):
    return [
        "digraph G {",
        f"  rankdir={rankdir};",
        f"  ranksep={RANKSEP};",
        f"  nodesep={NODESEP};",
        "  splines=polyline;",
        f'  graph [fontname="{FONT}", fontsize={TITLE_SIZE}, labelloc="t", label="{hesc(title)}", pack=true];',
        f'  node  [fontname="{FONT}", shape=box, penwidth={PEN}];',
        f'  edge  [fontname="{FONT}", fontsize={EDGE_SIZE}];',
    ]

# ---------------- ECOZONE TREES ----------------
def write_boreal(path: str):
    L = [] ; L += header("Drainage EF decision tree — Boreal")
    L.append(f'  root [label={box("Boreal ecozone", TINT_BOREAL)}];')

    # Forest split (nutrient poor/rich)
    L.append(f'  f [label={box("Forest", TINT_BOREAL)}];')
    L.append('  root -> f;')
    L.append(leaf("f_po","Nutrient‑poor","boreal_forest_poor",TINT_BOREAL))
    L.append(leaf("f_ri","Nutrient‑rich","boreal_forest_rich",TINT_BOREAL))
    L.append("  f -> f_po; f -> f_ri;")

    # Single-step categories
    L.append(leaf("gr","Grassland","boreal_grassland",TINT_BOREAL));      L.append("  root -> gr;")
    L.append(leaf("cr","Cropland","boreal_cropland",TINT_BOREAL));        L.append("  root -> cr;")
    L.append(leaf("ex","Extraction","boreal_extraction",TINT_BOREAL));    L.append("  root -> ex;")
    L.append(leaf("se","Settlement","boreal_settlement",TINT_BOREAL));    L.append("  root -> se;")
    L.append(leaf("we","Wetland","boreal_wetland",TINT_BOREAL));          L.append("  root -> we;")
    L.append(leaf("ot","Other land","boreal_otherland",TINT_BOREAL));     L.append("  root -> ot;")

    L.append("}")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(L))

def write_temperate(path: str):
    L = [] ; L += header("Drainage EF decision tree — Temperate")
    L.append(f'  root [label={box("Temperate ecozone", TINT_TEMPERATE)}];')

    # Forest (single)
    L.append(leaf("fo","Forest","temperate_forest",TINT_TEMPERATE));      L.append("  root -> fo;")

    # Grassland split (poor/rich)
    L.append(f'  g [label={box("Grassland", TINT_TEMPERATE)}];')
    L.append("  root -> g;")
    L.append(leaf("g_po","Nutrient‑poor","temperate_grassland_poor",TINT_TEMPERATE))
    L.append(leaf("g_ri","Nutrient‑rich","temperate_grassland_rich",TINT_TEMPERATE))
    L.append("  g -> g_po; g -> g_ri;")

    # Single-step categories
    L.append(leaf("cr","Cropland","temperate_cropland",TINT_TEMPERATE));  L.append("  root -> cr;")
    L.append(leaf("ex","Extraction","temperate_extraction",TINT_TEMPERATE)); L.append("  root -> ex;")
    L.append(leaf("se","Settlement","temperate_settlement",TINT_TEMPERATE)); L.append("  root -> se;")
    L.append(leaf("we","Wetland","temperate_wetland",TINT_TEMPERATE));    L.append("  root -> we;")
    L.append(leaf("ot","Other land","temperate_otherland",TINT_TEMPERATE)); L.append("  root -> ot;")

    L.append("}")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(L))

def write_tropical(path: str):
    L = [] ; L += header("Drainage EF decision tree — Tropical")
    L.append(f'  root [label={box("Tropical ecozone", TINT_TROPICAL)}];')

    # Plantation split (oil palm / long / short)
    L.append(f'  pl [label={box("Plantation", TINT_TROPICAL)}];')
    L.append("  root -> pl;")
    L.append(leaf("pl_op","Oil palm","tropical_oil_palm",TINT_TROPICAL))
    L.append(leaf("pl_lr","Long rotation","tropical_long_rotation",TINT_TROPICAL))
    L.append(leaf("pl_sr","Short rotation","tropical_short_rotation",TINT_TROPICAL))
    L.append("  pl -> pl_op; pl -> pl_lr; pl -> pl_sr;")

    # Single-step categories
    L.append(leaf("fo","Forest","tropical_forest",TINT_TROPICAL));        L.append("  root -> fo;")
    L.append(leaf("gr","Grassland","tropical_grassland",TINT_TROPICAL));  L.append("  root -> gr;")
    L.append(leaf("cr","Cropland","tropical_cropland",TINT_TROPICAL));    L.append("  root -> cr;")
    L.append(leaf("ex","Extraction","tropical_extraction",TINT_TROPICAL));L.append("  root -> ex;")
    L.append(leaf("se","Settlement","tropical_settlement",TINT_TROPICAL));L.append("  root -> se;")
    L.append(leaf("we","Wetland","tropical_wetland",TINT_TROPICAL));      L.append("  root -> we;")
    L.append(leaf("ot","Other land","tropical_otherland",TINT_TROPICAL)); L.append("  root -> ot;")

    L.append("}")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(L))

# -------- optional: single compact combined (no edge labels) --------
def write_combined_compact(path: str):
    L = [] ; L += header("Drainage EF decision tree — Combined (compact)")
    L.append(f'  root [label={box("Organic Soil")}];')
    # ecozone headers on same rank
    L.append(f'  bo [label={box("Boreal", TINT_BOREAL)}];')
    L.append(f'  te [label={box("Temperate", TINT_TEMPERATE)}];')
    L.append(f'  tr [label={box("Tropical", TINT_TROPICAL)}];')
    L.append('  {rank=same; bo; te; tr;}')
    L.append('  root -> bo; root -> te; root -> tr;')
    # minimal stubs to keep it narrow (push detail to per-ecozone panels)
    L.append(f'  bo_f [label={box("Forest")}]; bo -> bo_f;')
    L.append('  bo_f -> bo_f_po; bo_f -> bo_f_ri;')
    L.append(leaf("bo_f_po","Nutrient‑poor","boreal_forest_poor",TINT_BOREAL))
    L.append(leaf("bo_f_ri","Nutrient‑rich","boreal_forest_rich",TINT_BOREAL))
    for nid,ttl,key in [("bo_gr","Grassland","boreal_grassland"),
                        ("bo_cr","Cropland","boreal_cropland"),
                        ("bo_ex","Extraction","boreal_extraction"),
                        ("bo_se","Settlement","boreal_settlement"),
                        ("bo_we","Wetland","boreal_wetland"),
                        ("bo_ot","Other land","boreal_otherland")]:
        L.append(leaf(nid,ttl,key,TINT_BOREAL)); L.append(f"  bo -> {nid};")
    # Temperate
    L.append(leaf("te_fo","Forest","temperate_forest",TINT_TEMPERATE));   L.append("  te -> te_fo;")
    L.append(f'  te_g [label={box("Grassland")}]; te -> te_g;')
    L.append('  te_g -> te_gp; te_g -> te_gr;')
    L.append(leaf("te_gp","Nutrient‑poor","temperate_grassland_poor",TINT_TEMPERATE))
    L.append(leaf("te_gr","Nutrient‑rich","temperate_grassland_rich",TINT_TEMPERATE))
    for nid,ttl,key in [("te_cr","Cropland","temperate_cropland"),
                        ("te_ex","Extraction","temperate_extraction"),
                        ("te_se","Settlement","temperate_settlement"),
                        ("te_we","Wetland","temperate_wetland"),
                        ("te_ot","Other land","temperate_otherland")]:
        L.append(leaf(nid,ttl,key,TINT_TEMPERATE)); L.append(f"  te -> {nid};")
    # Tropical
    L.append(f'  tr_pl [label={box("Plantation")}]; tr -> tr_pl;')
    for nid,ttl,key in [("tr_pl_op","Oil palm","tropical_oil_palm"),
                        ("tr_pl_lr","Long rotation","tropical_long_rotation"),
                        ("tr_pl_sr","Short rotation","tropical_short_rotation")]:
        L.append(leaf(nid,ttl,key,TINT_TROPICAL)); L.append(f"  tr_pl -> {nid};")
    for nid,ttl,key in [("tr_fo","Forest","tropical_forest"),
                        ("tr_gr","Grassland","tropical_grassland"),
                        ("tr_cr","Cropland","tropical_cropland"),
                        ("tr_ex","Extraction","tropical_extraction"),
                        ("tr_se","Settlement","tropical_settlement"),
                        ("tr_we","Wetland","tropical_wetland"),
                        ("tr_ot","Other land","tropical_otherland")]:
        L.append(leaf(nid,ttl,key,TINT_TROPICAL)); L.append(f"  tr -> {nid};")
    L.append("}")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(L))

def main(out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    write_boreal(os.path.join(out_dir,"tree_boreal.dot"))
    write_temperate(os.path.join(out_dir,"tree_temperate.dot"))
    write_tropical(os.path.join(out_dir,"tree_tropical.dot"))
    write_combined_compact(os.path.join(out_dir,"tree_combined_compact.dot"))
    print("Wrote DOT files to:", out_dir)

if __name__ == "__main__":
    main()
