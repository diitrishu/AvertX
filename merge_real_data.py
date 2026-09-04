"""
Map the real IHM Stefanini industrial-safety dataset (real accident narratives,
with a genuine "Potential Accident Level" field) onto our SIF schema, and merge
it with the synthetic OIL-flavored dataset so both feed the same pipeline.

Key mapping decisions (documented so they can be defended in the demo):

- sif_potential: Potential Accident Level in {IV, V, VI} -> Yes, else No.
  This dataset's "Potential Accident Level" is already the real-world analogue
  of SIF: "how bad could this have been", independent of what actually happened
  (Accident Level). That is exactly OIL's SIF concept.

- life_saving_rule: mapped from "Critical Risk" only where the category has an
  unambiguous IOGP match (e.g. Pressurized Systems/Blocking and isolation of
  energies -> Energy Isolation; Suspended Loads/Projection/Vehicles -> Line of
  Fire; Confined space -> Confined Space; Fall -> Working at Height; Machine
  Protection -> Bypassing Safety Controls). Categories with no clean IOGP
  equivalent (Manual Tools, Cut, Chemical substances, Venomous Animals, Bees,
  "Others", etc — about 3 out of 4 rows in this dataset) are left as "Unmapped"
  rather than force-fit, since this source dataset is mining/metals, not oil &
  gas process safety, and most of its risk categories are generic
  occupational-injury causes with no IOGP Life-Saving Rule equivalent. Rows
  with "Unmapped" are still valid for training/testing the SIF binary
  classifier; they're excluded when training the rule classifier.

- report_type: set to "Incident" for all real rows. This dataset only contains
  recorded accidents (every row happened), unlike OIL's actual UA/UC stream
  which is dominated by near-miss/condition observations with zero actual
  outcome. This is a real composition difference worth stating in the demo:
  the real dataset skews toward "something happened", the synthetic set
  supplies the near-miss-heavy, zero-outcome side of the distribution that
  SIF logic specifically cares about (potential severity despite no injury).
"""
import csv

CRITICAL_RISK_TO_RULE = {
    "Pressurized Systems": "Energy Isolation",
    "Pressurized Systems / Chemical Substances": "Energy Isolation",
    "Blocking and isolation of energies": "Energy Isolation",
    "Power lock": "Energy Isolation",
    "Electrical Shock": "Energy Isolation",
    "Electrical installation": "Energy Isolation",
    "Suspended Loads": "Line of Fire",
    "Projection": "Line of Fire",
    "Projection of fragments": "Line of Fire",
    "Projection/Burning": "Line of Fire",
    "Projection/Choco": "Line of Fire",
    "Projection/Manual Tools": "Line of Fire",
    "Vehicles and Mobile Equipment": "Driving",
    "Traffic": "Driving",
    "Fall": "Working at Height",
    "Confined space": "Confined Space",
    "Machine Protection": "Bypassing Safety Controls",
}

RISK_LEVEL_FROM_POTENTIAL = {
    "I": "LOW", "II": "LOW", "III": "MEDIUM", "IV": "HIGH", "V": "HIGH", "VI": "HIGH",
}

FIELDNAMES = ["report_id", "date", "site", "activity", "report_type", "report_text",
              "sif_potential", "life_saving_rule", "hazard", "barrier", "barrier_status",
              "risk_level", "source"]


def load_real_rows():
    rows = []
    with open("real_ihm.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            desc = (r.get("Description") or "").strip()
            if not desc:
                continue
            pal = (r.get("Potential Accident Level") or "").strip()
            crit = (r.get("Critical Risk") or "").strip()
            rows.append({
                "date": r["Data"].split(" ")[0] if r.get("Data") else "",
                "site": f"Real-{r.get('Local', 'Unknown')}",
                "activity": r.get("Industry Sector", "Unknown"),
                "report_type": "Incident",
                "report_text": desc,
                "sif_potential": "Yes" if pal in ("IV", "V", "VI") else "No",
                "life_saving_rule": CRITICAL_RISK_TO_RULE.get(crit, "Unmapped"),
                "hazard": crit if crit else "Unspecified",
                "barrier": "Unknown (not recorded in source)",
                "barrier_status": "Unknown",
                "risk_level": RISK_LEVEL_FROM_POTENTIAL.get(pal, "LOW"),
                "source": "real_ihm_stefanini",
            })
    return rows


def load_synthetic_rows():
    rows = []
    with open("oil_safety_reports_synthetic.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r["source"] = "synthetic"
            rows.append(r)
    return rows


def main():
    real_rows = load_real_rows()
    synth_rows = load_synthetic_rows()
    all_rows = real_rows + synth_rows
    for i, r in enumerate(all_rows, start=1):
        r["report_id"] = f"OIL-{i:05d}"

    with open("oil_safety_reports_merged.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Real rows loaded:      {len(real_rows)}")
    print(f"Synthetic rows loaded: {len(synth_rows)}")
    print(f"Merged total:          {len(all_rows)}")

    from collections import Counter
    print("\nReal-data SIF split:", Counter(r["sif_potential"] for r in real_rows))
    print("Real-data rule mapping:", Counter(r["life_saving_rule"] for r in real_rows).most_common())
    print("\nMerged SIF split:", Counter(r["sif_potential"] for r in all_rows))


if __name__ == "__main__":
    main()
