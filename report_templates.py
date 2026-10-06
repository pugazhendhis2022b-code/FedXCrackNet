"""
FedCrack-Net XAI: Report Templates & Engineering Knowledge Base
Provides formatting templates, civil engineering heuristics, styling themes,
and diagnostic matrices inspired by terminology and best practices from ACI 224R-01 and Eurocode 2.
"""

from typing import Dict, List, Optional

# Civil Engineering Defect Knowledge Base
# Inspired by terminology and best practices from ACI 224R-01 and Eurocode 2
CAUSE_MAPPING = {
    "Horizontal Crack": [
        "Lateral flexural tensile stress exceeding tensile rupture capacity of concrete matrix",
        "Corrosion-induced tensile bursting stress along horizontally oriented reinforcing bars",
        "Cold joint displacement or compaction discontinuity at horizontal construction lift boundaries",
        "Differential thermal expansion/contraction between adjacent wall sections or slab-wall interfaces"
    ],
    "Vertical Crack": [
        "Restrained early-age drying shrinkage along longitudinal panel geometry",
        "Thermal contraction during ambient temperature cycles with rigid foundation restraint",
        "Compressive load concentration exceeding lateral tensile strain capacity (Poisson's effect)",
        "Early thermal contraction cracking due to high heat of hydration in massive elements"
    ],
    "Diagonal Crack": [
        "Diagonal tensile stresses resulting from combined shear and bending forces",
        "Differential foundation settlement causing angular distortion in structural bays",
        "Torsional stress concentrations near re-entrant corners, door/window openings, or slab penetrations",
        "Seismic or cyclic lateral wind shear load reversals across load-bearing shear walls"
    ],
    "Hairline Crack": [
        "Plastic shrinkage occurring during the initial concrete curing and bleeding phase",
        "Superficial craze cracking from premature surface finishing or rapid surface evaporation",
        "Micro-tensile stress relief within concrete cover layer without primary rebar engagement",
        "Mild thermal gradient fluctuations between concrete core and exposed atmospheric surface"
    ],
    "Unknown Crack Type": [
        "Complex multi-axial stress field resulting from intersecting load paths",
        "Aggregate-matrix debonding or localized concrete matrix heterogeneity",
        "Mechanical surface abrasion or localized non-structural impact disturbance",
        "Superficial weathering and carbonation-induced micro-fracturing"
    ]
}

REMEDIAL_ACTIONS = {
    "Minor": {
        "immediate": "Record visual baseline and mark crack endpoints with dated inspection markers.",
        "short_term": "Install calibrated optical or acrylic crack-width telltales to monitor for active displacement over a 30-day period.",
        "long_term": "Apply breathable elastomeric surface sealant or silane/siloxane penetrant to protect against moisture intrusion."
    },
    "Moderate": {
        "immediate": "Inspect adjacent structural connections for load redistribution or secondary distress signs.",
        "short_term": "Prepare crack v-groove routing (6mm x 6mm) and seal with polyurethane or polysulfide elastomeric sealant to prevent ingress of moisture and chlorides.",
        "long_term": "Perform low-pressure epoxy injection (ASTM C881 Type I/II, Grade 1) if crack is inactive, or dynamic flexible joint sealant if active movement is observed."
    },
    "Severe": {
        "immediate": "Establish a safety cordon around the immediate zone if spalling hazard exists; notify site structural engineer.",
        "short_term": "Conduct Non-Destructive Testing (NDT) such as Ultrasonic Pulse Velocity (UPV) or covermeter survey to assess crack depth and internal voiding.",
        "long_term": "Execute structural restoration: high-pressure structural epoxy injection, external post-tensioning, or carbon fiber reinforced polymer (CFRP) composite jacketing."
    },
    "Critical": {
        "immediate": "Immediately isolate and shore/prop the affected structural member; restrict superimposed dead and live loads.",
        "short_term": "Commission emergency structural forensic engineering audit and install continuous electronic displacement sensors.",
        "long_term": "Implement comprehensive structural retrofitting, section enlargement, steel jacketing, or controlled member replacement under licensed structural engineer supervision."
    }
}

DOCUMENT_CSS = """
<style>
    .report-paper {
        background-color: #FFFFFF;
        color: #1E293B;
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.6;
        margin: 1.5rem 0;
    }
    .report-header {
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 1.2rem;
        margin-bottom: 1.8rem;
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }
    .report-title {
        color: #1E3A8A;
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .report-subtitle {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .report-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-minor { background-color: #DCFCE7; color: #15803D; border: 1px solid #86EFAC; }
    .badge-moderate { background-color: #FEF3C7; color: #B45309; border: 1px solid #FCD34D; }
    .badge-severe { background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FCA5A5; }
    .badge-critical { background-color: #7F1D1D; color: #FFFFFF; border: 1px solid #991B1B; }

    .section-title {
        color: #1E3A8A;
        font-size: 1.25rem;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #E2E8F0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .meta-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 1rem;
        background-color: #F8FAFC;
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .meta-item {
        font-size: 0.88rem;
    }
    .meta-label {
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.75rem;
        margin-bottom: 0.2rem;
    }
    .meta-value {
        color: #0F172A;
        font-weight: 700;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0 1.5rem 0;
        font-size: 0.9rem;
    }
    .report-table th {
        background-color: #1E3A8A;
        color: #FFFFFF;
        text-align: left;
        padding: 0.65rem 0.85rem;
        font-weight: 600;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .report-table td {
        padding: 0.65rem 0.85rem;
        border-bottom: 1px solid #E2E8F0;
        color: #334155;
    }
    .report-table tr:nth-child(even) {
        background-color: #F8FAFC;
    }
    .action-box {
        background-color: #EFF6FF;
        border-left: 4px solid #2563EB;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.8rem 0;
    }
    .action-title {
        font-weight: 700;
        color: #1E3A8A;
        font-size: 0.92rem;
        margin-bottom: 0.25rem;
    }
    .disclaimer-box {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 2rem;
        font-size: 0.8rem;
        color: #475569;
        line-height: 1.5;
    }
    .report-footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #E2E8F0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #94A3B8;
        font-size: 0.78rem;
    }
</style>
"""
