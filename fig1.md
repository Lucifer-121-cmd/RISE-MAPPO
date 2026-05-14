# RISE-MAPPO Architecture Diagram — Final Prompt

Generate a clean, publication-quality SVG system architecture diagram for an IEEE robotics journal paper. The diagram must be legible when printed at 18cm wide in a two-column PDF.

## DESIGN PHILOSOPHY

- **Fewer arrows, more layout.** Hierarchy is shown by vertical stacking, not by drawing every connection.
- **Maximum 8 arrows total** in the entire diagram. Every arrow must be readable without zooming.
- **No arrow may cross any text.** If an arrow would cross text, reroute it or remove it.
- **White space is your friend.** Leave generous gaps between components.
- **No decorative elements.** No icons, no heatmaps, no grid patterns, no robot illustrations.

## EXACT LAYOUT SPECIFICATION

Canvas: 1200 × 900 px, white background. Three horizontal bands separated by clear gaps.

### BAND 1 (y: 0–280): RISE-MAPPO Planning Layer
- Light blue background (#E8F0FE), 1px border
- Header bar: "RISE-MAPPO Planning Layer" (left) + "Every Kₛ = 25 steps" (right)
- Contains 3 boxes arranged LEFT → RIGHT with gray arrows between them:

```
┌─────────────┐       ┌─────────────────────────┐       ┌──────────────────────────────┐
│ Shared Actor │ ───→  │ GP-Uncertainty Attention │ ───→  │     Dual-Head Critic         │
│     πθ       │       │                         │       │  ┌────────┐  ┌─────────┐     │
│             │       │ wᵢ ∝ exp(qᵀhᵢ·(1+ησ̃ᵢ)) │       │  │ V_mean │  │ V_CVaR  │     │
│  aᵢ∈{0…K-1}│       │                         │       │  └───┬────┘  └────┬────┘     │
└─────────────┘       └─────────────────────────┘       │      └─────┬──────┘          │
                                                         │  A = A_mean − λ·A_risk       │
                                                         └──────────────────────────────┘
```

- "Local Obs oᵢ" label with small arrow entering Actor from the left
- "Global State s" label with small arrow entering Critic from above
- **C1 NOVEL**: Dashed blue (#1565C0) rounded rectangle enclosing ONLY the GP-Attention box and the Dual-Head Critic box. "C1: Novel" badge sits on top of this dashed border — NOT overlapping the border line. Place the badge ABOVE the dashed line with a white background so it visually "interrupts" the dashed border cleanly.
- V_mean box: blue fill (#DDE9FB), blue border
- V_CVaR box: orange fill (#FFE0B2), orange border
- Advantage formula in dark bar below the two heads

### BAND 2 (y: 320–520): Per-Robot Lyapunov-MPC Layer
- Light green background (#E8F5E9), 1px border
- Header bar: "Per-Robot Lyapunov-MPC" (left) + "Every Δt = 0.1s" (right)
- Contains 3 identical boxes side by side, evenly spaced:

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ Robot 1           │  │ Robot 2           │  │ Robot 3           │
│ MPC (CasADi/IPOPT)│ │ MPC (CasADi/IPOPT)│ │ MPC (CasADi/IPOPT)│
│                    │  │                    │  │                    │
│ V(x_{k+1}) ≤      │  │ V(x_{k+1}) ≤      │  │ V(x_{k+1}) ≤      │
│ (1-αL)V(xk)       │  │ (1-αL)V(xk)       │  │ (1-αL)V(xk)       │
│                    │  │                    │  │                    │
│ uᵢ = [vᵢ, ωᵢ]    │  │ uᵢ = [vᵢ, ωᵢ]    │  │ uᵢ = [vᵢ, ωᵢ]    │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

- "Robot N" as a small colored pill badge (green) in top-left of each box
- "MPC (CasADi/IPOPT)" as the box title
- Lyapunov formula centered
- Output "uᵢ = [vᵢ, ωᵢ]" in a pill at the bottom

### BAND 3 (y: 560–760): Distributed GP + BCM Fusion Layer
- Light purple background (#F3E5F5), 1px border
- Header bar: "Distributed GP with BCM Fusion" (left) + "Continuous" (right)
- Left side: 3 small boxes "Local GP₁", "Local GP₂", "Local GP₃", each showing only "(μᵢ*, σᵢ*)"
- Right side: 1 larger box "BCM Fusion" with formula "σ⁻²_BCM = Σσᵢ⁻² − (N−1)σ⁻²_prior"
- Small gray arrows from each Local GP → BCM Fusion box

### BOTTOM STRIP (y: 800–870): Environment
- Light gray background
- "Multi-Robot Search Environment" title
- Simple text labels: "3 robots · 5 targets · explored/unexplored regions · hazard zones"
- NO grid, NO robot icons, NO star icons — just the text description

## ARROWS (EXACTLY 8, NO MORE)

Only these arrows exist in the entire diagram. Each arrow has a WHITE-BACKGROUND PILL label so text never touches the line.

| # | From | To | Color | Label | Style |
|---|------|----|-------|-------|-------|
| 1 | Actor bottom | Horizontal bus splitting to 3 MPC tops | Blue (#0D47A1), 2px | "Subgoals xg,i" | Solid, arrow down |
| 2 | 3 MPC bottoms | merging to horizontal bus → Env top | Blue (#0D47A1), 2px | "Controls uᵢ = [vᵢ, ωᵢ]" | Solid, arrow down |
| 3 | Env top | splitting to 3 Local GP tops | Orange (#E65100), 1.5px | "Sensor data" | Solid, arrow up |
| 4 | BCM right edge | UP along right margin → into Critic right edge | Orange (#E65100), 1.5px | "σ_BCM, CVaR cᵗ" | Dashed, arrow up |
| 5 | BCM top | UP to MPC layer (horizontal bus to 3 MPCs) | Orange (#E65100), 1.5px | "Obstacle beliefs" | Solid, arrow up |
| 6 | Left of diagram | Actor left edge | Gray, 1px | "Local Obs oᵢ" | Solid |
| 7 | Top of Critic | from above into Critic | Gray, 1px | "Global State s" | Solid |
| 8 | GP-Attention right → Critic left | Gray, 1px | "Attended features" | Solid |

### ARROW ROUTING RULES:
- Arrow #4 (the critical feedback loop) routes along the RIGHT MARGIN of the diagram with 20px padding from the edge. It goes: BCM right edge → straight up along x≈1170 → turns left into Critic. This arrow must have CLEAR space — no other element within 15px.
- Arrow #1 uses a "bus" pattern: single line down from Actor, then horizontal bar, then 3 drops into MPCs.
- Arrow #2 uses reverse bus: 3 lines up from MPCs to horizontal bar, then single line down to Env.
- Arrow #5 uses same bus pattern but going UP from BCM through the gap between Layer 2 and Layer 3.
- NO arrows cross ANY box or text. Route around, not through.
- Every arrow label is inside a white pill (white fill, thin border matching arrow color).

## STYLING

### Fonts:
```css
.header { font: bold 15px 'Helvetica Neue', Arial, sans-serif; fill: #1F2A44; }
.subheader { font: 11px 'Helvetica Neue', Arial, sans-serif; fill: #5A6378; }
.boxTitle { font: bold 12px 'Helvetica Neue', Arial, sans-serif; fill: #1F2A44; }
.formula { font: italic 13px 'Georgia', 'Times New Roman', serif; fill: #1F2A44; }
.label { font: 10px 'Helvetica Neue', Arial, sans-serif; fill: #3F4A66; }
.pill { font: 10px 'Helvetica Neue', Arial, sans-serif; }
.badge { font: bold 10px 'Helvetica Neue', Arial, sans-serif; fill: white; }
```

### Colors:
- Layer 1 bg: #E8F0FE, border: #5B7FBE
- Layer 2 bg: #E8F5E9, border: #6FA56F
- Layer 3 bg: #F3E5F5, border: #A07AB0
- Environment bg: #F5F5F5, border: #B0B0B0
- V_mean box: fill #DDE9FB, border #1565C0
- V_CVaR box: fill #FFE0B2, border #E65100
- Advantage bar: fill #1F2A44 (dark), text white
- C1 dashed border: #1565C0, 1.5px, dash "6 4"
- C1 badge: solid fill #1565C0, white text, rounded pill
- Command arrows (down): #0D47A1, 2px solid
- Feedback arrows (up): #E65100, 1.5px (solid or dashed as specified)
- Internal arrows: #3F4A66, 1px

### Box style:
- All boxes: white fill, rounded corners (rx=6), 1px border
- Title area: slightly darker tint at top of box (like a card header)
- No shadows, no gradients, no 3D effects

## FINAL CHECKLIST — MUST ALL BE TRUE

- [ ] Total arrow count ≤ 8
- [ ] No line crosses any text anywhere in the diagram
- [ ] No text is clipped at any edge
- [ ] C1 badge sits cleanly on the dashed border without overlap
- [ ] Right-margin feedback arrow has ≥15px clearance from diagram edge
- [ ] All arrow labels are in white pills
- [ ] Minimum 40px vertical gap between each layer band
- [ ] All three MPC boxes are identical width and evenly spaced
- [ ] V_mean and V_CVaR boxes are clearly color-coded (blue vs orange)
- [ ] Advantage formula bar is clearly readable: A = A_mean − λ · A_risk
- [ ] BCM formula is readable: σ⁻²_BCM = Σσᵢ⁻² − (N−1)σ⁻²_prior
- [ ] GP-Attention formula is readable: wᵢ ∝ exp(qᵀhᵢ · (1 + η σ̃ᵢ))
- [ ] No decorative elements (no robot icons, no heatmaps, no grid patterns)
- [ ] Legible at 18cm print width (minimum 9px text, prefer 10-12px)
- [ ] Looks professional enough for IEEE Transactions or RA-L

Generate this as a clean SVG.